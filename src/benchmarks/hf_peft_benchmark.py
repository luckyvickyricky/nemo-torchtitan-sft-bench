"""HuggingFace Transformers + PEFT 벤치마크"""
import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tulu_loader import TuluDataLoader, create_data_collator
from utils.metrics import BenchmarkMetrics, count_consumed_tokens


class BenchmarkTrainer(Trainer):
    """메트릭 추적을 위한 커스텀 Trainer"""
    
    def __init__(self, *args, metrics_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_tracker = metrics_tracker
        self.pad_token_id = self.processing_class.pad_token_id
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """훈련 스텝 오버라이드"""
        if self.metrics_tracker:
            self.metrics_tracker.start_step()
        
        # 실제 토큰 수 계산 (패딩 제외)
        input_ids = inputs.get("input_ids")
        consumed_tokens = count_consumed_tokens(input_ids, self.pad_token_id)
        
        # 원래 training step 실행
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if self.metrics_tracker:
            # 메트릭 기록
            metrics = self.metrics_tracker.end_step(
                consumed_tokens=consumed_tokens,
                global_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size,
                max_seq_len=self.processing_class.model_max_length,
                step=self.state.global_step,
            )
            
            if metrics and self.state.global_step % 10 == 0:
                print(f"Step {self.state.global_step}: TPS={metrics['tps_consumed']:.2f}, "
                      f"Step Time={metrics['step_time_sec']:.4f}s, "
                      f"Pack Density={metrics['pack_density']:.4f}")
        
        return loss


def setup_quantization_config(use_qlora: bool):
    """양자화 설정"""
    if use_qlora:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="HF Transformers + PEFT Benchmark")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="모델 이름")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="시퀀스 길이")
    parser.add_argument("--use_qlora", action="store_true",
                        help="QLoRA 사용 (NF4 + BF16)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="최대 학습 스텝")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="사용할 데이터 샘플 수")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="로그 디렉토리")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 실험 이름 생성
    method = "qlora" if args.use_qlora else "lora"
    model_short = args.model_name.split("/")[-1]
    experiment_name = f"hf_{method}_{model_short}_seq{args.seq_length}"
    
    print(f"\n{'='*80}")
    print(f"실험: {experiment_name}")
    print(f"모델: {args.model_name}")
    print(f"방법: {method.upper()}")
    print(f"시퀀스 길이: {args.seq_length}")
    print(f"배치 크기: {args.batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"글로벌 배치 크기: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"{'='*80}\n")
    
    # 메트릭 트래커 초기화
    metrics_tracker = BenchmarkMetrics(args.log_dir, experiment_name)
    
    # 토크나이저 로드
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = args.seq_length
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    data_loader = TuluDataLoader(
        max_seq_length=args.seq_length,
        num_samples=args.num_samples,
    )
    train_dataset = data_loader.load_and_prepare(tokenizer)
    
    # 모델 로드
    print("모델 로드 중...")
    quantization_config = setup_quantization_config(args.use_qlora)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not args.use_qlora else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        use_cache=False,
    )
    
    # QLoRA의 경우 모델 준비
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    print(f"LoRA 설정 적용 중 (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Gradient checkpointing 호환성을 위해 input gradients 활성화
    model.enable_input_require_grads()
    
    # 트레이닝 인자
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, experiment_name),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        warmup_steps=10,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # 데이터 콜레이터
    data_collator = create_data_collator(tokenizer)
    
    # Trainer 생성
    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        metrics_tracker=metrics_tracker,
    )
    
    # 학습 시작
    print("\n학습 시작...\n")
    trainer.train()
    
    # 요약 통계
    print("\n" + "="*80)
    print("벤치마크 완료!")
    summary = metrics_tracker.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*80 + "\n")
    
    print(f"로그 저장: {metrics_tracker.log_file}")
    print(f"요약 저장: {metrics_tracker.log_file.replace('.jsonl', '_summary.json')}")


if __name__ == "__main__":
    main()

