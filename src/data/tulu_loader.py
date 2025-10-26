"""Tülu-3 SFT mixture 데이터 로더"""
import os
from typing import Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer


class TuluDataLoader:
    """Tülu-3 SFT mixture 데이터셋 로더"""
    
    # Tülu-3 SFT mixture 구성 요소
    # https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
    TULU_3_DATASETS = [
        "allenai/tulu-3-sft-mixture",
    ]
    
    def __init__(
        self,
        max_seq_length: int = 2048,
        num_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            max_seq_length: 최대 시퀀스 길이
            num_samples: 사용할 샘플 수 (None이면 전체)
            cache_dir: 캐시 디렉토리
        """
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        
    def load_and_prepare(self, tokenizer: PreTrainedTokenizer, split: str = "train") -> Dataset:
        """
        데이터셋 로드 및 전처리
        
        Args:
            tokenizer: 토크나이저
            split: 데이터셋 스플릿 ("train", "validation")
            
        Returns:
            tokenized_dataset: 토크나이즈된 데이터셋
        """
        print(f"Loading Tülu-3 SFT mixture from Hugging Face...")
        
        # Tülu-3 SFT mixture 로드
        dataset = load_dataset(
            "allenai/tulu-3-sft-mixture",
            split=split,
            cache_dir=self.cache_dir,
        )
        
        # 샘플 수 제한
        if self.num_samples is not None and self.num_samples < len(dataset):
            dataset = dataset.select(range(self.num_samples))
            
        print(f"Loaded {len(dataset)} examples")
        
        # 토크나이즈
        def tokenize_function(examples):
            """대화를 토크나이즈"""
            texts = []
            
            for messages in examples["messages"]:
                # 대화 형식을 텍스트로 변환
                if isinstance(messages, list):
                    # Chat template 적용
                    text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                else:
                    texts.append(str(messages))
            
            # 토크나이즈
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            
            # Labels 설정 (causal LM용)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        
        print(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset


def create_data_collator(tokenizer: PreTrainedTokenizer):
    """데이터 콜레이터 생성"""
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

