#!/bin/bash

# HF Transformers + PEFT 벤치마크 실행 스크립트
# DGX Spark RTX 5090 성능 벤치마크

set -e  # 에러 발생시 중단

echo "========================================================================"
echo "DGX Spark RTX 5090 - HF Transformers + PEFT 벤치마크"
echo "========================================================================"

# 환경 활성화
source .venv/bin/activate

# 로그 및 결과 디렉토리 생성
mkdir -p logs results

# 벤치마크 설정
MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
)

SEQ_LENGTHS=(512 1024 2048 4096 8192)

# 하이퍼파라미터
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
MAX_STEPS=100
NUM_SAMPLES=10000
LORA_R=64
LORA_ALPHA=16

echo ""
echo "벤치마크 설정:"
echo "- 모델: ${MODELS[@]}"
echo "- 시퀀스 길이: ${SEQ_LENGTHS[@]}"
echo "- 배치 크기: ${BATCH_SIZE}"
echo "- Gradient Accumulation: ${GRAD_ACCUM_STEPS}"
echo "- 글로벌 배치 크기: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "- 최대 스텝: ${MAX_STEPS}"
echo "- LoRA Rank: ${LORA_R}"
echo ""

# GPU 정보 출력
echo "GPU 정보:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# LoRA 벤치마크
echo "========================================================================"
echo "LoRA (BF16) 벤치마크 시작"
echo "========================================================================"

for MODEL in "${MODELS[@]}"; do
    for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
        echo ""
        echo ">>> 실행: LoRA - ${MODEL} - Seq Length: ${SEQ_LEN}"
        echo ""
        
        python src/benchmarks/hf_peft_benchmark.py \
            --model_name "${MODEL}" \
            --seq_length ${SEQ_LEN} \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
            --max_steps ${MAX_STEPS} \
            --num_samples ${NUM_SAMPLES} \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --log_dir logs \
            --output_dir results
        
        echo ""
        echo "✓ 완료: LoRA - ${MODEL} - Seq Length: ${SEQ_LEN}"
        echo ""
        
        # GPU 메모리 정리
        sleep 5
    done
done

# QLoRA 벤치마크
echo "========================================================================"
echo "QLoRA (NF4 + BF16) 벤치마크 시작"
echo "========================================================================"

for MODEL in "${MODELS[@]}"; do
    for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
        echo ""
        echo ">>> 실행: QLoRA - ${MODEL} - Seq Length: ${SEQ_LEN}"
        echo ""
        
        python src/benchmarks/hf_peft_benchmark.py \
            --model_name "${MODEL}" \
            --seq_length ${SEQ_LEN} \
            --use_qlora \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
            --max_steps ${MAX_STEPS} \
            --num_samples ${NUM_SAMPLES} \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --log_dir logs \
            --output_dir results
        
        echo ""
        echo "✓ 완료: QLoRA - ${MODEL} - Seq Length: ${SEQ_LEN}"
        echo ""
        
        # GPU 메모리 정리
        sleep 5
    done
done

echo ""
echo "========================================================================"
echo "모든 벤치마크 완료!"
echo "========================================================================"
echo ""
echo "결과 확인:"
echo "- 로그: logs/"
echo "- 요약: logs/*_summary.json"
echo "- 모델 체크포인트: results/"
echo ""
echo "결과 분석을 위해 다음 명령 실행:"
echo "  python scripts/analyze_results.py"
echo ""

