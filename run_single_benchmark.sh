#!/bin/bash

# 단일 벤치마크 실행 스크립트 (빠른 테스트용)

set -e

echo "========================================================================"
echo "단일 벤치마크 실행 (테스트용)"
echo "========================================================================"

# 환경 활성화
source .venv/bin/activate

# 로그 및 결과 디렉토리 생성
mkdir -p logs results

# 기본 설정
MODEL="${MODEL:-Qwen/Qwen3-4B}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
USE_QLORA="${USE_QLORA:-false}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-50}"

echo ""
echo "설정:"
echo "- 모델: ${MODEL}"
echo "- 시퀀스 길이: ${SEQ_LENGTH}"
echo "- QLoRA: ${USE_QLORA}"
echo "- 배치 크기: ${BATCH_SIZE}"
echo "- Gradient Accumulation: ${GRAD_ACCUM}"
echo "- 최대 스텝: ${MAX_STEPS}"
echo ""

# GPU 정보
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 실행 명령 구성
CMD="python src/benchmarks/hf_peft_benchmark.py \
    --model_name ${MODEL} \
    --seq_length ${SEQ_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps ${MAX_STEPS} \
    --num_samples 5000 \
    --log_dir logs \
    --output_dir results"

if [ "${USE_QLORA}" = "true" ]; then
    CMD="${CMD} --use_qlora"
fi

echo "실행 명령:"
echo "${CMD}"
echo ""

# 실행
eval ${CMD}

echo ""
echo "벤치마크 완료"
echo "로그 확인: logs/"
echo ""

