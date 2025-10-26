#!/bin/bash

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "uv 설치 하세요"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source \$HOME/.local/bin/env"
    exit 1
fi


uv python install 3.10

uv sync

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130


echo ""
echo "다음으로 아래 명령실행(torch nightly build)"
echo "  ./install_pytorch_gpu.sh"
echo ""
echo "- 모델 캐시: ~/.cache/huggingface/hub/"
echo "- 데이터셋 캐시: ~/.cache/huggingface/datasets/"
echo "- 로그: ./logs/"
echo "- 결과: ./results/"
echo ""
echo "1. 전체 벤치마크 (모든 모델 + 모든 시퀀스 길이):"
echo "   ./run_hf_benchmark.sh"
echo ""
echo "2. 단일 벤치마크 (빠른 테스트):"
echo "   ./run_single_benchmark.sh"
echo ""

