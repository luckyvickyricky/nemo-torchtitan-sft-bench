#!/bin/bash
# CUDA 13.0 전용 nightly build 설치 스크립트

uv pip uninstall torch torchvision torchaudio || true

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

uv pip install bitsandbytes

# ninja 증폭 OOM방지 세팅
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

uv run python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
print(f'CUDA 버전: {torch.version.cuda}')
print(f'cuDNN 버전: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU 개수: {torch.cuda.device_count()}')
    print(f'GPU 이름: {torch.cuda.get_device_name(0)}')
    print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0)}')
"

