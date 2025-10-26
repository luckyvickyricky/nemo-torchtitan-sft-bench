# RTX 5090 / DGX Spark Speed Bench


## Metric

- **TPS (Tokens Per Second)**: 실제 토큰 수(패딩 제외) ÷ step wall-time
- **Step Time**: 옵티마이저 갱신 단위의 시간 (Gradient Accumulation 포함)
- **Pack Density**: 실제 토큰 ÷ (Global Batch Size × Max Seq Length)
- **GPU Memory Usage**: 할당 및 예약된 메모리

## Quick start

### 1. 환경 설정

```bash
./setup.sh
./setup_torch_nightly.sh
```

### 2. 벤치마크 실행

```bash
./run_hf_benchmark.sh
./run_single_benchmark.sh
```

```bash
logs/
├── hf_lora_Qwen3-4B_seq2048_20241025_123456.jsonl        # 상세 로그
├── hf_lora_Qwen3-4B_seq2048_20241025_123456_summary.json # 요약 통계
├── hf_qlora_Qwen3-4B_seq2048_20241025_123457.jsonl
└── hf_qlora_Qwen3-4B_seq2048_20241025_123457_summary.json
```
