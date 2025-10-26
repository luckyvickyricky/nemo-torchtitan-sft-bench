"""메트릭 계산 유틸리티"""
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class BenchmarkMetrics:
    """벤치마크 메트릭 계산 및 로깅"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = None
        self.step_times = []
        self.token_counts = []
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{timestamp}.jsonl"
        )
        
    def start_step(self):
        """스텝 시작 시간 기록"""
        self.start_time = time.perf_counter()
        
    def end_step(self, consumed_tokens: int, global_batch_size: int, 
                 max_seq_len: int, step: int):
        """
        스텝 종료 및 메트릭 계산
        
        Args:
            consumed_tokens: 실제 토큰 수 (패딩 제외)
            global_batch_size: 글로벌 배치 사이즈
            max_seq_len: 최대 시퀀스 길이
            step: 현재 스텝 번호
        """
        if self.start_time is None:
            return
            
        step_time = time.perf_counter() - self.start_time
        self.step_times.append(step_time)
        self.token_counts.append(consumed_tokens)
        
        # TPS (consumed tokens per second)
        tps = consumed_tokens / step_time if step_time > 0 else 0
        
        # Pack density
        max_possible_tokens = global_batch_size * max_seq_len
        pack_density = consumed_tokens / max_possible_tokens if max_possible_tokens > 0 else 0
        
        # GPU 메모리 사용량
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        metrics = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "step_time_sec": round(step_time, 4),
            "consumed_tokens": consumed_tokens,
            "tps_consumed": round(tps, 2),
            "pack_density": round(pack_density, 4),
            "gpu_memory_allocated_gb": round(gpu_memory_allocated, 2),
            "gpu_memory_reserved_gb": round(gpu_memory_reserved, 2),
        }
        
        # JSONL 형식으로 로그 저장
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.start_time = None
        return metrics
        
    def get_summary(self) -> Dict[str, Any]:
        """전체 실행 요약 통계"""
        if not self.step_times:
            return {}
            
        import numpy as np
        
        total_tokens = sum(self.token_counts)
        total_time = sum(self.step_times)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.step_times),
            "total_consumed_tokens": total_tokens,
            "total_time_sec": round(total_time, 2),
            "avg_tps_consumed": round(avg_tps, 2),
            "avg_step_time_sec": round(np.mean(self.step_times), 4),
            "std_step_time_sec": round(np.std(self.step_times), 4),
            "min_step_time_sec": round(np.min(self.step_times), 4),
            "max_step_time_sec": round(np.max(self.step_times), 4),
        }
        
        # 요약 저장
        summary_file = self.log_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary


def count_consumed_tokens(input_ids: torch.Tensor, pad_token_id: Optional[int] = None) -> int:
    """
    패딩을 제외한 실제 토큰 수 계산
    
    Args:
        input_ids: 토큰 ID 텐서 [batch_size, seq_len]
        pad_token_id: 패딩 토큰 ID
        
    Returns:
        consumed_tokens: 실제 토큰 수
    """
    if pad_token_id is not None:
        mask = (input_ids != pad_token_id)
        return mask.sum().item()
    else:
        return input_ids.numel()

