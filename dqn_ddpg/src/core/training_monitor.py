"""
학습 진행 모니터링 시스템
실시간 지표 수집, GPU 사용량 추적, 학습 안정성 감지를 담당합니다.
"""

import time
import json
import psutil
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import torch

from .utils import get_gpu_memory_info


@dataclass
class TrainingSnapshot:
    """학습 스냅샷 정보"""
    timestamp: float
    episode_id: int
    step: int
    
    # 성능 지표
    reward: float
    loss: float
    q_values: Optional[List[float]] = None
    policy_entropy: Optional[float] = None
    
    # 시스템 지표
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_temperature: float = 0.0
    
    # 학습 안정성
    gradient_norm: Optional[float] = None
    learning_rate: float = 0.0
    exploration_rate: float = 0.0
    
    # 추가 메트릭
    extra_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.extra_metrics is None:
            self.extra_metrics = {}


class PerformanceAlert:
    """성능 경고 시스템"""
    
    def __init__(self):
        self.alerts = []
        self.alert_history = []
        
    def check_training_stability(self, recent_losses: deque, threshold: float = 10.0) -> bool:
        """학습 안정성 확인 (손실 발산 감지)"""
        if len(recent_losses) < 10:
            return True
        
        recent_avg = np.mean(list(recent_losses)[-10:])
        early_avg = np.mean(list(recent_losses)[:10])
        
        if recent_avg > early_avg * threshold:
            self.add_alert("LOSS_DIVERGENCE", f"손실이 {threshold}배 증가했습니다")
            return False
        
        return True
    
    def check_reward_plateau(self, recent_rewards: deque, window: int = 100, 
                           min_improvement: float = 0.01) -> bool:
        """보상 정체 감지"""
        if len(recent_rewards) < window * 2:
            return True
        
        rewards_list = list(recent_rewards)
        first_half = np.mean(rewards_list[-window*2:-window])
        second_half = np.mean(rewards_list[-window:])
        
        improvement = (second_half - first_half) / abs(first_half) if first_half != 0 else 0
        
        if improvement < min_improvement:
            self.add_alert("REWARD_PLATEAU", f"보상 개선이 {min_improvement*100:.1f}% 미만입니다")
            return False
        
        return True
    
    def check_gpu_memory(self, gpu_usage_percent: float, threshold: float = 95.0) -> bool:
        """GPU 메모리 사용량 확인"""
        if gpu_usage_percent > threshold:
            self.add_alert("GPU_MEMORY_HIGH", f"GPU 메모리 사용량이 {gpu_usage_percent:.1f}%입니다")
            return False
        
        return True
    
    def add_alert(self, alert_type: str, message: str):
        """경고 추가"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time()
        }
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        print(f"⚠️ {alert_type}: {message}")
    
    def get_active_alerts(self) -> List[Dict]:
        """활성 경고 목록 반환"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """경고 초기화"""
        self.alerts.clear()


class TrainingMonitor:
    """학습 진행 모니터링 시스템
    
    실시간으로 학습 진행 상황을 모니터링하고 성능 지표를 수집합니다.
    """
    
    def __init__(self, 
                 algorithm: str,
                 log_dir: str = "logs",
                 monitoring_interval: float = 1.0,
                 enable_gpu_monitoring: bool = True,
                 enable_alerts: bool = True):
        """
        Args:
            algorithm: 알고리즘 이름
            log_dir: 로그 저장 디렉토리
            monitoring_interval: 모니터링 간격 (초)
            enable_gpu_monitoring: GPU 모니터링 활성화
            enable_alerts: 경고 시스템 활성화
        """
        self.algorithm = algorithm.lower()
        self.log_dir = Path(log_dir)
        self.monitoring_interval = monitoring_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 경로
        self.training_log_file = self.log_dir / f"{self.algorithm}_training_monitor.json"
        self.system_log_file = self.log_dir / f"{self.algorithm}_system_monitor.json"
        
        # 데이터 저장소
        self.training_snapshots: List[TrainingSnapshot] = []
        self.system_metrics = defaultdict(deque)
        
        # 실시간 통계
        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        self.recent_q_values = deque(maxlen=100)
        
        # 모니터링 상태
        self.monitoring_active = False
        self.monitoring_thread = None
        self.start_time = None
        
        # 경고 시스템
        self.alert_system = PerformanceAlert() if enable_alerts else None
        
        # 커스텀 메트릭 콜백
        self.custom_metric_callbacks: List[Callable] = []
        
        print(f"📊 학습 모니터 초기화: {self.algorithm}")
        print(f"📁 로그 디렉토리: {self.log_dir}")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        
        if self.enable_gpu_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("🔍 시스템 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        # 최종 로그 저장
        self._save_logs()
        print("📊 모니터링 중지 및 로그 저장 완료")
    
    def _monitoring_loop(self):
        """모니터링 루프 (별도 스레드에서 실행)"""
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집
                self._collect_system_metrics()
                
                # 경고 확인
                if self.alert_system:
                    self._check_alerts()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"⚠️ 모니터링 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        timestamp = time.time()
        
        # CPU 및 메모리 사용량
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage_mb = memory_info.used / (1024 * 1024)
        
        # GPU 메트릭
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_temperature = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_info = get_gpu_memory_info()
                gpu_memory_used = gpu_info.get('memory_used_gb', 0.0)
                gpu_memory_total = gpu_info.get('memory_total_gb', 0.0)
                gpu_temperature = gpu_info.get('temperature', 0.0)
            except Exception:
                pass
        
        # 메트릭 저장
        self.system_metrics['timestamp'].append(timestamp)
        self.system_metrics['cpu_usage'].append(cpu_usage)
        self.system_metrics['memory_usage_mb'].append(memory_usage_mb)
        self.system_metrics['gpu_memory_used'].append(gpu_memory_used)
        self.system_metrics['gpu_memory_total'].append(gpu_memory_total)
        self.system_metrics['gpu_temperature'].append(gpu_temperature)
        
        # 메모리 효율성을 위해 오래된 데이터 제거
        max_length = 10000
        for key, values in self.system_metrics.items():
            if len(values) > max_length:
                for _ in range(len(values) - max_length):
                    values.popleft()
    
    def record_training_step(self, 
                           episode_id: int,
                           step: int,
                           reward: float,
                           loss: float,
                           q_values: Optional[List[float]] = None,
                           learning_rate: float = 0.0,
                           exploration_rate: float = 0.0,
                           **extra_metrics):
        """학습 스텝 기록
        
        Args:
            episode_id: 에피소드 ID
            step: 스텝 번호
            reward: 보상
            loss: 손실값
            q_values: Q값 리스트
            learning_rate: 학습률
            exploration_rate: 탐험률
            **extra_metrics: 추가 메트릭
        """
        timestamp = time.time()
        
        # 최근 값들 업데이트
        self.recent_rewards.append(reward)
        if loss > 0:
            self.recent_losses.append(loss)
        if q_values:
            self.recent_q_values.append(np.mean(q_values))
        
        # 현재 시스템 상태
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage_mb = memory_info.used / (1024 * 1024)
        
        # GPU 정보
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if torch.cuda.is_available():
            try:
                gpu_info = get_gpu_memory_info()
                gpu_memory_used = gpu_info.get('memory_used_gb', 0.0)
                gpu_memory_total = gpu_info.get('memory_total_gb', 0.0)
            except Exception:
                pass
        
        # 정책 엔트로피 계산 (Q값이 있는 경우)
        policy_entropy = None
        if q_values and len(q_values) > 1:
            q_array = np.array(q_values)
            # Softmax 적용
            exp_q = np.exp(q_array - np.max(q_array))
            prob = exp_q / np.sum(exp_q)
            # 엔트로피 계산
            policy_entropy = -np.sum(prob * np.log(prob + 1e-10))
        
        # 스냅샷 생성
        snapshot = TrainingSnapshot(
            timestamp=timestamp,
            episode_id=episode_id,
            step=step,
            reward=reward,
            loss=loss,
            q_values=q_values,
            policy_entropy=policy_entropy,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            extra_metrics=extra_metrics
        )
        
        self.training_snapshots.append(snapshot)
        
        # 커스텀 메트릭 콜백 실행
        for callback in self.custom_metric_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"⚠️ 커스텀 메트릭 콜백 오류: {e}")
        
        # 주기적으로 로그 저장
        if len(self.training_snapshots) % 100 == 0:
            self._save_logs()
    
    def _check_alerts(self):
        """경고 상태 확인"""
        if not self.alert_system:
            return
        
        # 학습 안정성 확인
        self.alert_system.check_training_stability(self.recent_losses)
        
        # 보상 정체 확인
        self.alert_system.check_reward_plateau(self.recent_rewards)
        
        # GPU 메모리 확인
        if self.system_metrics['gpu_memory_used'] and self.system_metrics['gpu_memory_total']:
            latest_used = list(self.system_metrics['gpu_memory_used'])[-1]
            latest_total = list(self.system_metrics['gpu_memory_total'])[-1]
            if latest_total > 0:
                usage_percent = (latest_used / latest_total) * 100
                self.alert_system.check_gpu_memory(usage_percent)
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        stats = {
            'algorithm': self.algorithm,
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'total_snapshots': len(self.training_snapshots),
            'recent_reward_stats': {},
            'recent_loss_stats': {},
            'system_stats': {}
        }
        
        # 보상 통계
        if self.recent_rewards:
            rewards_array = np.array(list(self.recent_rewards))
            stats['recent_reward_stats'] = {
                'mean': float(np.mean(rewards_array)),
                'std': float(np.std(rewards_array)),
                'min': float(np.min(rewards_array)),
                'max': float(np.max(rewards_array)),
                'count': len(self.recent_rewards)
            }
        
        # 손실 통계
        if self.recent_losses:
            losses_array = np.array(list(self.recent_losses))
            stats['recent_loss_stats'] = {
                'mean': float(np.mean(losses_array)),
                'std': float(np.std(losses_array)),
                'min': float(np.min(losses_array)),
                'max': float(np.max(losses_array)),
                'count': len(self.recent_losses)
            }
        
        # 시스템 통계
        if self.system_metrics['cpu_usage']:
            cpu_values = list(self.system_metrics['cpu_usage'])[-100:]  # 최근 100개
            memory_values = list(self.system_metrics['memory_usage_mb'])[-100:]
            
            stats['system_stats'] = {
                'avg_cpu_usage': float(np.mean(cpu_values)),
                'avg_memory_usage_mb': float(np.mean(memory_values)),
            }
            
            if self.system_metrics['gpu_memory_used']:
                gpu_used = list(self.system_metrics['gpu_memory_used'])[-100:]
                gpu_total = list(self.system_metrics['gpu_memory_total'])[-100:]
                if gpu_total and gpu_total[-1] > 0:
                    stats['system_stats']['avg_gpu_usage_percent'] = float(np.mean(gpu_used) / gpu_total[-1] * 100)
        
        # 경고 정보
        if self.alert_system:
            stats['active_alerts'] = self.alert_system.get_active_alerts()
            stats['alert_count'] = len(self.alert_system.alert_history)
        
        return stats
    
    def add_custom_metric_callback(self, callback: Callable[[TrainingSnapshot], None]):
        """커스텀 메트릭 콜백 추가"""
        self.custom_metric_callbacks.append(callback)
    
    def _save_logs(self):
        """로그 파일 저장"""
        try:
            # 학습 스냅샷 저장
            training_data = {
                'algorithm': self.algorithm,
                'start_time': self.start_time,
                'last_updated': time.time(),
                'snapshots': [asdict(snapshot) for snapshot in self.training_snapshots[-1000:]]  # 최근 1000개만
            }
            
            with open(self.training_log_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # 시스템 메트릭 저장
            system_data = {
                'algorithm': self.algorithm,
                'last_updated': time.time(),
                'metrics': {
                    key: list(values)[-1000:] for key, values in self.system_metrics.items()  # 최근 1000개만
                }
            }
            
            with open(self.system_log_file, 'w') as f:
                json.dump(system_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ 로그 저장 오류: {e}")
    
    def export_summary_report(self, output_path: str = None) -> str:
        """요약 리포트 내보내기"""
        if output_path is None:
            output_path = self.log_dir / f"{self.algorithm}_training_summary.json"
        
        # 통계 계산
        stats = self.get_current_statistics()
        
        # 에피소드별 통계
        episode_stats = defaultdict(list)
        for snapshot in self.training_snapshots:
            episode_stats[snapshot.episode_id].append(snapshot)
        
        episode_summary = []
        for episode_id, snapshots in episode_stats.items():
            episode_rewards = [s.reward for s in snapshots]
            episode_losses = [s.loss for s in snapshots if s.loss > 0]
            
            summary = {
                'episode_id': episode_id,
                'total_steps': len(snapshots),
                'total_reward': sum(episode_rewards),
                'avg_reward_per_step': np.mean(episode_rewards),
                'avg_loss': np.mean(episode_losses) if episode_losses else 0,
                'duration': snapshots[-1].timestamp - snapshots[0].timestamp if snapshots else 0
            }
            episode_summary.append(summary)
        
        # 리포트 생성
        report = {
            'algorithm': self.algorithm,
            'generation_time': time.time(),
            'overall_stats': stats,
            'episode_summary': episode_summary,
            'performance_trends': self._calculate_performance_trends()
        }
        
        # 파일 저장
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 요약 리포트 저장: {output_path}")
        return str(output_path)
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 계산"""
        if len(self.recent_rewards) < 20:
            return {}
        
        rewards = list(self.recent_rewards)
        recent_half = rewards[len(rewards)//2:]
        early_half = rewards[:len(rewards)//2]
        
        trend = {
            'reward_trend': 'improving' if np.mean(recent_half) > np.mean(early_half) else 'declining',
            'reward_improvement_rate': (np.mean(recent_half) - np.mean(early_half)) / abs(np.mean(early_half)) if np.mean(early_half) != 0 else 0,
            'convergence_indicator': np.std(recent_half) / np.mean(recent_half) if np.mean(recent_half) != 0 else float('inf')
        }
        
        return trend