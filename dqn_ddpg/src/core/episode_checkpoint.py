"""
에피소드 체크포인트 매니저
모델 저장, 메타데이터 관리, GPU 메모리 최적화를 담당합니다.
"""

import os
import json
import time
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

from .utils import get_device, get_gpu_memory_info


@dataclass
class EpisodeMetrics:
    """에피소드별 메트릭 정보"""
    episode_id: int
    total_reward: float
    episode_length: int
    average_loss: float
    exploration_rate: float  # epsilon for DQN, noise level for DDPG
    timestamp: float
    
    # 성능 지표
    success_rate: float = 0.0
    convergence_score: float = 0.0
    
    # GPU 관련
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    training_time: float = 0.0
    
    # 알고리즘별 추가 메트릭
    extra_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.extra_metrics is None:
            self.extra_metrics = {}


@dataclass
class ModelCheckpoint:
    """모델 체크포인트 정보"""
    episode_id: int
    model_path: str
    metrics: EpisodeMetrics
    model_size_mb: float
    algorithm: str
    config_snapshot: Dict[str, Any]
    
    @property
    def performance_score(self) -> float:
        """성능 점수 계산 (보상 + 수렴도)"""
        return self.metrics.total_reward + (self.metrics.convergence_score * 100)


class EpisodeCheckpointManager:
    """에피소드 체크포인트 관리자
    
    주요 기능:
    - 에피소드별 모델 자동 저장
    - 메타데이터 및 성능 지표 추적
    - GPU 메모리 효율적 저장
    - 체크포인트 정리 및 관리
    """
    
    def __init__(self, 
                 base_path: str = "models",
                 algorithm: str = "dqn",
                 environment: str = "cartpole-v1",  # 환경 이름 추가
                 save_frequency: int = 1,  # 모든 에피소드 저장
                 keep_latest: int = 10000,  # 최대한 많이 보관
                 keep_best: int = 100,  # 최고 성능 모델들도 더 많이 보관
                 auto_cleanup: bool = False):  # 자동 정리 비활성화
        """
        Args:
            base_path: 모델 저장 기본 경로
            algorithm: 알고리즘 이름 (dqn, ddpg)
            environment: 환경 이름 (cartpole-v1, pendulum-v1 등)
            save_frequency: 저장 주기 (에피소드 단위)
            keep_latest: 보관할 최신 체크포인트 수
            keep_best: 보관할 최고 성능 체크포인트 수
            auto_cleanup: 자동 정리 활성화
        """
        self.base_path = Path(base_path)
        self.algorithm = algorithm.lower()
        self.environment = environment.lower()
        self.save_frequency = save_frequency
        self.keep_latest = keep_latest
        self.keep_best = keep_best
        self.auto_cleanup = auto_cleanup
        
        # 환경별 경로 설정
        self.model_dir = self.base_path / self.algorithm / self.environment
        self.metadata_file = self.model_dir / "checkpoint_metadata.json"
        self.training_log = self.model_dir / "training_log.json"
        
        # 디렉토리 생성
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 로드
        self.checkpoints: List[ModelCheckpoint] = []
        self.training_history: List[EpisodeMetrics] = []
        self._load_metadata()
        
        # 통계
        self.total_saved = 0
        self.total_size_mb = 0.0
    
    def _load_metadata(self):
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # 체크포인트 정보 복원
                for checkpoint_data in data.get('checkpoints', []):
                    metrics_data = checkpoint_data['metrics']
                    metrics = EpisodeMetrics(**metrics_data)
                    
                    checkpoint = ModelCheckpoint(
                        episode_id=checkpoint_data['episode_id'],
                        model_path=checkpoint_data['model_path'],
                        metrics=metrics,
                        model_size_mb=checkpoint_data['model_size_mb'],
                        algorithm=checkpoint_data['algorithm'],
                        config_snapshot=checkpoint_data['config_snapshot']
                    )
                    self.checkpoints.append(checkpoint)
                
                # 학습 히스토리 복원
                for history_data in data.get('training_history', []):
                    metrics = EpisodeMetrics(**history_data)
                    self.training_history.append(metrics)
                
                # 통계 복원
                self.total_saved = data.get('total_saved', 0)
                self.total_size_mb = data.get('total_size_mb', 0.0)
                
            except Exception as e:
                print(f"[WARNING] 메타데이터 로드 실패: {e}")
    
    def _save_metadata(self):
        """메타데이터 파일 저장"""
        data = {
            'algorithm': self.algorithm,
            'environment': self.environment,
            'total_saved': self.total_saved,
            'total_size_mb': self.total_size_mb,
            'last_updated': time.time(),
            'checkpoints': [
                {
                    'episode_id': cp.episode_id,
                    'model_path': cp.model_path,
                    'metrics': asdict(cp.metrics),
                    'model_size_mb': cp.model_size_mb,
                    'algorithm': cp.algorithm,
                    'config_snapshot': cp.config_snapshot
                }
                for cp in self.checkpoints
            ],
            'training_history': [
                asdict(metrics) for metrics in self.training_history
            ]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def should_save_checkpoint(self, episode_id: int, metrics: EpisodeMetrics) -> bool:
        """체크포인트 저장 여부 결정"""
        # 모든 에피소드 저장 (save_frequency가 1이므로)
        if episode_id % self.save_frequency == 0:
            return True
        
        # 추가적으로 성능 개선 시에도 저장
        if self.checkpoints:
            best_reward = max(cp.metrics.total_reward for cp in self.checkpoints)
            if metrics.total_reward > best_reward * 1.01:  # 1% 이상 개선 시에도 저장
                return True
        
        return False
    
    def save_checkpoint(self, 
                       episode_id: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       metrics: EpisodeMetrics,
                       config: Dict[str, Any],
                       force_save: bool = False) -> Optional[str]:
        """체크포인트 저장
        
        Args:
            episode_id: 에피소드 ID
            model: 저장할 모델
            optimizer: 옵티마이저
            metrics: 에피소드 메트릭
            config: 설정 스냅샷
            force_save: 강제 저장 여부
            
        Returns:
            저장된 파일 경로 (저장하지 않은 경우 None)
        """
        # 저장 여부 확인
        if not force_save and not self.should_save_checkpoint(episode_id, metrics):
            return None
        
        # GPU 메모리 정보 업데이트
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            metrics.gpu_memory_used = gpu_info.get('memory_used_gb', 0.0)
            metrics.gpu_memory_total = gpu_info.get('memory_total_gb', 0.0)
        
        # 파일 경로 생성
        model_filename = f"episode_{episode_id:04d}.pth"
        model_path = self.model_dir / model_filename
        
        # 모델을 CPU로 이동 (GPU 메모리 절약)
        device = next(model.parameters()).device
        model.cpu()
        
        try:
            # 체크포인트 데이터 준비
            checkpoint_data = {
                'episode_id': episode_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': asdict(metrics),
                'config': config,
                'algorithm': self.algorithm,
                'save_time': time.time()
            }
            
            # 모델 저장
            torch.save(checkpoint_data, model_path)
            
            # 파일 크기 계산
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # 체크포인트 정보 생성
            checkpoint = ModelCheckpoint(
                episode_id=episode_id,
                model_path=str(model_path),
                metrics=metrics,
                model_size_mb=file_size_mb,
                algorithm=self.algorithm,
                config_snapshot=config.copy()
            )
            
            # 체크포인트 추가
            self.checkpoints.append(checkpoint)
            self.total_saved += 1
            self.total_size_mb += file_size_mb
            
            # 메타데이터 저장
            self._save_metadata()
            
            # 자동 정리
            if self.auto_cleanup:
                self._cleanup_checkpoints()
            
            print(f"[INFO] 체크포인트 저장: {model_path} ({file_size_mb:.2f}MB)")
            
            return str(model_path)
        
        finally:
            # 모델을 원래 디바이스로 복원
            model.to(device)
    
    def record_episode_metrics(self, metrics: EpisodeMetrics):
        """에피소드 메트릭 기록"""
        self.training_history.append(metrics)
        
        # 주기적으로 로그 저장
        if len(self.training_history) % 10 == 0:
            self._save_training_log()
    
    def _save_training_log(self):
        """학습 로그 저장"""
        log_data = {
            'algorithm': self.algorithm,
            'environment': self.environment,
            'total_episodes': len(self.training_history),
            'last_updated': time.time(),
            'training_history': [asdict(metrics) for metrics in self.training_history]
        }
        
        with open(self.training_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def load_checkpoint(self, episode_id: int = None, best: bool = False) -> Optional[Dict[str, Any]]:
        """체크포인트 로드
        
        Args:
            episode_id: 특정 에피소드 ID (None이면 최신)
            best: 최고 성능 모델 로드 여부
            
        Returns:
            체크포인트 데이터 (없으면 None)
        """
        if not self.checkpoints:
            return None
        
        # 최고 성능 모델 선택
        if best:
            checkpoint = max(self.checkpoints, key=lambda cp: cp.performance_score)
        elif episode_id is not None:
            # 특정 에피소드 검색
            checkpoint = None
            for cp in self.checkpoints:
                if cp.episode_id == episode_id:
                    checkpoint = cp
                    break
            if checkpoint is None:
                return None
        else:
            # 최신 모델 선택
            checkpoint = max(self.checkpoints, key=lambda cp: cp.episode_id)
        
        # 체크포인트 로드
        try:
            checkpoint_data = torch.load(checkpoint.model_path, map_location='cpu')
            print(f"[INFO] 체크포인트 로드: {checkpoint.model_path}")
            return checkpoint_data
        except Exception as e:
            print(f"[ERROR] 체크포인트 로드 실패: {e}")
            return None
    
    def get_best_models(self, metric: str = 'reward', top_k: int = 10) -> List[ModelCheckpoint]:
        """최고 성능 모델들 반환
        
        Args:
            metric: 정렬 기준 ('reward', 'convergence', 'performance')
            top_k: 반환할 모델 수
            
        Returns:
            성능순으로 정렬된 체크포인트 리스트
        """
        if not self.checkpoints:
            return []
        
        if metric == 'reward':
            key_func = lambda cp: cp.metrics.total_reward
        elif metric == 'convergence':
            key_func = lambda cp: cp.metrics.convergence_score
        elif metric == 'performance':
            key_func = lambda cp: cp.performance_score
        else:
            key_func = lambda cp: cp.metrics.total_reward
        
        sorted_checkpoints = sorted(self.checkpoints, key=key_func, reverse=True)
        return sorted_checkpoints[:top_k]
    
    def _cleanup_checkpoints(self):
        """오래된 체크포인트 정리"""
        if len(self.checkpoints) <= self.keep_latest + self.keep_best:
            return
        
        # 최신 체크포인트들
        latest_checkpoints = sorted(self.checkpoints, key=lambda cp: cp.episode_id)[-self.keep_latest:]
        
        # 최고 성능 체크포인트들
        best_checkpoints = self.get_best_models(top_k=self.keep_best)
        
        # 보관할 체크포인트 집합
        keep_checkpoints = set()
        keep_checkpoints.update(latest_checkpoints)
        keep_checkpoints.update(best_checkpoints)
        
        # 삭제할 체크포인트들
        to_delete = [cp for cp in self.checkpoints if cp not in keep_checkpoints]
        
        for checkpoint in to_delete:
            try:
                Path(checkpoint.model_path).unlink()
                self.total_size_mb -= checkpoint.model_size_mb
                print(f"[INFO] 오래된 체크포인트 삭제: {checkpoint.model_path}")
            except Exception as e:
                print(f"[WARNING] 체크포인트 삭제 실패: {e}")
        
        # 체크포인트 리스트 업데이트
        self.checkpoints = list(keep_checkpoints)
        
        # 메타데이터 저장
        self._save_metadata()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        if not self.training_history:
            return {}
        
        rewards = [m.total_reward for m in self.training_history]
        lengths = [m.episode_length for m in self.training_history]
        losses = [m.average_loss for m in self.training_history if m.average_loss > 0]
        
        summary = {
            'algorithm': self.algorithm,
            'total_episodes': len(self.training_history),
            'total_checkpoints': len(self.checkpoints),
            'total_storage_mb': self.total_size_mb,
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'latest_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            },
            'episode_length_stats': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths)
            }
        }
        
        if losses:
            summary['loss_stats'] = {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses)
            }
        
        if self.checkpoints:
            best_checkpoint = max(self.checkpoints, key=lambda cp: cp.performance_score)
            summary['best_model'] = {
                'episode_id': best_checkpoint.episode_id,
                'reward': best_checkpoint.metrics.total_reward,
                'performance_score': best_checkpoint.performance_score,
                'path': best_checkpoint.model_path
            }
        
        return summary
    
    def export_training_data(self, format: str = 'json') -> str:
        """학습 데이터 내보내기
        
        Args:
            format: 출력 형식 ('json', 'csv')
            
        Returns:
            내보낸 파일 경로
        """
        if format == 'json':
            export_path = self.model_dir / f"{self.algorithm}_training_export.json"
            export_data = {
                'summary': self.get_training_summary(),
                'training_history': [asdict(m) for m in self.training_history],
                'checkpoints': [
                    {
                        'episode_id': cp.episode_id,
                        'metrics': asdict(cp.metrics),
                        'model_size_mb': cp.model_size_mb,
                        'path': cp.model_path
                    }
                    for cp in self.checkpoints
                ]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            export_path = self.model_dir / f"{self.algorithm}_training_export.csv"
            
            # 학습 히스토리를 DataFrame으로 변환
            df_data = []
            for metrics in self.training_history:
                row = asdict(metrics)
                row.pop('extra_metrics', None)  # 복잡한 딕셔너리 제외
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(export_path, index=False)
        
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
        
        print(f"[INFO] 학습 데이터 내보내기 완료: {export_path}")
        return str(export_path)