"""
고급 수렴 감지 시스템
더 정교한 수렴 조건과 학습 완료 판정을 제공합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import scipy.stats as stats


@dataclass
class ConvergenceMetrics:
    """수렴 지표"""
    coefficient_of_variation: float  # 변동 계수
    improvement_rate: float  # 성능 개선률
    stability_score: float  # 안정성 점수
    trend_slope: float  # 트렌드 기울기
    plateau_duration: int  # 정체 기간
    confidence_level: float  # 수렴 신뢰도
    
    is_converged: bool = False
    convergence_reason: str = ""


class AdvancedConvergenceDetector:
    """고급 수렴 감지기
    
    다양한 통계적 방법을 사용하여 학습 수렴을 정확하게 감지합니다.
    """
    
    def __init__(self,
                 min_episodes: int = 500,
                 stability_window: int = 200,
                 cv_threshold: float = 0.05,
                 improvement_threshold: float = 0.01,
                 plateau_patience: int = 100,
                 confidence_threshold: float = 0.8):
        """
        Args:
            min_episodes: 수렴 판정을 위한 최소 에피소드 수
            stability_window: 안정성 확인 윈도우 크기
            cv_threshold: 변동 계수 임계값
            improvement_threshold: 성능 개선 임계값
            plateau_patience: 정체 허용 기간
            confidence_threshold: 수렴 신뢰도 임계값
        """
        self.min_episodes = min_episodes
        self.stability_window = stability_window
        self.cv_threshold = cv_threshold
        self.improvement_threshold = improvement_threshold
        self.plateau_patience = plateau_patience
        self.confidence_threshold = confidence_threshold
        
        # 상태 추적
        self.reward_history = deque(maxlen=10000)
        self.convergence_history = []
        self.plateau_start = None
        self.last_significant_improvement = 0
        
    def update(self, episode: int, reward: float) -> ConvergenceMetrics:
        """새로운 에피소드 결과로 수렴 상태 업데이트
        
        Args:
            episode: 에피소드 번호
            reward: 에피소드 보상
            
        Returns:
            수렴 지표
        """
        self.reward_history.append(reward)
        
        # 최소 에피소드 수 확인
        if episode < self.min_episodes or len(self.reward_history) < self.stability_window:
            return ConvergenceMetrics(
                coefficient_of_variation=float('inf'),
                improvement_rate=float('inf'),
                stability_score=0.0,
                trend_slope=0.0,
                plateau_duration=0,
                confidence_level=0.0,
                is_converged=False,
                convergence_reason="충분한 데이터 없음"
            )
        
        # 수렴 지표 계산
        metrics = self._calculate_convergence_metrics(episode)
        
        # 수렴 판정
        convergence_decision = self._make_convergence_decision(metrics, episode)
        
        metrics.is_converged = convergence_decision['is_converged']
        metrics.convergence_reason = convergence_decision['reason']
        
        # 히스토리 업데이트
        self.convergence_history.append(metrics)
        
        return metrics
    
    def _calculate_convergence_metrics(self, episode: int) -> ConvergenceMetrics:
        """수렴 지표 계산"""
        rewards = np.array(list(self.reward_history))
        
        # 1. 변동 계수 (Coefficient of Variation)
        recent_rewards = rewards[-self.stability_window:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        # 2. 성능 개선률
        if len(rewards) >= self.stability_window * 2:
            # 최근 절반과 이전 절반 비교
            window_half = self.stability_window // 2
            recent_half = rewards[-window_half:]
            previous_half = rewards[-self.stability_window:-window_half]
            
            recent_mean = np.mean(recent_half)
            previous_mean = np.mean(previous_half)
            
            if previous_mean != 0:
                improvement_rate = abs(recent_mean - previous_mean) / abs(previous_mean)
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = float('inf')
        
        # 3. 안정성 점수 (분산의 역수 기반)
        stability_score = 1.0 / (1.0 + std_reward) if std_reward >= 0 else 0.0
        
        # 4. 트렌드 분석
        episodes_range = np.arange(len(recent_rewards))
        slope, _, r_value, _, _ = stats.linregress(episodes_range, recent_rewards)
        
        # 5. 정체 기간 계산
        plateau_duration = self._calculate_plateau_duration(episode)
        
        # 6. 신뢰도 계산
        confidence_level = self._calculate_confidence_level(cv, improvement_rate, stability_score, abs(slope))
        
        return ConvergenceMetrics(
            coefficient_of_variation=cv,
            improvement_rate=improvement_rate,
            stability_score=stability_score,
            trend_slope=slope,
            plateau_duration=plateau_duration,
            confidence_level=confidence_level
        )
    
    def _calculate_plateau_duration(self, episode: int) -> int:
        """정체 기간 계산"""
        if len(self.reward_history) < 50:
            return 0
        
        rewards = list(self.reward_history)
        
        # 최근 50에피소드의 평균과 그 이전 50에피소드의 평균 비교
        recent_avg = np.mean(rewards[-50:])
        
        plateau_count = 0
        for i in range(len(rewards) - 100, 0, -50):
            if i < 50:
                break
            
            compare_avg = np.mean(rewards[i:i+50])
            improvement = abs(recent_avg - compare_avg) / abs(compare_avg) if compare_avg != 0 else 0
            
            if improvement < self.improvement_threshold:
                plateau_count += 50
            else:
                break
        
        return plateau_count
    
    def _calculate_confidence_level(self, cv: float, improvement_rate: float, 
                                  stability_score: float, trend_slope: float) -> float:
        """수렴 신뢰도 계산"""
        # 각 지표별 점수 계산 (0-1 스케일)
        cv_score = max(0, 1 - (cv / self.cv_threshold))
        improvement_score = max(0, 1 - (improvement_rate / self.improvement_threshold))
        stability_normalized = min(1.0, stability_score)
        trend_score = max(0, 1 - abs(trend_slope) * 100)  # 기울기가 0에 가까울수록 높은 점수
        
        # 가중 평균으로 신뢰도 계산
        weights = [0.3, 0.3, 0.25, 0.15]  # CV, 개선률, 안정성, 트렌드 순
        confidence = (cv_score * weights[0] + 
                     improvement_score * weights[1] + 
                     stability_normalized * weights[2] + 
                     trend_score * weights[3])
        
        return confidence
    
    def _make_convergence_decision(self, metrics: ConvergenceMetrics, episode: int) -> Dict[str, any]:
        """수렴 판정"""
        reasons = []
        convergence_factors = 0
        
        # 1. 변동 계수 확인
        if metrics.coefficient_of_variation < self.cv_threshold:
            convergence_factors += 1
            reasons.append(f"CV({metrics.coefficient_of_variation:.4f}) < {self.cv_threshold}")
        
        # 2. 성능 개선률 확인
        if metrics.improvement_rate < self.improvement_threshold:
            convergence_factors += 1
            reasons.append(f"개선률({metrics.improvement_rate:.4f}) < {self.improvement_threshold}")
        
        # 3. 정체 기간 확인
        if metrics.plateau_duration >= self.plateau_patience:
            convergence_factors += 1
            reasons.append(f"정체 기간({metrics.plateau_duration}) >= {self.plateau_patience}")
        
        # 4. 신뢰도 확인
        if metrics.confidence_level >= self.confidence_threshold:
            convergence_factors += 1
            reasons.append(f"신뢰도({metrics.confidence_level:.3f}) >= {self.confidence_threshold}")
        
        # 5. 트렌드 안정성 확인
        if abs(metrics.trend_slope) < 0.001:  # 거의 평평한 트렌드
            convergence_factors += 1
            reasons.append(f"트렌드 기울기({metrics.trend_slope:.6f}) ≈ 0")
        
        # 수렴 판정: 5개 조건 중 4개 이상 만족
        is_converged = convergence_factors >= 4
        
        if is_converged:
            reason = f"수렴 완료 ({convergence_factors}/5): " + "; ".join(reasons)
        else:
            reason = f"수렴 중 ({convergence_factors}/5): " + "; ".join(reasons) if reasons else "수렴 조건 미달"
        
        return {
            'is_converged': is_converged,
            'reason': reason,
            'convergence_factors': convergence_factors
        }
    
    def get_convergence_summary(self) -> Dict[str, any]:
        """수렴 상태 요약"""
        if not self.convergence_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.convergence_history[-1]
        
        # 수렴 추세 분석
        if len(self.convergence_history) >= 10:
            recent_confidences = [m.confidence_level for m in self.convergence_history[-10:]]
            confidence_trend = np.mean(np.diff(recent_confidences))
        else:
            confidence_trend = 0.0
        
        return {
            'status': 'converged' if latest_metrics.is_converged else 'training',
            'latest_metrics': latest_metrics,
            'total_episodes': len(self.reward_history),
            'confidence_trend': confidence_trend,
            'convergence_probability': latest_metrics.confidence_level,
            'estimated_episodes_to_convergence': self._estimate_episodes_to_convergence()
        }
    
    def _estimate_episodes_to_convergence(self) -> Optional[int]:
        """수렴까지 예상 에피소드 수 추정"""
        if len(self.convergence_history) < 50:
            return None
        
        # 최근 신뢰도 증가 추세 분석
        recent_confidences = [m.confidence_level for m in self.convergence_history[-50:]]
        
        if len(recent_confidences) < 10:
            return None
        
        # 선형 회귀로 추세 분석
        episodes = np.arange(len(recent_confidences))
        slope, intercept, r_value, _, _ = stats.linregress(episodes, recent_confidences)
        
        if slope <= 0 or r_value < 0.3:  # 증가 추세가 없거나 상관관계가 낮음
            return None
        
        # 신뢰도가 임계값에 도달할 때까지의 에피소드 예측
        current_confidence = recent_confidences[-1]
        episodes_needed = (self.confidence_threshold - current_confidence) / slope
        
        if episodes_needed <= 0:
            return 0  # 이미 수렴
        elif episodes_needed > 1000:
            return None  # 너무 먼 미래
        else:
            return int(episodes_needed)
    
    def should_stop_training(self, episode: int, patience: int = 50) -> Tuple[bool, str]:
        """학습 중단 여부 결정
        
        Args:
            episode: 현재 에피소드
            patience: 수렴 후 추가 대기 에피소드
            
        Returns:
            (중단 여부, 중단 이유)
        """
        if not self.convergence_history:
            return False, "데이터 부족"
        
        latest_metrics = self.convergence_history[-1]
        
        # 수렴 확인
        if latest_metrics.is_converged:
            # 수렴 후 patience만큼 더 기다림
            converged_episodes = sum(1 for m in self.convergence_history[-patience:] if m.is_converged)
            
            if converged_episodes >= patience:
                return True, f"수렴 완료 후 {patience}에피소드 대기 완료"
        
        # 매우 긴 정체 상태
        if latest_metrics.plateau_duration > self.plateau_patience * 2:
            return True, f"장기간 정체 ({latest_metrics.plateau_duration}에피소드)"
        
        # 성능 악화 감지
        if len(self.reward_history) >= 200:
            recent_mean = np.mean(list(self.reward_history)[-100:])
            previous_mean = np.mean(list(self.reward_history)[-200:-100])
            
            if previous_mean != 0:
                degradation = (previous_mean - recent_mean) / abs(previous_mean)
                if degradation > 0.1:  # 10% 이상 성능 악화
                    return True, f"성능 악화 감지 ({degradation:.2%})"
        
        return False, "학습 계속"