import numpy as np
from typing import Optional


class OUNoise:
    """Ornstein-Uhlenbeck 노이즈 프로세스
    
    DDPG에서 연속적인 행동 공간 탐험을 위해 사용되는 시간 상관 노이즈입니다.
    하지만 최근 연구에서는 가우시안 노이즈가 동일하게 효과적임이 밝혀졌습니다.
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2, seed: Optional[int] = None):
        """
        Args:
            size: 행동 차원
            mu: 평균 회귀 값
            theta: 평균으로의 회귀 속도
            sigma: 노이즈의 변동성
            seed: 랜덤 시드
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()
    
    def reset(self):
        """내부 상태를 평균값으로 초기화"""
        self.state = self.mu.copy()
    
    def sample(self) -> np.ndarray:
        """다음 노이즈 샘플 생성"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=self.mu.shape)
        self.state = x + dx
        return self.state


class GaussianNoise:
    """가우시안 노이즈 프로세스
    
    DDPG에서 탐험을 위한 간단하고 효과적인 노이즈입니다.
    최근 연구에서는 OU 노이즈만큼 효과적임이 입증되었습니다.
    """
    
    def __init__(self, size: int, mu: float = 0.0, sigma: float = 0.2, 
                 decay_rate: float = 0.995, min_sigma: float = 0.01):
        """
        Args:
            size: 행동 차원
            mu: 노이즈 평균
            sigma: 노이즈 표준편차
            decay_rate: 시그마 감소율
            min_sigma: 최소 시그마 값
        """
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.initial_sigma = sigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
    
    def reset(self):
        """시그마를 초기값으로 재설정"""
        self.sigma = self.initial_sigma
    
    def sample(self) -> np.ndarray:
        """가우시안 노이즈 샘플 생성"""
        noise = np.random.normal(self.mu, self.sigma, size=self.size)
        return noise
    
    def decay(self):
        """노이즈 수준을 점진적으로 감소"""
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)