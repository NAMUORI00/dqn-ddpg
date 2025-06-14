import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class NormalizeWrapper(gym.Wrapper):
    """상태 정규화 래퍼
    
    상태를 정규화하여 학습 안정성을 향상시킵니다.
    """
    
    def __init__(self, env: gym.Env):
        super(NormalizeWrapper, self).__init__(env)
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
        self.running_mean = np.zeros(env.observation_space.shape)
        self.running_var = np.ones(env.observation_space.shape)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        state, info = self.env.reset(**kwargs)
        return self._normalize_state(state), info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_state(next_state), reward, terminated, truncated, info
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """온라인 평균/분산 업데이트 및 정규화"""
        self.state_count += 1
        
        # Welford's online algorithm
        delta = state - self.running_mean
        self.running_mean += delta / self.state_count
        self.running_var += delta * (state - self.running_mean)
        
        # 정규화
        std = np.sqrt(self.running_var / max(1, self.state_count - 1))
        std = np.maximum(std, 1e-6)  # 0 방지
        
        normalized_state = (state - self.running_mean) / std
        return normalized_state.astype(np.float32)


class ActionWrapper(gym.Wrapper):
    """행동 스케일링 래퍼
    
    연속 행동 공간을 [-1, 1]로 정규화합니다.
    """
    
    def __init__(self, env: gym.Env):
        super(ActionWrapper, self).__init__(env)
        
        # 연속 행동 공간만 지원
        assert isinstance(env.action_space, gym.spaces.Box), \
            "ActionWrapper only supports Box action spaces"
        
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        
        # 행동 공간을 [-1, 1]로 재정의
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """[-1, 1] 범위의 행동을 원래 범위로 변환"""
        # 선형 변환: [-1, 1] -> [low, high]
        true_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        true_action = np.clip(true_action, self.action_low, self.action_high)
        
        return self.env.step(true_action)


class RewardScaleWrapper(gym.Wrapper):
    """보상 스케일링 래퍼
    
    보상을 스케일링하여 학습 안정성을 향상시킵니다.
    """
    
    def __init__(self, env: gym.Env, scale: float = 0.1):
        super(RewardScaleWrapper, self).__init__(env)
        self.scale = scale
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        state, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        return state, scaled_reward, terminated, truncated, info


def create_dqn_env(env_name: str = "CartPole-v1") -> gym.Env:
    """DQN용 환경 생성
    
    이산적 행동 공간을 가진 환경을 생성합니다.
    """
    env = gym.make(env_name)
    # CartPole은 이미 잘 정규화되어 있으므로 추가 래핑 불필요
    return env


def create_ddpg_env(env_name: str = "Pendulum-v1") -> gym.Env:
    """DDPG용 환경 생성
    
    연속적 행동 공간을 가진 환경을 생성하고 적절히 래핑합니다.
    """
    env = gym.make(env_name)
    env = ActionWrapper(env)  # 행동을 [-1, 1]로 정규화
    env = RewardScaleWrapper(env, scale=0.1)  # 보상 스케일링
    return env