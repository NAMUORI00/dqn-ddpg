import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """경험 리플레이 버퍼 구현
    
    DQN과 DDPG 모두에서 사용되는 경험 리플레이 버퍼입니다.
    시간적 상관관계를 제거하고 안정적인 학습을 위해 사용됩니다.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: 버퍼의 최대 크기
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """경험을 버퍼에 추가
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """버퍼에서 배치 샘플링
        
        Args:
            batch_size: 샘플링할 배치 크기
            
        Returns:
            states, actions, rewards, next_states, dones의 배치
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.float32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """버퍼의 현재 크기 반환"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """배치 샘플링이 가능한지 확인"""
        return len(self.buffer) >= batch_size