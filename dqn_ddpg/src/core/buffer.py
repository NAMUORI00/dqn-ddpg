import numpy as np
import torch
import random
from collections import deque
from typing import Tuple, List, Optional, Union


class ReplayBuffer:
    """경험 리플레이 버퍼 구현
    
    DQN과 DDPG 모두에서 사용되는 경험 리플레이 버퍼입니다.
    시간적 상관관계를 제거하고 안정적인 학습을 위해 사용됩니다.
    """
    
    def __init__(self, capacity: int, use_gpu: bool = False, device: Optional[torch.device] = None):
        """
        Args:
            capacity: 버퍼의 최대 크기
            use_gpu: GPU에 버퍼 저장 여부
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.capacity = capacity
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if device is None:
            from .utils import get_device
            self.device = get_device() if self.use_gpu else torch.device('cpu')
        else:
            self.device = device
            
        self.buffer = deque(maxlen=capacity)
        self._use_pinned_memory = self.use_gpu  # CPU-GPU 전송 최적화
    
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
        # GPU 버퍼인 경우 텐서로 변환하여 저장
        if self.use_gpu:
            state_tensor = torch.from_numpy(state).float().to(self.device, non_blocking=True)
            action_tensor = torch.from_numpy(action).float().to(self.device, non_blocking=True)
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
            next_state_tensor = torch.from_numpy(next_state).float().to(self.device, non_blocking=True)
            done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device)
            
            self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))
        else:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        """버퍼에서 배치 샘플링
        
        Args:
            batch_size: 샘플링할 배치 크기
            
        Returns:
            states, actions, rewards, next_states, dones의 배치
        """
        batch = random.sample(self.buffer, batch_size)
        
        if self.use_gpu:
            # GPU에서 직접 배치 생성 (더 빠름)
            states = torch.stack([e[0] for e in batch])
            actions = torch.stack([e[1] for e in batch])
            rewards = torch.stack([e[2] for e in batch])
            next_states = torch.stack([e[3] for e in batch])
            dones = torch.stack([e[4] for e in batch])
            
            return states, actions, rewards, next_states, dones
        else:
            # CPU 버전 (기존 코드)
            states = np.array([e[0] for e in batch], dtype=np.float32)
            actions = np.array([e[1] for e in batch], dtype=np.float32)
            rewards = np.array([e[2] for e in batch], dtype=np.float32)
            next_states = np.array([e[3] for e in batch], dtype=np.float32)
            dones = np.array([e[4] for e in batch], dtype=np.float32)
            
            return states, actions, rewards, next_states, dones
    
    def sample_tensor(self, batch_size: int, target_device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
        """버퍼에서 배치를 샘플링하여 텐서로 반환
        
        Args:
            batch_size: 샘플링할 배치 크기
            target_device: 반환할 텐서의 디바이스
            
        Returns:
            GPU 텐서 형태의 배치
        """
        if target_device is None:
            target_device = self.device
            
        if self.use_gpu:
            # 이미 GPU에 있는 경우
            states, actions, rewards, next_states, dones = self.sample(batch_size)
            if target_device != self.device:
                # 다른 GPU로 전송이 필요한 경우
                states = states.to(target_device, non_blocking=True)
                actions = actions.to(target_device, non_blocking=True)
                rewards = rewards.to(target_device, non_blocking=True)
                next_states = next_states.to(target_device, non_blocking=True)
                dones = dones.to(target_device, non_blocking=True)
            return states, actions, rewards, next_states, dones
        else:
            # CPU 버퍼에서 GPU로 전송
            states, actions, rewards, next_states, dones = self.sample(batch_size)
            
            # Pinned memory 사용하여 빠른 전송
            if self._use_pinned_memory:
                states = torch.from_numpy(states).pin_memory().to(target_device, non_blocking=True)
                actions = torch.from_numpy(actions).pin_memory().to(target_device, non_blocking=True)
                rewards = torch.from_numpy(rewards).pin_memory().to(target_device, non_blocking=True)
                next_states = torch.from_numpy(next_states).pin_memory().to(target_device, non_blocking=True)
                dones = torch.from_numpy(dones).pin_memory().to(target_device, non_blocking=True)
            else:
                states = torch.FloatTensor(states).to(target_device)
                actions = torch.FloatTensor(actions).to(target_device)
                rewards = torch.FloatTensor(rewards).to(target_device)
                next_states = torch.FloatTensor(next_states).to(target_device)
                dones = torch.FloatTensor(dones).to(target_device)
            
            return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """버퍼의 현재 크기 반환"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """배치 샘플링이 가능한지 확인"""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """버퍼 초기화"""
        self.buffer.clear()
        if self.use_gpu:
            # GPU 메모리 정리
            torch.cuda.empty_cache()


class PrioritizedReplayBuffer(ReplayBuffer):
    """우선순위 경험 리플레이 버퍼
    
    TD 오류가 큰 경험을 더 자주 샘플링하여 학습 효율성을 향상시킵니다.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, use_gpu: bool = False, 
                 device: Optional[torch.device] = None):
        """
        Args:
            capacity: 버퍼의 최대 크기
            alpha: 우선순위 지수 (0: 균등, 1: 완전 우선순위)
            beta: 중요도 샘플링 보정 (0: 보정 없음, 1: 완전 보정)
            beta_increment: 에피소드마다 beta 증가량
            use_gpu: GPU에 버퍼 저장 여부
            device: 사용할 디바이스
        """
        super().__init__(capacity, use_gpu, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        
        # 우선순위 저장
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """새로운 경험을 최대 우선순위로 추가"""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        """우선순위 기반 샘플링
        
        Returns:
            states, actions, rewards, next_states, dones, weights, indices
        """
        # 우선순위를 확률로 변환
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # 중요도 샘플링 가중치 계산
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        if self.use_gpu:
            states = torch.stack([batch[i][0] for i in range(batch_size)])
            actions = torch.stack([batch[i][1] for i in range(batch_size)])
            rewards = torch.stack([batch[i][2] for i in range(batch_size)])
            next_states = torch.stack([batch[i][3] for i in range(batch_size)])
            dones = torch.stack([batch[i][4] for i in range(batch_size)])
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            states = np.array([batch[i][0] for i in range(batch_size)], dtype=np.float32)
            actions = np.array([batch[i][1] for i in range(batch_size)], dtype=np.float32)
            rewards = np.array([batch[i][2] for i in range(batch_size)], dtype=np.float32)
            next_states = np.array([batch[i][3] for i in range(batch_size)], dtype=np.float32)
            dones = np.array([batch[i][4] for i in range(batch_size)], dtype=np.float32)
        
        # Beta 증가
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: Union[List[int], torch.Tensor], 
                         priorities: Union[np.ndarray, torch.Tensor]) -> None:
        """TD 오류 기반으로 우선순위 업데이트
        
        Args:
            indices: 업데이트할 인덱스
            priorities: 새로운 우선순위 (일반적으로 TD 오류의 절댓값)
        """
        if torch.is_tensor(priorities):
            priorities = priorities.cpu().numpy()
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
            
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority + self.epsilon)