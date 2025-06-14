import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, Optional

from ..networks import QNetwork
from ..core import ReplayBuffer, hard_update, get_device


class DQNAgent:
    """DQN (Deep Q-Network) 에이전트
    
    이산적 행동 공간에서 작동하며, Q-값을 통해 암묵적으로 결정적 정책을 구현합니다.
    주요 특징:
    - Q-network를 통해 각 행동의 가치 추정
    - argmax를 통한 결정적 행동 선택
    - ε-greedy 전략으로 탐험
    - 타겟 네트워크로 학습 안정성 향상
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: Optional[torch.device] = None):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 개수 (이산)
            learning_rate: 학습률
            gamma: 할인 인자
            epsilon: 초기 탐험율
            epsilon_min: 최소 탐험율
            epsilon_decay: 탐험율 감소율
            buffer_size: 리플레이 버퍼 크기
            batch_size: 배치 크기
            target_update_freq: 타겟 네트워크 업데이트 주기
            device: 연산 디바이스
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or get_device()
        
        # Q-네트워크와 타겟 네트워크
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        hard_update(self.target_network, self.q_network)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 리플레이 버퍼
        self.buffer = ReplayBuffer(buffer_size)
        
        # 학습 스텝 카운터
        self.update_step = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """행동 선택
        
        이산적 행동 공간에서 Q-값 기반으로 행동을 선택합니다.
        학습 중에는 ε-greedy, 평가 시에는 결정적 선택을 합니다.
        
        Args:
            state: 현재 상태
            deterministic: True면 항상 최적 행동 선택 (평가용)
            
        Returns:
            선택된 행동 (정수)
        """
        if not deterministic and random.random() < self.epsilon:
            # 탐험: 무작위 행동 선택
            return random.randrange(self.action_dim)
        else:
            # 활용: Q-값이 가장 높은 행동 선택 (결정적)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        """경험 저장"""
        # DQN은 이산 행동을 사용하므로 action을 배열로 변환
        action_array = np.array([action], dtype=np.float32)
        self.buffer.push(state, action_array, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """네트워크 업데이트
        
        Returns:
            학습 메트릭 딕셔너리
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions.squeeze()).to(self.device)  # 이산 행동
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 현재 Q-값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 타겟 Q-값 계산
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 손실 계산 및 역전파
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 타겟 네트워크 업데이트
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            hard_update(self.target_network, self.q_network)
        
        # 엡실론 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        metrics = {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
        
        return metrics
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """주어진 상태에서 모든 행동의 Q-값 반환
        
        결정적 정책 분석을 위한 메서드입니다.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().squeeze()
    
    def save(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_step': self.update_step
        }, path)
    
    def load(self, path: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_step = checkpoint['update_step']