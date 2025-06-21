import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, Optional, Any

from .base_agent import BaseReinforcementAgent
from ..networks import QNetwork
from ..core import hard_update


class DQNAgent(BaseReinforcementAgent):
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
                 device: Optional[torch.device] = None,
                 use_gpu_buffer: bool = False,
                 use_mixed_precision: bool = False):
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
            use_gpu_buffer: GPU에 리플레이 버퍼 저장
            use_mixed_precision: Mixed Precision Training 사용
        """
        # Initialize base agent
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_gpu_buffer=use_gpu_buffer,
            use_mixed_precision=use_mixed_precision
        )
        
        # DQN-specific parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_step = 0
        
        # Q-네트워크와 타겟 네트워크
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        
        # Register networks with base agent
        self._register_network('q_network', self.q_network)
        self._register_network('target_network', self.target_network)
        
        # 타겟 네트워크 초기화
        hard_update(self.target_network, self.q_network)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self._register_optimizer('optimizer', self.optimizer)
    
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
                
                # 안전성 체크
                if action < 0 or action >= self.action_dim:
                    print(f"⚠️ DQN 행동 오류: action={action}, q_values={q_values}")
                    action = 0  # 안전한 기본값
            
            return action
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Compatibility alias for select_action"""
        return self.select_action(state, deterministic)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        """경험 저장"""
        # Use base class method which handles action conversion
        super().store_transition(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """네트워크 업데이트 (GPU 최적화)
        
        Returns:
            학습 메트릭 딕셔너리
        """
        # Get batch data using base class method
        batch_data = self._get_batch_data()
        if batch_data is None:
            return {}
        
        states, actions, rewards, next_states, dones = batch_data
        
        # DQN-specific preprocessing
        actions = actions.squeeze().long()  # Convert to discrete actions
        
        # 행동 인덱스 안전성 체크
        if torch.any(actions < 0) or torch.any(actions >= self.action_dim):
            print(f"⚠️ DQN update 행동 오류: actions={actions[:5]}, action_dim={self.action_dim}")
            actions = torch.clamp(actions, 0, self.action_dim - 1)
        
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Compute Q-values and targets
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                loss = F.mse_loss(current_q_values, target_q_values)
        else:
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update network using base class method
        self.optimizer.zero_grad()
        self._update_mixed_precision(loss, self.optimizer)
        
        # 타겟 네트워크 업데이트
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            hard_update(self.target_network, self.q_network)
        
        # 엡실론 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # GPU 메모리 최적화 using base class method
        self._optimize_gpu_memory()
        
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
    
    def _get_custom_save_data(self) -> Dict[str, Any]:
        """Get DQN-specific data to save"""
        return {
            'epsilon': self.epsilon,
            'update_step': self.update_step,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'target_update_freq': self.target_update_freq
        }
    
    def _load_custom_save_data(self, checkpoint: Dict[str, Any]) -> None:
        """Load DQN-specific data from checkpoint"""
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.update_step = checkpoint.get('update_step', 0)
        # Other DQN parameters use defaults if not in checkpoint
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        self.target_update_freq = checkpoint.get('target_update_freq', self.target_update_freq)