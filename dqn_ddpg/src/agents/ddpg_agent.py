import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .base_agent import BaseReinforcementAgent
from ..networks import Actor, Critic
from ..core import GaussianNoise, soft_update


class DDPGAgent(BaseReinforcementAgent):
    """DDPG (Deep Deterministic Policy Gradient) 에이전트
    
    연속적 행동 공간에서 작동하며, 액터 네트워크를 통해 명시적으로 결정적 정책을 구현합니다.
    주요 특징:
    - 액터-크리틱 구조
    - 액터가 직접 결정적 행동 출력
    - 가우시안 노이즈로 탐험
    - Polyak averaging으로 타겟 네트워크 업데이트
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bound: float = 1.0,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_sigma: float = 0.2,
                 noise_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 device: Optional[torch.device] = None,
                 use_gpu_buffer: bool = False,
                 use_mixed_precision: bool = False):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원 (연속)
            action_bound: 행동의 최대 절댓값
            actor_lr: 액터 학습률
            critic_lr: 크리틱 학습률
            gamma: 할인 인자
            tau: 소프트 업데이트 비율
            noise_sigma: 노이즈 표준편차
            noise_decay: 노이즈 감소율
            buffer_size: 리플레이 버퍼 크기
            batch_size: 배치 크기
            device: 연산 디바이스
            use_gpu_buffer: GPU에 리플레이 버퍼 저장
            use_mixed_precision: Mixed Precision Training 사용
        """
        # Initialize base agent with multiple learning rates
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate={'actor': actor_lr, 'critic': critic_lr},
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_gpu_buffer=use_gpu_buffer,
            use_mixed_precision=use_mixed_precision
        )
        
        # DDPG-specific parameters
        self.action_bound = action_bound
        self.tau = tau
        
        # 액터-크리틱 네트워크
        self.actor = Actor(state_dim, action_dim, action_bound=action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim, action_bound=action_bound)
        self.target_critic = Critic(state_dim, action_dim)
        
        # Register networks with base agent
        self._register_network('actor', self.actor)
        self._register_network('critic', self.critic)
        self._register_network('target_actor', self.target_actor)
        self._register_network('target_critic', self.target_critic)
        
        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저 (액터와 크리틱 학습률 분리)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self._register_optimizer('actor_optimizer', self.actor_optimizer)
        self._register_optimizer('critic_optimizer', self.critic_optimizer)
        
        # 노이즈 프로세스 (가우시안 노이즈 권장)
        self.noise = GaussianNoise(
            size=action_dim,
            sigma=noise_sigma,
            decay_rate=noise_decay
        )
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """행동 선택
        
        액터 네트워크가 직접 결정적 행동을 출력합니다.
        학습 중에는 노이즈를 추가하여 탐험합니다.
        
        Args:
            state: 현재 상태
            add_noise: True면 탐험을 위한 노이즈 추가
            
        Returns:
            선택된 연속 행동
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 액터가 결정적 행동 출력
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().squeeze()
        
        # 1차원 배열로 만들기 (환경 호환성)
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        elif action.ndim == 0:
            action = np.array([action.item()], dtype=np.float32)
        
        # 탐험을 위한 노이즈 추가
        if add_noise:
            # GPU에서 노이즈 생성 가능
            if self.device.type == 'cuda' and isinstance(action, np.ndarray):
                action_tensor = torch.from_numpy(action).to(self.device)
                noise_tensor = torch.randn_like(action_tensor) * self.noise.sigma
                action_tensor = torch.clamp(action_tensor + noise_tensor, -self.action_bound, self.action_bound)
                action = action_tensor.cpu().numpy()
            else:
                noise = self.noise.sample()
                action = action + noise
                action = np.clip(action, -self.action_bound, self.action_bound)
        
        # dtype 확보 (환경 호환성)
        return action.astype(np.float32)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """경험 저장"""
        # Use base class method
        super().store_transition(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """액터-크리틱 네트워크 업데이트 (GPU 최적화)
        
        Returns:
            학습 메트릭 딕셔너리
        """
        # Get batch data using base class method
        batch_data = self._get_batch_data()
        if batch_data is None:
            return {}
        
        states, actions, rewards, next_states, dones = batch_data
        
        # DDPG-specific preprocessing
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Compute critic loss
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    next_actions = self.target_actor(next_states)
                    target_q_values = self.target_critic(next_states, next_actions)
                    target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
                current_q_values = self.critic(states, actions)
                critic_loss = F.mse_loss(current_q_values, target_q_values)
        else:
            with torch.no_grad():
                next_actions = self.target_actor(next_states)
                target_q_values = self.target_critic(next_states, next_actions)
                target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
            current_q_values = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update critic using base class method
        self.critic_optimizer.zero_grad()
        self._update_mixed_precision(critic_loss, self.critic_optimizer)
        
        # Compute actor loss
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                actor_loss = -self.critic(states, self.actor(states)).mean()
        else:
            actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor using base class method
        self.actor_optimizer.zero_grad()
        self._update_mixed_precision(actor_loss, self.actor_optimizer)
        
        # 타겟 네트워크 소프트 업데이트 (Polyak averaging)
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        
        # 노이즈 감소
        self.noise.decay()
        
        self.training_steps += 1
        
        # GPU 메모리 최적화 using base class method
        self._optimize_gpu_memory()
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q_values.mean().item(),
            'noise_sigma': self.noise.sigma
        }
        
        return metrics
    
    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """순수 결정적 행동 반환
        
        노이즈 없이 액터의 출력만 반환합니다.
        결정적 정책 분석을 위한 메서드입니다.
        """
        return self.select_action(state, add_noise=False)
    
    def reset_noise(self) -> None:
        """노이즈 프로세스 리셋"""
        self.noise.reset()
    
    def _get_custom_save_data(self) -> Dict[str, Any]:
        """Get DDPG-specific data to save"""
        return {
            'action_bound': self.action_bound,
            'tau': self.tau,
            'noise_sigma': self.noise.sigma,
            'training_steps': self.training_steps
        }
    
    def _load_custom_save_data(self, checkpoint: Dict[str, Any]) -> None:
        """Load DDPG-specific data from checkpoint"""
        self.action_bound = checkpoint.get('action_bound', self.action_bound)
        self.tau = checkpoint.get('tau', self.tau)
        self.noise.sigma = checkpoint.get('noise_sigma', self.noise.sigma)
        self.training_steps = checkpoint.get('training_steps', 0)
    
    def reset_for_new_episode(self) -> None:
        """Reset agent state for new episode"""
        super().reset_for_new_episode()
        # DDPG can optionally reset noise process
        # self.reset_noise()  # Uncomment if noise reset is desired