import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from ..networks import Actor, Critic
from ..core import ReplayBuffer, GaussianNoise, soft_update, get_device


class DDPGAgent:
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
                 device: Optional[torch.device] = None):
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
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or get_device()
        
        # 액터-크리틱 네트워크
        self.actor = Actor(state_dim, action_dim, action_bound=action_bound).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # 타겟 네트워크
        self.target_actor = Actor(state_dim, action_dim, action_bound=action_bound).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저 (액터와 크리틱 학습률 분리)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 노이즈 프로세스 (가우시안 노이즈 권장)
        self.noise = GaussianNoise(
            size=action_dim,
            sigma=noise_sigma,
            decay_rate=noise_decay
        )
        
        # 리플레이 버퍼
        self.buffer = ReplayBuffer(buffer_size)
        
        # 학습 통계
        self.training_steps = 0
    
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
            noise = self.noise.sample()
            action = action + noise
            # 행동 범위 제한
            action = np.clip(action, -self.action_bound, self.action_bound)
        
        # dtype 확보 (환경 호환성)
        return action.astype(np.float32)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """경험 저장"""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """액터-크리틱 네트워크 업데이트
        
        Returns:
            학습 메트릭 딕셔너리
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 크리틱 업데이트
        with torch.no_grad():
            # 타겟 액터가 다음 상태에서 결정적 행동 생성
            next_actions = self.target_actor(next_states)
            # 타겟 크리틱이 Q-값 평가
            target_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # 액터 업데이트
        # 정책 경사: Q(s, μ(s))를 최대화
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트 (Polyak averaging)
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        
        # 노이즈 감소
        self.noise.decay()
        
        self.training_steps += 1
        
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
    
    def save(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_sigma': self.noise.sigma,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.noise.sigma = checkpoint['noise_sigma']
        self.training_steps = checkpoint['training_steps']