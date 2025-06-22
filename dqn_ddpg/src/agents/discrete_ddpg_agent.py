"""
DiscreteDDPGAgent - 이산 행동 공간에서 작동하는 DDPG 변형

기존 DDPG를 이산 행동 공간에서 사용할 수 있도록 행동 변환을 통해 확장한 에이전트입니다.
DQN과 DDPG를 동일한 이산 환경에서 공정하게 비교하기 위해 설계되었습니다.

핵심 아이디어:
- DDPG의 연속 출력을 이산 행동으로 변환
- Gumbel-Softmax 또는 확률적 매핑 사용
- 기존 DDPG의 모든 학습 메커니즘 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .ddpg_agent import DDPGAgent
from ..networks.actor import Actor
from ..networks.critic import Critic


class DiscreteActor(nn.Module):
    """이산 행동을 위한 Actor 네트워크"""
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.num_actions = num_actions
        
        # 네트워크 구성
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
            
        # 출력층 - 각 액션에 대한 logits
        layers.append(nn.Linear(input_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """상태를 받아 행동 확률 분포를 반환"""
        logits = self.network(state)
        # Gumbel-Softmax for differentiable discrete sampling
        return F.gumbel_softmax(logits, tau=1.0, hard=False)


class DiscreteCritic(nn.Module):
    """이산 행동을 위한 Critic 네트워크"""
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.num_actions = num_actions  # num_actions 저장
        
        # 상태 인코더
        state_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            state_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)
        
        # 액션과 상태를 결합하는 최종 층
        self.final_layer = nn.Sequential(
            nn.Linear(input_dim + num_actions, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """상태와 행동을 받아 Q값을 반환"""
        state_features = self.state_encoder(state)
        
        # action이 3차원인 경우 (batch_size, 1, num_actions) -> (batch_size, num_actions)로 변환
        if action.dim() == 3 and action.size(1) == 1:
            action = action.squeeze(1)
        # action이 정수 인덱스인 경우 one-hot으로 변환
        elif action.dim() == 1 or (action.dim() == 2 and action.size(1) == 1):
            if action.dim() == 2:
                action = action.squeeze(1)
            action_onehot = torch.zeros(action.size(0), self.num_actions, device=action.device)
            action_onehot.scatter_(1, action.long().unsqueeze(1), 1.0)
            action = action_onehot
        
        combined = torch.cat([state_features, action], dim=1)
        return self.final_layer(combined)


class DiscreteDDPGAgent(DDPGAgent):
    """이산 행동 공간용 DDPG 에이전트
    
    연속 DDPG를 이산 행동 공간에 적응시켜 DQN과 공정한 비교를 가능하게 합니다.
    Gumbel-Softmax를 사용하여 미분가능한 이산 행동 선택을 구현합니다.
    """
    
    def __init__(self,
                 state_dim: int,
                 num_actions: int = 2,  # CartPole의 경우 2개
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 device: Optional[str] = None,
                 use_gpu_buffer: bool = True,
                 use_mixed_precision: bool = True,
                 noise_sigma: float = 0.1,
                 noise_decay: float = 0.995,
                 **kwargs):
        
        self.num_actions = num_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # 부모 클래스 초기화 (action_dim을 num_actions로 설정)
        super().__init__(
            state_dim=state_dim,
            action_dim=num_actions,  # 이산 행동 개수
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_gpu_buffer=use_gpu_buffer,
            use_mixed_precision=use_mixed_precision,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
            **kwargs
        )
        
        # 기존 네트워크를 이산 버전으로 교체
        self._replace_networks()
        
    def _replace_networks(self):
        """연속 네트워크를 이산 버전으로 교체"""
        # Actor 네트워크 교체
        self.actor = DiscreteActor(
            self.state_dim, 
            self.num_actions,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        self.target_actor = DiscreteActor(
            self.state_dim, 
            self.num_actions,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        # Critic 네트워크 교체  
        self.critic = DiscreteCritic(
            self.state_dim,
            self.num_actions,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        self.target_critic = DiscreteCritic(
            self.state_dim,
            self.num_actions,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저 재설정
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> int:
        """행동 선택 - 이산 행동 반환"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor에서 행동 확률 분포 획득
            action_probs = self.actor(state)
            
            if add_noise and getattr(self, 'training', True):
                # 노이즈 추가 (확률 분포에 엔트로피 증가)
                noise = torch.randn_like(action_probs) * self.noise.sigma
                action_probs = F.softmax(action_probs + noise, dim=-1)
            
            # 가장 높은 확률의 행동 선택
            action = torch.argmax(action_probs, dim=-1).cpu().numpy()[0]
            
        return int(action)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Compatibility alias for select_action"""
        return self.select_action(state, not deterministic)  # add_noise is opposite of deterministic
        
    def update(self) -> Dict[str, float]:
        """네트워크 업데이트 - 이산 행동 버전"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 경험 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 이산 행동을 원-핫 인코딩으로 변환
        actions = torch.LongTensor(actions).to(self.device)
        actions_onehot = F.one_hot(actions, num_classes=self.num_actions).float()
        
        # Critic 업데이트
        with torch.no_grad():
            next_action_probs = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_action_probs)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions_onehot)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        if self.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                critic_loss = F.mse_loss(current_q, target_q)
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
        else:
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Actor 업데이트
        predicted_action_probs = self.actor(states)
        actor_loss = -self.critic(states, predicted_action_probs).mean()
        
        self.actor_optimizer.zero_grad()
        if self.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                predicted_action_probs = self.actor(states)
                actor_loss = -self.critic(states, predicted_action_probs).mean()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        # 노이즈 감소
        if hasattr(self.noise, 'decay'):
            self.noise.decay()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'noise_sigma': getattr(self.noise, 'sigma', 0.0)
        }
    
    def _soft_update(self, target_network, source_network):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
        
    def save_checkpoint(self, filepath: str, episode: int, avg_reward: float):
        """체크포인트 저장"""
        checkpoint = {
            'episode': episode,
            'avg_reward': avg_reward,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'num_actions': self.num_actions,
            'state_dim': self.state_dim
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        return checkpoint['episode'], checkpoint['avg_reward']