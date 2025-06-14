import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Actor(nn.Module):
    """DDPG를 위한 액터 네트워크
    
    상태를 입력받아 결정적 행동을 직접 출력합니다.
    이는 DDPG의 핵심으로, 명시적 결정적 정책을 구현합니다.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300],
                 action_bound: float = 1.0):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원 (연속)
            hidden_dims: 은닉층 차원들
            action_bound: 행동의 최대 절댓값
        """
        super(Actor, self).__init__()
        
        self.action_bound = action_bound
        
        # 네트워크 구성
        layers = []
        prev_dim = state_dim
        
        # 은닉층
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i == 0:
                # 첫 번째 층 후에만 배치 정규화 (선택적)
                # layers.append(nn.BatchNorm1d(hidden_dim))
                pass
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.actor_net = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            
        Returns:
            결정적 행동 [batch_size, action_dim], 범위: [-action_bound, action_bound]
        """
        # 네트워크 출력
        action = self.actor_net(state)
        
        # tanh를 사용하여 행동을 제한된 범위로 매핑
        action = torch.tanh(action) * self.action_bound
        
        return action
    
    def _initialize_weights(self):
        """가중치 초기화
        
        마지막 층은 작은 값으로 초기화하여 초기 행동이 0 근처에 있도록 합니다.
        """
        for i, layer in enumerate(self.actor_net):
            if isinstance(layer, nn.Linear):
                if i == len(self.actor_net) - 1:  # 마지막 층
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    nn.init.uniform_(layer.bias, -3e-3, 3e-3)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)


class ActorWithLayerNorm(nn.Module):
    """Layer Normalization을 사용하는 액터 네트워크
    
    배치 정규화 대신 층 정규화를 사용하여 더 안정적인 학습을 제공합니다.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300],
                 action_bound: float = 1.0):
        super(ActorWithLayerNorm, self).__init__()
        
        self.action_bound = action_bound
        
        # 첫 번째 은닉층
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        
        # 두 번째 은닉층
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        
        # 출력층
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)
        
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)