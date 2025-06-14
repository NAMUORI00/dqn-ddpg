import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QNetwork(nn.Module):
    """DQN을 위한 Q-네트워크
    
    상태를 입력받아 각 이산 행동에 대한 Q-값을 출력합니다.
    이를 통해 암묵적으로 결정적 정책을 구현합니다.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 개수 (이산)
            hidden_dims: 은닉층 차원들
        """
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # 은닉층 구성
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 출력층: 각 행동에 대한 Q-값
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.q_network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            
        Returns:
            Q-값들 [batch_size, action_dim]
        """
        return self.q_network(state)
    
    def _initialize_weights(self):
        """Xavier 초기화"""
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)


class DuelingQNetwork(nn.Module):
    """Dueling DQN을 위한 Q-네트워크
    
    상태 가치(V)와 이점(A)을 분리하여 학습하는 개선된 아키텍처입니다.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(DuelingQNetwork, self).__init__()
        
        # 공통 특징 추출층
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # 가치 스트림
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # 이점 스트림
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in [self.feature_layer, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)