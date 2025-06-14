import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Critic(nn.Module):
    """DDPG를 위한 크리틱 네트워크
    
    상태와 행동을 입력받아 Q-값을 출력합니다.
    DQN과 달리 연속적인 행동을 직접 입력받아 처리합니다.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300]):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            hidden_dims: 은닉층 차원들
        """
        super(Critic, self).__init__()
        
        # 상태 처리 경로
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        
        # 상태와 행동을 결합한 후의 경로
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            action: 행동 텐서 [batch_size, action_dim]
            
        Returns:
            Q-값 [batch_size, 1]
        """
        # 상태 처리
        state_out = F.relu(self.fc1(state))
        
        # 상태와 행동 결합
        concat = torch.cat([state_out, action], dim=1)
        
        # Q-값 계산
        out = F.relu(self.fc2(concat))
        q_value = self.fc3(out)
        
        return q_value
    
    def _initialize_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)


class CriticWithLayerNorm(nn.Module):
    """Layer Normalization을 사용하는 크리틱 네트워크
    
    더 안정적인 학습을 위해 층 정규화를 적용합니다.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300]):
        super(CriticWithLayerNorm, self).__init__()
        
        # 상태 처리
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        
        # 상태+행동 처리
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        
        # 출력
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 상태 처리
        state_out = F.relu(self.ln1(self.fc1(state)))
        
        # 상태와 행동 결합
        concat = torch.cat([state_out, action], dim=1)
        
        # Q-값 계산
        out = F.relu(self.ln2(self.fc2(concat)))
        q_value = self.fc3(out)
        
        return q_value
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)