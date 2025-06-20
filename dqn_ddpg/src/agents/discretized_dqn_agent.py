"""
DiscretizedDQNAgent - 연속 행동 공간에서 작동하는 DQN 변형

기존 DQN을 연속 행동 공간에서 사용할 수 있도록 행동 이산화를 통해 확장한 에이전트입니다.
DQN과 DDPG를 동일한 연속 환경에서 공정하게 비교하기 위해 설계되었습니다.

핵심 아이디어:
- 연속 행동 공간을 N개의 이산 구간으로 분할
- DQN의 이산 행동 선택을 연속 값으로 매핑
- 기존 DQN의 모든 학습 메커니즘 유지
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from .dqn_agent import DQNAgent


class DiscretizedDQNAgent(DQNAgent):
    """연속 행동 공간용 이산화 DQN 에이전트
    
    연속 행동 공간을 이산 구간으로 나누어 기존 DQN을 적용합니다.
    이를 통해 DQN과 DDPG를 동일한 연속 환경에서 비교할 수 있습니다.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_bound: float = 1.0,
                 num_actions: int = 21,  # 홀수 추천 (0 포함)
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
            action_bound: 연속 행동의 최대 절댓값 ([-action_bound, +action_bound])
            num_actions: 이산화할 행동의 개수 (권장: 홀수, 0 포함)
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
        # 부모 클래스 초기화 (action_dim을 num_actions로 설정)
        super().__init__(
            state_dim=state_dim,
            action_dim=num_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device
        )
        
        self.action_bound = action_bound
        self.num_actions = num_actions
        
        # 이산 행동 인덱스를 연속 값으로 매핑하는 테이블 생성
        self.action_values = self._create_action_mapping()
        
        print(f"DiscretizedDQNAgent 초기화:")
        print(f"  연속 행동 범위: [{-action_bound:.2f}, {+action_bound:.2f}]")
        print(f"  이산화 개수: {num_actions}")
        print(f"  행동 매핑: {self.action_values}")
    
    def _create_action_mapping(self) -> np.ndarray:
        """이산 행동 인덱스를 연속 값으로 매핑하는 테이블 생성
        
        Returns:
            행동 매핑 테이블 (shape: [num_actions])
        """
        # [-action_bound, +action_bound] 범위를 num_actions개로 균등 분할
        action_values = np.linspace(-self.action_bound, self.action_bound, self.num_actions)
        return action_values.astype(np.float32)
    
    def discrete_to_continuous(self, discrete_action: int) -> np.ndarray:
        """이산 행동을 연속 행동으로 변환
        
        Args:
            discrete_action: 이산 행동 인덱스 (0 ~ num_actions-1)
            
        Returns:
            연속 행동 값 (shape: [1])
        """
        return np.array([self.action_values[discrete_action]], dtype=np.float32)
    
    def continuous_to_discrete(self, continuous_action: float) -> int:
        """연속 행동을 가장 가까운 이산 행동으로 변환
        
        Args:
            continuous_action: 연속 행동 값
            
        Returns:
            가장 가까운 이산 행동 인덱스
        """
        # 가장 가까운 이산 값 찾기
        distances = np.abs(self.action_values - continuous_action)
        return int(np.argmin(distances))
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """행동 선택 (연속 값 반환)
        
        DQN의 이산 행동 선택을 수행한 후 연속 값으로 변환합니다.
        
        Args:
            state: 현재 상태
            deterministic: True면 항상 최적 행동 선택 (평가용)
            
        Returns:
            선택된 연속 행동 (shape: [1])
        """
        # 부모 클래스의 이산 행동 선택
        discrete_action = super().select_action(state, deterministic)
        
        # 연속 값으로 변환
        continuous_action = self.discrete_to_continuous(discrete_action)
        
        return continuous_action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """경험 저장
        
        연속 행동을 이산 행동으로 변환하여 저장합니다.
        
        Args:
            state: 현재 상태
            action: 연속 행동 (shape: [1])
            reward: 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        # 연속 행동을 이산 행동으로 변환
        continuous_value = action[0] if isinstance(action, np.ndarray) else action
        discrete_action = self.continuous_to_discrete(continuous_value)
        
        # 부모 클래스의 저장 메서드 호출
        super().store_transition(state, discrete_action, reward, next_state, done)
    
    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """순수 결정적 행동 반환 (DDPG와의 비교를 위한 메서드)
        
        노이즈 없이 Q-값이 가장 높은 행동을 연속 값으로 반환합니다.
        """
        return self.select_action(state, deterministic=True)
    
    def get_action_distribution(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """주어진 상태에서 모든 이산 행동의 Q-값과 연속 값 반환
        
        결정적 정책 분석을 위한 메서드입니다.
        
        Returns:
            Tuple[Q-값 배열, 연속 행동 값 배열]
        """
        q_values = super().get_q_values(state)
        return q_values, self.action_values
    
    def analyze_policy_determinism(self, states: np.ndarray, num_samples: int = 100) -> Dict[str, float]:
        """정책의 결정성 분석
        
        동일한 상태에서 반복 행동 선택 시 일관성을 측정합니다.
        
        Args:
            states: 분석할 상태들
            num_samples: 각 상태당 샘플링 횟수
            
        Returns:
            결정성 분석 결과
        """
        determinism_scores = []
        q_value_stabilities = []
        
        for state in states:
            # 동일 상태에서 여러 번 행동 선택
            actions = []
            q_values_list = []
            
            for _ in range(num_samples):
                action = self.select_action(state, deterministic=True)
                actions.append(action[0])
                
                q_values = super().get_q_values(state)
                q_values_list.append(q_values)
            
            # 결정성 점수: 동일한 행동 선택 비율
            unique_actions = len(set(actions))
            determinism_score = 1.0 if unique_actions == 1 else 0.0
            determinism_scores.append(determinism_score)
            
            # Q-값 안정성: Q-값의 분산
            q_values_array = np.array(q_values_list)
            q_value_stability = np.mean(np.var(q_values_array, axis=0))
            q_value_stabilities.append(q_value_stability)
        
        return {
            'determinism_score': np.mean(determinism_scores),
            'q_value_stability': np.mean(q_value_stabilities),
            'consistency_rate': np.mean(determinism_scores),
            'decision_confidence': np.mean([np.max(super().get_q_values(state)) for state in states])
        }
    
    def compare_with_continuous_action(self, continuous_action: float) -> Dict[str, float]:
        """연속 행동과 이산화된 행동 비교
        
        Args:
            continuous_action: 비교할 연속 행동 값
            
        Returns:
            비교 결과
        """
        # 가장 가까운 이산 행동 찾기
        discrete_idx = self.continuous_to_discrete(continuous_action)
        discrete_value = self.action_values[discrete_idx]
        
        # 이산화 오차 계산
        discretization_error = abs(continuous_action - discrete_value)
        
        # 해상도 분석
        action_resolution = 2 * self.action_bound / (self.num_actions - 1)
        
        return {
            'original_action': continuous_action,
            'discretized_action': discrete_value,
            'discretization_error': discretization_error,
            'action_resolution': action_resolution,
            'error_ratio': discretization_error / action_resolution
        }
    
    def get_info(self) -> Dict[str, any]:
        """에이전트 정보 반환"""
        info = {
            'agent_type': 'DiscretizedDQN',
            'policy_type': 'implicit_deterministic',
            'action_space': 'continuous_via_discretization',
            'action_bound': self.action_bound,
            'num_discrete_actions': self.num_actions,
            'action_resolution': 2 * self.action_bound / (self.num_actions - 1),
            'action_mapping': self.action_values.tolist(),
            'exploration_strategy': 'epsilon_greedy',
            'current_epsilon': self.epsilon
        }
        return info