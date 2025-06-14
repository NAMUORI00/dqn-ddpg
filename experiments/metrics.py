"""
학습 성능 메트릭 및 평가 함수들
"""
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import deque


class MetricsTracker:
    """학습 메트릭 추적 클래스"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.q_values = []
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
    
    def add_episode(self, reward: float, length: int):
        """에피소드 결과 추가"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
    
    def add_training_step(self, metrics: Dict[str, float]):
        """학습 스텝 메트릭 추가"""
        if 'loss' in metrics:
            self.training_losses.append(metrics['loss'])
        if 'q_value' in metrics:
            self.q_values.append(metrics['q_value'])
    
    def get_current_stats(self) -> Dict[str, float]:
        """현재 통계 반환"""
        stats = {}
        
        if self.recent_rewards:
            stats['mean_reward'] = np.mean(self.recent_rewards)
            stats['std_reward'] = np.std(self.recent_rewards)
            stats['min_reward'] = np.min(self.recent_rewards)
            stats['max_reward'] = np.max(self.recent_rewards)
        
        if self.recent_lengths:
            stats['mean_length'] = np.mean(self.recent_lengths)
        
        if self.training_losses:
            stats['mean_loss'] = np.mean(self.training_losses[-100:])
        
        if self.q_values:
            stats['mean_q_value'] = np.mean(self.q_values[-100:])
        
        return stats
    
    def is_solved(self, threshold: float, episodes: int = 100) -> bool:
        """문제 해결 여부 확인"""
        if len(self.recent_rewards) < episodes:
            return False
        return np.mean(list(self.recent_rewards)[-episodes:]) >= threshold


def evaluate_agent(agent, env, episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
    """에이전트 평가
    
    Args:
        agent: 평가할 에이전트
        env: 평가 환경
        episodes: 평가 에피소드 수
        deterministic: 결정적 정책 사용 여부
        
    Returns:
        평가 결과 딕셔너리
    """
    rewards = []
    lengths = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 에이전트 유형에 따른 행동 선택
            if hasattr(agent, 'select_action'):
                if hasattr(agent, 'get_deterministic_action'):  # DDPG
                    action = agent.get_deterministic_action(state) if deterministic else agent.select_action(state)
                else:  # DQN
                    action = agent.select_action(state, deterministic=deterministic)
            else:
                raise ValueError("Unknown agent type")
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'success_rate': np.sum(np.array(rewards) > 0) / len(rewards)
    }


def compute_action_consistency(agent, states: np.ndarray, trials: int = 10) -> float:
    """행동 일관성 계산
    
    동일한 상태에서 여러 번 행동을 선택했을 때의 일관성을 측정합니다.
    
    Args:
        agent: 테스트할 에이전트
        states: 테스트 상태들
        trials: 각 상태에서 반복 횟수
        
    Returns:
        일관성 점수 (0~1)
    """
    consistent_states = 0
    
    for state in states:
        actions = []
        
        for _ in range(trials):
            if hasattr(agent, 'get_deterministic_action'):  # DDPG
                action = agent.get_deterministic_action(state)
            else:  # DQN
                action = agent.select_action(state, deterministic=True)
            actions.append(action)
        
        # 일관성 확인
        if hasattr(agent, 'get_deterministic_action'):  # DDPG (연속)
            actions_array = np.array(actions)
            std = np.std(actions_array, axis=0)
            is_consistent = np.all(std < 1e-6)
        else:  # DQN (이산)
            is_consistent = len(set(actions)) == 1
        
        if is_consistent:
            consistent_states += 1
    
    return consistent_states / len(states)


def measure_exploration_diversity(agent, state: np.ndarray, samples: int = 100) -> Dict[str, float]:
    """탐험 다양성 측정
    
    Args:
        agent: 테스트할 에이전트
        state: 테스트 상태
        samples: 샘플 수
        
    Returns:
        다양성 메트릭
    """
    actions = []
    
    for _ in range(samples):
        if hasattr(agent, 'select_action'):
            if hasattr(agent, 'get_deterministic_action'):  # DDPG
                action = agent.select_action(state, add_noise=True)
            else:  # DQN
                # 일시적으로 epsilon 설정
                original_epsilon = getattr(agent, 'epsilon', 0.0)
                agent.epsilon = 0.3
                action = agent.select_action(state, deterministic=False)
                agent.epsilon = original_epsilon
        actions.append(action)
    
    if hasattr(agent, 'get_deterministic_action'):  # DDPG
        actions_array = np.array(actions)
        diversity = {
            'action_std': np.mean(np.std(actions_array, axis=0)),
            'action_range': np.mean(np.max(actions_array, axis=0) - np.min(actions_array, axis=0)),
            'unique_ratio': len(np.unique(actions_array.round(3), axis=0)) / len(actions_array)
        }
    else:  # DQN
        unique_actions = len(set(actions))
        total_actions = len(actions)
        diversity = {
            'unique_actions': unique_actions,
            'diversity_ratio': unique_actions / total_actions,
            'entropy': -sum(actions.count(a)/total_actions * np.log2(actions.count(a)/total_actions) 
                           for a in set(actions))
        }
    
    return diversity