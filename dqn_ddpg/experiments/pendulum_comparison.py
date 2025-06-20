#!/usr/bin/env python3
"""
Pendulum-v1 환경에서 DDPG vs DQN 성능 비교 실험

DDPG가 우수한 환경에서의 비교를 통해 알고리즘 적합성을 분석합니다.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))  # experiments/
parent_dir = os.path.dirname(project_root)  # dqn_ddpg/
sys.path.insert(0, parent_dir)

from src.agents.ddpg_agent import DDPGAgent
from src.agents.dqn_agent import DQNAgent  
from src.environments.wrappers import create_ddpg_env, create_dqn_env
from src.core.utils import set_seed
import gymnasium as gym


class PendulumDQNWrapper(gym.Wrapper):
    """Pendulum 환경을 DQN용으로 이산화하는 래퍼"""
    
    def __init__(self, env: gym.Env, num_actions: int = 21):
        super().__init__(env)
        self.num_actions = num_actions
        
        # 연속 행동 [-2, 2]를 이산 행동으로 변환
        self.action_space = gym.spaces.Discrete(num_actions)
        self.action_mapping = np.linspace(-2.0, 2.0, num_actions)
        
    def step(self, action: int) -> Tuple:
        # 이산 행동을 연속 행동으로 변환
        continuous_action = np.array([self.action_mapping[action]], dtype=np.float32)
        return self.env.step(continuous_action)


def create_pendulum_dqn_env() -> gym.Env:
    """DQN용 Pendulum 환경 생성"""
    env = gym.make("Pendulum-v1")
    env = PendulumDQNWrapper(env, num_actions=21)
    return env


def train_agent(agent, env, max_episodes: int = 200, eval_frequency: int = 25) -> Dict:
    """에이전트 학습 및 성능 평가"""
    scores = []
    eval_scores = []
    eval_episodes = []
    
    print(f"Training {agent.__class__.__name__} for {max_episodes} episodes...")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 200  # Pendulum 기본 step limit
        
        while step_count < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            agent.update()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        
        # 주기적 평가
        if (episode + 1) % eval_frequency == 0:
            eval_score = evaluate_agent(agent, env, episodes=10)
            eval_scores.append(eval_score)
            eval_episodes.append(episode + 1)
            print(f"Episode {episode + 1}: Score = {total_reward:.2f}, Eval Score = {eval_score:.2f}")
    
    return {
        'scores': scores,
        'eval_scores': eval_scores,
        'eval_episodes': eval_episodes,
        'final_score': np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
        'best_score': max(scores),
        'final_eval_score': eval_scores[-1] if eval_scores else 0
    }


def evaluate_agent(agent, env, episodes: int = 10) -> float:
    """에이전트 성능 평가"""
    total_rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            action = agent.select_action(state, add_noise=False)  # 탐험 없이 평가
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)


def run_pendulum_comparison():
    """Pendulum 환경에서 DDPG vs DQN 비교 실험 실행"""
    print("🎯 Pendulum-v1 환경에서 DDPG vs DQN 성능 비교 시작")
    print("=" * 60)
    
    # 시드 설정
    seed = 42
    set_seed(seed)
    
    # 환경 설정
    ddpg_env = create_ddpg_env("Pendulum-v1")
    dqn_env = create_pendulum_dqn_env()
    
    # 에이전트 설정
    state_dim = 3  # cos(θ), sin(θ), θ_dot
    action_dim = 1  # 토크
    
    ddpg_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.005,
        buffer_size=50000,
        batch_size=64,
        noise_sigma=0.1
    )
    
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=21,  # 이산화된 행동 수
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=100
    )
    
    # 학습 실행 (빠른 테스트용으로 축소)
    max_episodes = 200
    
    print("🤖 DDPG 학습 시작...")
    ddpg_results = train_agent(ddpg_agent, ddpg_env, max_episodes)
    
    print("\n🤖 DQN 학습 시작...")
    dqn_results = train_agent(dqn_agent, dqn_env, max_episodes)
    
    # 결과 정리
    results = {
        'experiment_config': {
            'environment': 'Pendulum-v1',
            'max_episodes': max_episodes,
            'seed': seed,
            'ddpg_config': {
                'actor_lr': 0.0001,
                'critic_lr': 0.001,
                'gamma': 0.99,
                'tau': 0.005,
                'noise_sigma': 0.1,
                'buffer_size': 50000,
                'batch_size': 64
            },
            'dqn_config': {
                'learning_rate': 0.001,
                'epsilon_decay': 0.995,
                'gamma': 0.99,
                'buffer_size': 50000,
                'batch_size': 64,
                'num_actions': 21
            }
        },
        'training_results': {
            'ddpg': ddpg_results,
            'dqn': dqn_results
        },
        'comparison': {
            'ddpg_final': ddpg_results['final_score'],
            'dqn_final': dqn_results['final_score'],
            'performance_ratio': abs(ddpg_results['final_score'] / dqn_results['final_score']) if dqn_results['final_score'] != 0 else float('inf'),
            'ddpg_advantage': ddpg_results['final_score'] > dqn_results['final_score']
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 실험 결과 요약")
    print("=" * 60)
    print(f"DDPG 최종 성능: {ddpg_results['final_score']:.2f}")
    print(f"DQN 최종 성능:  {dqn_results['final_score']:.2f}")
    print(f"성능 비율:      {results['comparison']['performance_ratio']:.2f}x")
    print(f"DDPG 우위:      {'✅' if results['comparison']['ddpg_advantage'] else '❌'}")
    
    # 결과 저장
    os.makedirs('results/pendulum_comparison', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/pendulum_comparison/pendulum_results_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {result_file}")
    
    # 시각화 생성
    create_pendulum_visualizations(results, timestamp)
    
    return results


def create_pendulum_visualizations(results: Dict, timestamp: str):
    """Pendulum 실험 결과 시각화 생성"""
    ddpg_scores = results['training_results']['ddpg']['scores']
    dqn_scores = results['training_results']['dqn']['scores']
    
    # 1. 학습 곡선 비교
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    episodes = range(1, len(ddpg_scores) + 1)
    plt.plot(episodes, ddpg_scores, label='DDPG', color='blue', alpha=0.7)
    plt.plot(episodes, dqn_scores, label='DQN', color='red', alpha=0.7)
    
    # 이동 평균 추가
    window = 50
    if len(ddpg_scores) >= window:
        ddpg_ma = np.convolve(ddpg_scores, np.ones(window)/window, mode='valid')
        dqn_ma = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
        ma_episodes = range(window, len(ddpg_scores) + 1)
        plt.plot(ma_episodes, ddpg_ma, label='DDPG (MA50)', color='darkblue', linewidth=2)
        plt.plot(ma_episodes, dqn_ma, label='DQN (MA50)', color='darkred', linewidth=2)
    
    plt.title('Pendulum-v1: Learning Curves Comparison')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 최종 성능 비교
    plt.subplot(2, 2, 2)
    algorithms = ['DDPG', 'DQN']
    final_scores = [
        results['comparison']['ddpg_final'],
        results['comparison']['dqn_final']
    ]
    colors = ['blue', 'red']
    
    bars = plt.bar(algorithms, final_scores, color=colors, alpha=0.7)
    plt.title('Final Performance Comparison')
    plt.ylabel('Average Reward (Last 100 episodes)')
    
    # 값 표시
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. 성능 분포 히스토그램
    plt.subplot(2, 2, 3)
    plt.hist(ddpg_scores[-100:], bins=20, alpha=0.5, label='DDPG', color='blue', density=True)
    plt.hist(dqn_scores[-100:], bins=20, alpha=0.5, label='DQN', color='red', density=True)
    plt.title('Performance Distribution (Last 100 episodes)')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 통계 정보
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
Pendulum-v1 Experiment Results

DDPG Performance:
• Final Score: {results['comparison']['ddpg_final']:.2f}
• Best Score: {results['training_results']['ddpg']['best_score']:.2f}
• Std Dev: {np.std(ddpg_scores[-100:]):.2f}

DQN Performance:
• Final Score: {results['comparison']['dqn_final']:.2f}
• Best Score: {results['training_results']['dqn']['best_score']:.2f}
• Std Dev: {np.std(dqn_scores[-100:]):.2f}

Comparison:
• Performance Ratio: {results['comparison']['performance_ratio']:.2f}x
• DDPG Advantage: {'✅ Yes' if results['comparison']['ddpg_advantage'] else '❌ No'}

Environment: Pendulum-v1 (Continuous Control)
Episodes: {results['experiment_config']['max_episodes']}
Timestamp: {results['timestamp']}
"""
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    
    # 저장
    viz_file = f'results/pendulum_comparison/pendulum_visualization_{timestamp}.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 시각화 저장: {viz_file}")
    
    plt.show()


if __name__ == "__main__":
    results = run_pendulum_comparison()