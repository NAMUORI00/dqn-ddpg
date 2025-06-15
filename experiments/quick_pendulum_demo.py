#!/usr/bin/env python3
"""
Pendulum-v1 환경에서 DDPG vs DQN 빠른 데모

시간을 단축하여 DDPG 우위를 빠르게 확인하는 데모입니다.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.agents.ddpg_agent import DDPGAgent
from src.agents.dqn_agent import DQNAgent  
from src.environments.wrappers import create_ddpg_env
from src.core.utils import set_seed
import gymnasium as gym


class PendulumDQNWrapper(gym.Wrapper):
    """Pendulum 환경을 DQN용으로 이산화하는 래퍼"""
    
    def __init__(self, env: gym.Env, num_actions: int = 11):  # 더 적은 행동으로 빠른 학습
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
    env = PendulumDQNWrapper(env, num_actions=11)
    return env


def quick_test_agent(agent, env, episodes: int = 20) -> Dict:
    """에이전트 빠른 테스트"""
    scores = []
    
    print(f"Testing {agent.__class__.__name__} for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            # 행동 선택 (DQN은 epsilon-greedy, DDPG는 노이즈 추가)
            if hasattr(agent, 'select_action'):
                if 'DDPG' in agent.__class__.__name__:
                    action = agent.select_action(state, add_noise=True)
                else:  # DQN
                    action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 경험 저장 및 학습
            if hasattr(agent, 'store_transition'):
                agent.store_transition(state, action, reward, next_state, terminated or truncated)
                agent.update()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        
        if (episode + 1) % 5 == 0:
            recent_avg = np.mean(scores[-5:])
            print(f"Episode {episode + 1}: Recent avg = {recent_avg:.2f}")
    
    return {
        'scores': scores,
        'final_score': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
        'best_score': max(scores),
        'worst_score': min(scores),
        'std_score': np.std(scores)
    }


def evaluate_final_performance(agent, env, episodes: int = 10) -> float:
    """최종 성능 평가 (노이즈 없음)"""
    total_rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            # 평가 시에는 노이즈 없음
            if hasattr(agent, 'select_action'):
                if 'DDPG' in agent.__class__.__name__:
                    action = agent.select_action(state, add_noise=False)
                else:  # DQN
                    action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)


def run_quick_pendulum_demo():
    """빠른 Pendulum 데모 실행"""
    print("🎯 Pendulum-v1 빠른 데모: DDPG vs DQN")
    print("=" * 50)
    
    # 시드 설정
    seed = 42
    set_seed(seed)
    
    # 환경 설정
    ddpg_env = create_ddpg_env("Pendulum-v1")
    dqn_env = create_pendulum_dqn_env()
    
    # 에이전트 설정 (더 간단한 설정)
    state_dim = 3
    action_dim = 1
    
    ddpg_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=0.001,  # 더 빠른 학습
        critic_lr=0.002,
        gamma=0.99,
        tau=0.01,  # 더 빠른 타겟 업데이트
        buffer_size=10000,  # 더 작은 버퍼
        batch_size=32,
        noise_sigma=0.2
    )
    
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=11,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,  # 더 빠른 탐험 감소
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=50  # 더 자주 업데이트
    )
    
    # 빠른 학습 테스트
    training_episodes = 50
    
    print("🤖 DDPG 빠른 학습...")
    ddpg_results = quick_test_agent(ddpg_agent, ddpg_env, training_episodes)
    
    print("\n🤖 DQN 빠른 학습...")
    dqn_results = quick_test_agent(dqn_agent, dqn_env, training_episodes)
    
    # 최종 성능 평가
    print("\n📊 최종 성능 평가...")
    ddpg_final = evaluate_final_performance(ddpg_agent, ddpg_env)
    dqn_final = evaluate_final_performance(dqn_agent, dqn_env)
    
    # 결과 정리
    results = {
        'experiment_config': {
            'environment': 'Pendulum-v1',
            'training_episodes': training_episodes,
            'evaluation_episodes': 10,
            'seed': seed,
            'note': 'Quick demo for DDPG advantage demonstration'
        },
        'training_results': {
            'ddpg': ddpg_results,
            'dqn': dqn_results
        },
        'final_evaluation': {
            'ddpg_final': ddpg_final,
            'dqn_final': dqn_final,
            'performance_ratio': abs(ddpg_final / dqn_final) if dqn_final != 0 else float('inf'),
            'ddpg_advantage': bool(ddpg_final > dqn_final)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📊 빠른 데모 결과")
    print("=" * 50)
    print(f"DDPG 학습 성능: {ddpg_results['final_score']:.2f}")
    print(f"DQN 학습 성능:  {dqn_results['final_score']:.2f}")
    print(f"DDPG 최종 평가: {ddpg_final:.2f}")
    print(f"DQN 최종 평가:  {dqn_final:.2f}")
    print(f"성능 비율:      {results['final_evaluation']['performance_ratio']:.2f}x")
    print(f"DDPG 우위:      {'✅' if results['final_evaluation']['ddpg_advantage'] else '❌'}")
    
    # 결과 저장
    os.makedirs('results/pendulum_comparison', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/pendulum_comparison/quick_demo_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {result_file}")
    
    # 간단한 시각화
    create_quick_visualization(results, timestamp)
    
    return results


def create_quick_visualization(results: Dict, timestamp: str):
    """빠른 시각화 생성"""
    ddpg_scores = results['training_results']['ddpg']['scores']
    dqn_scores = results['training_results']['dqn']['scores']
    
    plt.figure(figsize=(12, 6))
    
    # 1. 학습 곡선
    plt.subplot(1, 2, 1)
    episodes = range(1, len(ddpg_scores) + 1)
    plt.plot(episodes, ddpg_scores, label='DDPG', color='blue', alpha=0.7)
    plt.plot(episodes, dqn_scores, label='DQN', color='red', alpha=0.7)
    
    # 이동 평균
    window = 10
    if len(ddpg_scores) >= window:
        ddpg_ma = np.convolve(ddpg_scores, np.ones(window)/window, mode='valid')
        dqn_ma = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
        ma_episodes = range(window, len(ddpg_scores) + 1)
        plt.plot(ma_episodes, ddpg_ma, label='DDPG (MA10)', color='darkblue', linewidth=2)
        plt.plot(ma_episodes, dqn_ma, label='DQN (MA10)', color='darkred', linewidth=2)
    
    plt.title('Pendulum-v1: Quick Learning Demo')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 최종 성능 비교
    plt.subplot(1, 2, 2)
    algorithms = ['DDPG', 'DQN']
    final_scores = [
        results['final_evaluation']['ddpg_final'],
        results['final_evaluation']['dqn_final']
    ]
    colors = ['blue', 'red']
    
    bars = plt.bar(algorithms, final_scores, color=colors, alpha=0.7)
    plt.title('Final Performance (No Noise Evaluation)')
    plt.ylabel('Average Reward')
    
    # 값 표시
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 성능 비율 추가 표시
    ratio = results['final_evaluation']['performance_ratio']
    plt.text(0.5, max(final_scores) * 0.8, f'Ratio: {ratio:.1f}x', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # 저장
    viz_file = f'results/pendulum_comparison/quick_demo_viz_{timestamp}.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 시각화 저장: {viz_file}")
    
    plt.show()


if __name__ == "__main__":
    results = run_quick_pendulum_demo()