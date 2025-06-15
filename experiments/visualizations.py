"""
시각화 도구들
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import os


def plot_learning_curves(dqn_metrics: Dict, ddpg_metrics: Dict, save_path: str = None):
    """학습 곡선 플로팅"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN vs DDPG Learning Performance Comparison', fontsize=16)
    
    # 1. 에피소드 보상
    ax = axes[0, 0]
    if 'episode_rewards' in dqn_metrics:
        episodes_dqn = range(len(dqn_metrics['episode_rewards']))
        ax.plot(episodes_dqn, dqn_metrics['episode_rewards'], 
                label='DQN', alpha=0.7, color='blue')
        # 이동 평균
        window = min(50, len(dqn_metrics['episode_rewards'])//10)
        if window > 1:
            ma_rewards = np.convolve(dqn_metrics['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(dqn_metrics['episode_rewards'])), 
                   ma_rewards, color='blue', linewidth=2)
    
    if 'episode_rewards' in ddpg_metrics:
        episodes_ddpg = range(len(ddpg_metrics['episode_rewards']))
        ax.plot(episodes_ddpg, ddpg_metrics['episode_rewards'], 
                label='DDPG', alpha=0.7, color='red')
        # 이동 평균
        window = min(50, len(ddpg_metrics['episode_rewards'])//10)
        if window > 1:
            ma_rewards = np.convolve(ddpg_metrics['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(ddpg_metrics['episode_rewards'])), 
                   ma_rewards, color='red', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 에피소드 길이
    ax = axes[0, 1]
    if 'episode_lengths' in dqn_metrics:
        ax.plot(dqn_metrics['episode_lengths'], label='DQN', alpha=0.7, color='blue')
    if 'episode_lengths' in ddpg_metrics:
        ax.plot(ddpg_metrics['episode_lengths'], label='DDPG', alpha=0.7, color='red')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 학습 손실
    ax = axes[1, 0]
    if 'training_losses' in dqn_metrics and dqn_metrics['training_losses']:
        ax.plot(dqn_metrics['training_losses'], label='DQN', alpha=0.7, color='blue')
    if 'training_losses' in ddpg_metrics and ddpg_metrics['training_losses']:
        ax.plot(ddpg_metrics['training_losses'], label='DDPG (Critic)', alpha=0.7, color='red')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Q-값 변화
    ax = axes[1, 1]
    if 'q_values' in dqn_metrics and dqn_metrics['q_values']:
        ax.plot(dqn_metrics['q_values'], label='DQN Q-values', alpha=0.7, color='blue')
    if 'q_values' in ddpg_metrics and ddpg_metrics['q_values']:
        ax.plot(ddpg_metrics['q_values'], label='DDPG Q-values', alpha=0.7, color='red')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Q-value')
    ax.set_title('Q-Value Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_deterministic_policy(dqn_agent, ddpg_agent, 
                                  dqn_env, ddpg_env, 
                                  save_path: str = None):
    """결정적 정책 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('결정적 정책 특성 분석', fontsize=16)
    
    # DQN 분석
    # 1. Q-값 히트맵
    ax = axes[0, 0]
    states = np.array([dqn_env.observation_space.sample() for _ in range(50)])
    q_values_matrix = []
    for state in states:
        q_vals = dqn_agent.get_q_values(state)
        q_values_matrix.append(q_vals)
    
    q_values_matrix = np.array(q_values_matrix)
    im = ax.imshow(q_values_matrix.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('State Index')
    ax.set_ylabel('Action')
    ax.set_title('DQN: Q-값 분포')
    plt.colorbar(im, ax=ax)
    
    # 2. 행동 선택 분포
    ax = axes[0, 1]
    selected_actions = [np.argmax(q_vals) for q_vals in q_values_matrix]
    action_counts = np.bincount(selected_actions, minlength=dqn_env.action_space.n)
    ax.bar(range(len(action_counts)), action_counts)
    ax.set_xlabel('Action')
    ax.set_ylabel('Selection Count')
    ax.set_title('DQN: 행동 선택 분포')
    
    # 3. Q-값 차이 분석
    ax = axes[0, 2]
    q_differences = []
    for q_vals in q_values_matrix:
        sorted_q = np.sort(q_vals)[::-1]
        if len(sorted_q) > 1:
            q_differences.append(sorted_q[0] - sorted_q[1])
    
    ax.hist(q_differences, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Q-value Difference (Best - Second Best)')
    ax.set_ylabel('Frequency')
    ax.set_title('DQN: Q-값 차이 분포')
    
    # DDPG 분석
    # 4. 액터 출력 분포
    ax = axes[1, 0]
    states = np.array([ddpg_env.observation_space.sample() for _ in range(100)])
    actions = []
    for state in states:
        action = ddpg_agent.get_deterministic_action(state)
        actions.append(action)
    
    actions = np.array(actions)
    for i in range(actions.shape[1]):  # 각 행동 차원별로
        ax.hist(actions[:, i], bins=20, alpha=0.7, 
               label=f'Action Dim {i}', edgecolor='black')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Frequency')
    ax.set_title('DDPG: 액터 출력 분포')
    ax.legend()
    
    # 5. 행동 일관성 테스트
    ax = axes[1, 1]
    test_state = ddpg_env.observation_space.sample()
    repeated_actions = []
    for _ in range(20):
        action = ddpg_agent.get_deterministic_action(test_state)
        repeated_actions.append(action)
    
    repeated_actions = np.array(repeated_actions)
    std_per_dim = np.std(repeated_actions, axis=0)
    ax.bar(range(len(std_per_dim)), std_per_dim)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('DDPG: 행동 일관성 (낮을수록 결정적)')
    ax.set_yscale('log')
    
    # 6. 노이즈의 영향
    ax = axes[1, 2]
    deterministic_actions = []
    noisy_actions = []
    
    for state in states[:20]:
        det_action = ddpg_agent.select_action(state, add_noise=False)
        noisy_action = ddpg_agent.select_action(state, add_noise=True)
        
        deterministic_actions.append(det_action)
        noisy_actions.append(noisy_action)
    
    deterministic_actions = np.array(deterministic_actions)
    noisy_actions = np.array(noisy_actions)
    
    differences = np.linalg.norm(noisy_actions - deterministic_actions, axis=1)
    ax.hist(differences, bins=15, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Action Difference (L2 norm)')
    ax.set_ylabel('Frequency')
    ax.set_title('DDPG: 노이즈로 인한 행동 변화')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_comparison_summary(dqn_eval: Dict, ddpg_eval: Dict, save_path: str = None):
    """최종 비교 요약 플롯"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('DQN vs DDPG 최종 성능 비교', fontsize=16)
    
    algorithms = ['DQN', 'DDPG']
    
    # 1. 평균 보상 비교
    ax = axes[0]
    rewards = [dqn_eval.get('mean_reward', 0), ddpg_eval.get('mean_reward', 0)]
    errors = [dqn_eval.get('std_reward', 0), ddpg_eval.get('std_reward', 0)]
    
    bars = ax.bar(algorithms, rewards, yerr=errors, capsize=5, 
                  color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Average Reward')
    ax.set_title('평균 보상')
    ax.grid(True, alpha=0.3)
    
    # 2. 에피소드 길이 비교
    ax = axes[1]
    lengths = [dqn_eval.get('mean_length', 0), ddpg_eval.get('mean_length', 0)]
    
    bars = ax.bar(algorithms, lengths, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Average Episode Length')
    ax.set_title('평균 에피소드 길이')
    ax.grid(True, alpha=0.3)
    
    # 3. 성공률 비교
    ax = axes[2]
    success_rates = [dqn_eval.get('success_rate', 0), ddpg_eval.get('success_rate', 0)]
    
    bars = ax.bar(algorithms, success_rates, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Success Rate')
    ax.set_title('성공률')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_experiment_report(results: Dict, save_path: str):
    """실험 결과 리포트 생성"""
    # 모든 시각화를 하나의 PDF로 저장하는 등의 기능
    # 여기서는 간단히 텍스트 리포트 생성
    
    report = f"""
# DQN vs DDPG 비교 실험 결과 리포트

## 실험 개요
- DQN: 이산적 행동 공간 (CartPole-v1)
- DDPG: 연속적 행동 공간 (Pendulum-v1)

## 주요 결과

### DQN 결과
- 평균 보상: {results.get('dqn_eval', {}).get('mean_reward', 'N/A'):.2f}
- 표준편차: {results.get('dqn_eval', {}).get('std_reward', 'N/A'):.2f}
- 평균 에피소드 길이: {results.get('dqn_eval', {}).get('mean_length', 'N/A'):.1f}

### DDPG 결과
- 평균 보상: {results.get('ddpg_eval', {}).get('mean_reward', 'N/A'):.2f}
- 표준편차: {results.get('ddpg_eval', {}).get('std_reward', 'N/A'):.2f}
- 평균 에피소드 길이: {results.get('ddpg_eval', {}).get('mean_length', 'N/A'):.1f}

## 결정적 정책 특성

### DQN (암묵적 결정적 정책)
- Q-값 기반 행동 선택
- argmax를 통한 결정적 행동
- 탐험: ε-greedy

### DDPG (명시적 결정적 정책)
- 액터 네트워크 직접 출력
- 연속 행동 공간 지원
- 탐험: 가우시안 노이즈

## 결론
두 알고리즘 모두 각자의 도메인에서 결정적 정책을 성공적으로 구현했습니다.
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)