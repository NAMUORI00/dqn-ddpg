#!/usr/bin/env python3
"""
간단한 DQN vs DDPG 학습 및 시각화 스크립트
학습, 평가, 비디오 녹화를 포함한 전체 파이프라인
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import gymnasium as gym

from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import set_seed, get_device
from src.core.video_manager import VideoConfig, VideoManager


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_dqn(config: dict, episodes: int = 100) -> tuple:
    """DQN 훈련"""
    print("\n" + "="*50)
    print("DQN 훈련 시작")
    print("="*50)
    
    # 환경 생성
    env = create_dqn_env(config['environment']['name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 에이전트 생성
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        epsilon_decay=config['agent']['epsilon_decay'],
        buffer_size=config['agent']['buffer_size'],
        batch_size=config['agent']['batch_size']
    )
    
    # 학습 메트릭
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    
    # 학습 루프
    for episode in tqdm(range(episodes), desc="DQN Training"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        recent_rewards.append(total_reward)
        
        # 타겟 네트워크 업데이트
        if episode % config['agent'].get('target_update_freq', 100) == 0:
            agent.update_target_network()
        
        # 진행 상황 출력
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'algorithm': 'DQN'
    }


def train_ddpg(config: dict, episodes: int = 100) -> tuple:
    """DDPG 훈련"""
    print("\n" + "="*50)
    print("DDPG 훈련 시작")
    print("="*50)
    
    # 환경 생성
    env = create_ddpg_env(config['environment']['name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 에이전트 생성
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=config['agent']['actor_lr'],
        critic_lr=config['agent']['critic_lr'],
        gamma=config['agent']['gamma'],
        tau=config['agent']['tau'],
        noise_sigma=config['agent']['noise_sigma'],
        buffer_size=config['agent']['buffer_size'],
        batch_size=config['agent']['batch_size']
    )
    
    # 학습 메트릭
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    
    # 학습 루프
    for episode in tqdm(range(episodes), desc="DDPG Training"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            # Warmup 이후 학습 시작
            if len(agent.buffer) > config['training'].get('warmup_steps', 1000):
                agent.train()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        recent_rewards.append(total_reward)
        
        # 진행 상황 출력
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    env.close()
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'algorithm': 'DDPG'
    }


def evaluate_agent(agent, env_name: str, episodes: int = 10, render: bool = False) -> dict:
    """에이전트 평가"""
    if hasattr(agent, 'select_action'):  # DQN인지 확인
        env = create_dqn_env(env_name)
        is_dqn = True
    else:
        env = create_ddpg_env(env_name)
        is_dqn = False
    
    rewards = []
    lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if is_dqn:
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state, add_noise=False)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        lengths.append(steps)
    
    env.close()
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'rewards': rewards
    }


def create_video(agent, env_name: str, filename: str, episodes: int = 3):
    """에이전트 행동 비디오 생성"""
    print(f"비디오 생성 중: {filename}")
    
    # 환경 설정
    if hasattr(agent, 'select_action'):  # DQN
        base_env = create_dqn_env(env_name)
        is_dqn = True
    else:  # DDPG
        base_env = create_ddpg_env(env_name)
        is_dqn = False
    
    # 비디오 녹화 환경 생성
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder=os.path.dirname(filename),
        name_prefix=os.path.basename(filename).replace('.mp4', ''),
        episode_trigger=lambda x: True  # 모든 에피소드 녹화
    )
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        while True:
            if is_dqn:
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state, add_noise=False)
            
            state, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
    
    env.close()
    print(f"비디오 저장 완료: {filename}")


def plot_results(dqn_results: dict, ddpg_results: dict, save_path: str = 'results/training_comparison.png'):
    """학습 결과 시각화"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # DQN 학습 곡선
    ax1.plot(dqn_results['episode_rewards'], alpha=0.6, label='Episode Rewards')
    
    # 이동 평균 계산
    window = 20
    if len(dqn_results['episode_rewards']) >= window:
        moving_avg = np.convolve(dqn_results['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(dqn_results['episode_rewards'])), 
                moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax1.set_title('DQN Learning Curve')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # DDPG 학습 곡선
    ax2.plot(ddpg_results['episode_rewards'], alpha=0.6, label='Episode Rewards')
    
    if len(ddpg_results['episode_rewards']) >= window:
        moving_avg = np.convolve(ddpg_results['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(ddpg_results['episode_rewards'])), 
                moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax2.set_title('DDPG Learning Curve')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 에피소드 길이 비교
    ax3.plot(dqn_results['episode_lengths'], alpha=0.6, label='DQN')
    ax3.plot(ddpg_results['episode_lengths'], alpha=0.6, label='DDPG')
    ax3.set_title('Episode Lengths Comparison')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 최종 성능 박스플롯
    recent_dqn = dqn_results['episode_rewards'][-20:] if len(dqn_results['episode_rewards']) >= 20 else dqn_results['episode_rewards']
    recent_ddpg = ddpg_results['episode_rewards'][-20:] if len(ddpg_results['episode_rewards']) >= 20 else ddpg_results['episode_rewards']
    
    ax4.boxplot([recent_dqn, recent_ddpg], labels=['DQN', 'DDPG'])
    ax4.set_title('Final Performance (Last 20 Episodes)')
    ax4.set_ylabel('Reward')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"학습 결과 시각화 저장: {save_path}")
    plt.show()


def main():
    """메인 실행 함수"""
    # 시드 설정
    set_seed(42)
    
    print("🚀 DQN vs DDPG 전체 실험 시작")
    print(f"사용 디바이스: {get_device()}")
    
    # 설정 로드
    dqn_config = load_config('configs/dqn_config.yaml')
    ddpg_config = load_config('configs/ddpg_config.yaml')
    
    # 결과 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    
    # DQN 훈련
    print("\n🎯 1단계: DQN 훈련")
    dqn_agent, dqn_results = train_dqn(dqn_config, episodes=200)
    
    # DDPG 훈련
    print("\n🎯 2단계: DDPG 훈련")
    ddpg_agent, ddpg_results = train_ddpg(ddpg_config, episodes=200)
    
    # 평가
    print("\n🎯 3단계: 에이전트 평가")
    dqn_eval = evaluate_agent(dqn_agent, dqn_config['environment']['name'])
    ddpg_eval = evaluate_agent(ddpg_agent, ddpg_config['environment']['name'])
    
    print(f"DQN 평가 결과: {dqn_eval['mean_reward']:.2f} ± {dqn_eval['std_reward']:.2f}")
    print(f"DDPG 평가 결과: {ddpg_eval['mean_reward']:.2f} ± {ddpg_eval['std_reward']:.2f}")
    
    # 시각화
    print("\n🎯 4단계: 결과 시각화")
    plot_results(dqn_results, ddpg_results)
    
    # 비디오 생성
    print("\n🎯 5단계: 비디오 생성")
    try:
        create_video(dqn_agent, dqn_config['environment']['name'], 'videos/dqn_demo.mp4')
        create_video(ddpg_agent, ddpg_config['environment']['name'], 'videos/ddpg_demo.mp4')
    except Exception as e:
        print(f"비디오 생성 중 오류: {e}")
        print("비디오 생성을 건너뜁니다.")
    
    print("\n✅ 전체 실험 완료!")
    print("결과를 확인하세요:")
    print("- 학습 곡선: results/training_comparison.png")
    print("- 비디오: videos/dqn_demo.mp4, videos/ddpg_demo.mp4")


if __name__ == "__main__":
    main()