#!/usr/bin/env python3
"""
DDPG ContinuousCartPole 수동 학습 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time

from src.core.utils import get_device, set_seed
from src.core.buffer import ReplayBuffer
from src.core.noise import GaussianNoise
from src.networks.actor import Actor
from src.networks.critic import Critic
from src.environments.continuous_cartpole import ContinuousCartPole


class SpecialDDPGAgent:
    """ContinuousCartPole 수렴을 위한 특별 조정된 DDPG"""
    
    def __init__(self, state_dim: int, action_dim: int, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # ContinuousCartPole에 최적화된 하이퍼파라미터
        self.actor_lr = 0.000001   # 더욱 낮은 학습률
        self.critic_lr = 0.00001
        self.gamma = 0.99
        self.tau = 0.0001  # 매우 느린 타겟 업데이트
        self.batch_size = 256
        
        # 네트워크 생성 (작은 네트워크)
        self.actor = Actor(state_dim, action_dim, hidden_dims=[32, 32]).to(device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dims=[32, 32]).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dims=[32, 32]).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dims=[32, 32]).to(device)
        
        # 타겟 네트워크 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 리플레이 버퍼
        self.buffer = ReplayBuffer(100000, device=device)
        
        # 노이즈 (더 작게)
        self.noise = GaussianNoise(action_dim, sigma=0.01, decay_rate=0.9998)
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """액션 선택"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
            
        if add_noise:
            noise = self.noise.sample()
            action += noise
            
        return np.clip(action, -1.0, 1.0)
        
    def update(self) -> dict:
        """네트워크 업데이트"""
        if len(self.buffer) < self.batch_size:
            return {}
            
        # 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic 업데이트
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)  # 더 강한 클리핑
        self.critic_optimizer.step()
        
        # Actor 업데이트
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
        
    def soft_update(self, target, source):
        """소프트 업데이트"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


def train_ddpg_continuous():
    """DDPG ContinuousCartPole 수렴 학습"""
    
    print("🚀 DDPG ContinuousCartPole 수렴 학습 시작")
    
    # 설정
    device = get_device()
    set_seed(42)
    
    # 환경 생성
    env = ContinuousCartPole()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"🔧 환경: state_dim={state_dim}, action_dim={action_dim}")
    print(f"🔧 디바이스: {device}")
    
    # 에이전트 생성
    agent = SpecialDDPGAgent(state_dim, action_dim, device)
    
    # 결과 저장
    results_dir = Path("results/ddpg_continuous")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("models/ddpg_continuous")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 메트릭
    episode_rewards = []
    episode_lengths = []
    
    # 학습 파라미터
    max_episodes = 3000
    max_steps = 500
    warmup_episodes = 200  # 더 긴 워밍업
    
    print(f"📊 설정: max_episodes={max_episodes}, warmup={warmup_episodes}")
    
    start_time = time.time()
    
    # 워밍업
    print("🔥 워밍업 시작...")
    for ep in range(warmup_episodes):
        state, _ = env.reset()
        
        for step in range(max_steps):
            action = np.random.uniform(-1, 1, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 보상 스케일링
            scaled_reward = reward * 0.001  # 매우 작은 보상
            
            agent.buffer.push(state, action, scaled_reward, next_state, done)
            
            state = next_state
            if done:
                break
                
        if ep % 50 == 0:
            print(f"   워밍업 {ep}/{warmup_episodes}")
            
    print("✅ 워밍업 완료")
    
    # 메인 학습
    print("🎯 메인 학습 시작...")
    best_avg_reward = -float('inf')
    no_improvement_count = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 액션 선택
            if episode < 100:  # 더 긴 초기 탐험
                action = np.random.uniform(-1, 1, action_dim)
            else:
                action = agent.select_action(state, add_noise=True)
                
            # 환경 스텝
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 보상 재구성 (ContinuousCartPole용)
            if done and step < 100:
                balance_reward = -2.0  # 일찍 끝나면 큰 페널티
            elif step >= 400:
                balance_reward = 3.0   # 오래 버티면 큰 보너스
            else:
                balance_reward = 1.0   # 기본 보상
                
            scaled_reward = balance_reward * 0.001
            
            # 버퍼에 저장
            agent.buffer.push(state, action, scaled_reward, next_state, done)
            
            # 학습
            if len(agent.buffer) >= agent.batch_size:
                agent.update()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 노이즈 감소
        agent.noise.decay()
        
        # 진행상황 출력
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            elapsed = time.time() - start_time
            
            print(f"📊 Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward:6.1f} | "
                  f"Length: {episode_length:3d} | "
                  f"AvgLen: {avg_length:5.1f} | "
                  f"Noise: {agent.noise.sigma:.6f} | "
                  f"Time: {elapsed/60:.1f}m")
            
            # 개선 확인
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improvement_count = 0
                
                # 최고 성능 모델 저장
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'episode': episode,
                    'avg_reward': avg_reward
                }, models_dir / 'best_model.pth')
                
            else:
                no_improvement_count += 1
        
        # 모델 저장 (주기적)
        if episode % 300 == 0:
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode,
                'avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            }, models_dir / f'checkpoint_episode_{episode}.pth')
        
        # 수렴 확인
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward >= 350.0:  # ContinuousCartPole 수렴
                print(f"🎉 수렴! Episode {episode}, 평균 보상: {avg_reward:.1f}")
                break
                
        # 조기 종료
        if no_improvement_count >= 500:
            print(f"⚠️ {no_improvement_count} 에피소드 동안 개선 없음. 조기 종료.")
            break
    
    # 최종 결과
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    
    print(f"\n✅ DDPG ContinuousCartPole 학습 완료")
    print(f"📊 총 에피소드: {len(episode_rewards)}")
    print(f"📊 최종 평균 보상: {final_avg_reward:.1f}")
    print(f"📊 최고 평균 보상: {best_avg_reward:.1f}")
    print(f"⏰ 총 시간: {total_time/60:.1f}분")
    print(f"🎯 수렴 여부: {final_avg_reward >= 350.0}")
    
    # 결과 저장
    results = {
        'algorithm': 'ddpg_continuous',
        'environment': 'ContinuousCartPole',
        'total_episodes': len(episode_rewards),
        'final_avg_reward': final_avg_reward,
        'best_avg_reward': best_avg_reward,
        'max_reward': max(episode_rewards),
        'training_time': total_time,
        'converged': final_avg_reward >= 350.0,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"ddpg_continuous_{timestamp}.json"
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 결과 저장: {result_path}")
    
    return results


if __name__ == "__main__":
    results = train_ddpg_continuous()
    
    # 결과 출력
    print("\n" + "="*60)
    print("📈 최종 결과")
    print("="*60)
    print(f"수렴 여부: {results['converged']}")
    print(f"최종 평균 보상: {results['final_avg_reward']:.1f}")
    print(f"최고 평균 보상: {results['best_avg_reward']:.1f}")
    print(f"총 에피소드: {results['total_episodes']}")
    print(f"학습 시간: {results['training_time']/60:.1f}분")