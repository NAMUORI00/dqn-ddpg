#!/usr/bin/env python3
"""
개선된 DQN으로 CartPole-v1 학습 스크립트

웹 자료 기반 최적화된 하이퍼파라미터:
- Learning Rate: 0.0001 (안정성)
- Target Update: 10,000 스텝 (안정성)
- Batch Size: 128 (품질)
- Epsilon Decay: 0.9995 (충분한 탐험)
- Double DQN 옵션
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import gymnasium as gym

from src.agents.improved_dqn_agent import ImprovedDQNAgent
from src.core.utils import get_device, set_seed


class ImprovedDQNTrainer:
    """개선된 DQN 학습 매니저"""
    
    def __init__(self, 
                 episodes: int = 2000,
                 use_double_dqn: bool = False,
                 device: str = None):
        self.episodes = episodes
        self.use_double_dqn = use_double_dqn
        self.device = device or get_device()
        
        # 세션 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = "DoubleDQN" if use_double_dqn else "ImprovedDQN"
        self.session_dir = Path(f"results/improved_dqn/{timestamp}_{method_name}_CartPole-v1")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.session_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        print(f"🚀 개선된 DQN 학습 초기화")
        print(f"   방법: {method_name}")
        print(f"   디바이스: {self.device}")
        print(f"   에피소드: {self.episodes}")
        print(f"   📁 결과 디렉토리: {self.session_dir}")
    
    def create_environment(self):
        """CartPole-v1 환경 생성"""
        env = gym.make("CartPole-v1")
        env.reset(seed=42)
        return env
    
    def create_agent(self, env):
        """개선된 DQN 에이전트 생성"""
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # 웹 자료 기반 최적화된 하이퍼파라미터
        agent = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.0001,        # 보수적 학습률
            gamma=0.99,
            epsilon=1.0,                 # 충분한 초기 탐험
            epsilon_min=0.01,
            epsilon_decay=0.9995,        # 느린 탐험 감소
            buffer_size=100000,          # 큰 리플레이 버퍼
            batch_size=128,              # 큰 배치 크기
            target_update_freq=10000,    # 드문 타겟 업데이트
            device=self.device,
            use_double_dqn=self.use_double_dqn
        )
        
        return agent
    
    def train(self):
        """개선된 DQN 학습 실행"""
        env = self.create_environment()
        agent = self.create_agent(env)
        
        # 학습 메트릭 추적
        episode_rewards = []
        episode_lengths = []
        losses = []
        convergence_history = []
        checkpoint_episodes = []
        
        start_time = time.time()
        best_avg_reward = -float('inf')
        
        print(f"\n🎯 개선된 DQN 학습 시작 (CartPole-v1)")
        print(f"   목표: 475+ 평균 보상으로 수렴")
        print(f"   체크포인트: 10 에피소드마다")
        print()
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            for step in range(500):  # CartPole-v1 max steps
                # 행동 선택
                action = agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                agent.store_transition(state, action, reward, next_state, done)
                
                # 학습
                if len(agent.buffer) >= agent.batch_size:
                    loss_dict = agent.update()
                    if loss_dict:
                        episode_losses.append(loss_dict)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 에피소드 완료 처리
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode_losses:
                avg_loss = np.mean([loss.get('loss', 0) for loss in episode_losses])
                losses.append(avg_loss)
            
            # 수렴 추적
            convergence_metrics = agent.track_episode_reward(episode_reward)
            convergence_history.append({
                'episode': int(episode),
                'reward': float(episode_reward),
                'avg_reward_100': float(convergence_metrics.get('avg_reward_100', 0)),
                'std_reward_100': float(convergence_metrics.get('std_reward_100', 0)),
                'converged': bool(convergence_metrics.get('converged', False)),
                'episodes_tracked': int(convergence_metrics.get('episodes_tracked', 0))
            })
            
            # 10 에피소드마다 체크포인트 저장
            if episode % 10 == 0:
                model_path = self.checkpoints_dir / f"episode_{episode:04d}.pth"
                agent.save(str(model_path))
                checkpoint_episodes.append(episode)
                
                # 메타데이터 저장
                metadata = {
                    'episode': int(episode),
                    'reward': float(episode_reward),
                    'timestamp': datetime.now().isoformat(),
                    'avg_reward_100': float(convergence_metrics.get('avg_reward_100', 0)),
                    'std_reward_100': float(convergence_metrics.get('std_reward_100', 0)),
                    'converged': bool(convergence_metrics.get('converged', False)),
                    'episodes_tracked': int(convergence_metrics.get('episodes_tracked', 0))
                }
                metadata_path = self.checkpoints_dir / f"episode_{episode:04d}_meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # 진행상황 출력
            if episode % 100 == 0:
                elapsed = time.time() - start_time
                progress = (episode + 1) / self.episodes * 100
                
                # 수렴 상태 확인
                convergence_status = agent.get_convergence_status()
                avg_reward = convergence_status['avg_reward']
                convergence_progress = convergence_status['progress'] * 100
                
                print(f"📊 Episode {episode:4d} ({progress:5.1f}%) | "
                      f"Reward: {episode_reward:7.1f} | "
                      f"Avg: {avg_reward:7.1f} | "
                      f"수렴: {convergence_progress:5.1f}% | "
                      f"ε: {agent.epsilon:.4f} | "
                      f"Time: {elapsed/60:.1f}m")
                
                # 최고 성능 모델 저장
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model_path = self.session_dir / "best_model.pth"
                    agent.save(str(best_model_path))
                
                # 수렴 달성 확인
                if convergence_status['converged']:
                    print(f"🎉 수렴 달성! 평균 보상: {avg_reward:.1f} >= 475.0")
        
        # 학습 완료
        total_time = time.time() - start_time
        final_convergence = agent.get_convergence_status()
        
        print(f"\n✅ 개선된 DQN 학습 완료")
        print(f"📊 최종 평균 보상: {final_convergence['avg_reward']:.1f}")
        print(f"📊 수렴 진행률: {final_convergence['progress']*100:.1f}%")
        print(f"📊 수렴 달성: {'✅' if final_convergence['converged'] else '❌'}")
        print(f"⏰ 총 시간: {total_time/60:.1f}분")
        print(f"💾 체크포인트 개수: {len(checkpoint_episodes)}개")
        
        env.close()
        
        # 결과 저장
        result = {
            'method': 'DoubleDQN' if self.use_double_dqn else 'ImprovedDQN',
            'algorithm': 'DQN',
            'environment': 'CartPole-v1',
            'total_episodes': self.episodes,
            'final_avg_reward': float(final_convergence['avg_reward']),
            'best_avg_reward': float(best_avg_reward),
            'converged': bool(final_convergence['converged']),
            'convergence_progress': float(final_convergence['progress']),
            'max_reward': float(max(episode_rewards)),
            'min_reward': float(min(episode_rewards)),
            'training_time': float(total_time),
            'hyperparameters': {
                'learning_rate': 0.0001,
                'batch_size': 128,
                'target_update_freq': 10000,
                'epsilon_decay': 0.9995,
                'use_double_dqn': self.use_double_dqn
            },
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths],
            'losses': [float(l) for l in losses] if losses else [],
            'convergence_history': convergence_history,
            'checkpoint_episodes': [int(e) for e in checkpoint_episodes],
            'num_checkpoints': len(checkpoint_episodes),
            'session_dir': str(self.session_dir),
            'checkpoints_dir': str(self.checkpoints_dir)
        }
        
        result_file = self.session_dir / "training_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 결과 저장: {result_file}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='개선된 DQN CartPole-v1 학습')
    parser.add_argument('--episodes', type=int, default=2000, help='에피소드 수 (기본: 2000)')
    parser.add_argument('--double-dqn', action='store_true', help='Double DQN 사용')
    parser.add_argument('--device', type=str, default=None, help='디바이스 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(42)
    
    # 트레이너 초기화 및 학습
    trainer = ImprovedDQNTrainer(
        episodes=args.episodes,
        use_double_dqn=args.double_dqn,
        device=args.device
    )
    
    result = trainer.train()
    
    # 결과 요약 출력
    method_name = "Double DQN" if args.double_dqn else "Improved DQN"
    print(f"\n🎯 {method_name} 최종 결과:")
    print(f"   최종 평균 보상: {result['final_avg_reward']:.1f}")
    print(f"   수렴 여부: {'✅' if result['converged'] else '❌'}")
    print(f"   학습 시간: {result['training_time']/60:.1f}분")
    print(f"   체크포인트: {result['num_checkpoints']}개")


if __name__ == "__main__":
    main()