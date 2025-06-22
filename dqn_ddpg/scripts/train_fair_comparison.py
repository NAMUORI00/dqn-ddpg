#!/usr/bin/env python3
"""
공정한 DQN vs DDPG 비교를 위한 통일된 하이퍼파라미터 학습 스크립트

모든 알고리즘에 동일한 하이퍼파라미터 적용:
- Learning Rate: 0.0001
- Batch Size: 128  
- Buffer Size: 100,000
- Gamma: 0.99
- Decay Rate: 0.9995 (epsilon/noise)

알고리즘별 필수 차이점만 유지:
- DQN: Target network hard update (1000 steps)
- DDPG: Target network soft update (tau=0.005)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time
import argparse
from typing import Dict, List, Any
import gymnasium as gym

from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.discrete_ddpg_agent import DiscreteDDPGAgent
from src.environments.env_factory import SimpleEnvironmentFactory
from src.core.config_manager import ConfigManager
from src.core.utils import get_device, set_seed


class FairComparisonTrainer:
    """공정한 비교를 위한 통일된 하이퍼파라미터 학습 매니저"""
    
    # 통일된 하이퍼파라미터
    UNIFIED_PARAMS = {
        'learning_rate': 0.0001,      # 모든 알고리즘 동일
        'batch_size': 128,            # 모든 알고리즘 동일
        'buffer_size': 100000,        # 모든 알고리즘 동일
        'gamma': 0.99,                # 모든 알고리즘 동일
        'decay_rate': 0.9995,         # epsilon/noise decay 동일
        'episodes': 2000              # 학습 에피소드 동일
    }
    
    def __init__(self, device: str = None):
        self.device = device or get_device()
        self.config_manager = ConfigManager()
        self.env_factory = SimpleEnvironmentFactory(self.config_manager)
        
        # 세션별 타임스탬프 생성
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 저장 디렉토리 (공정 비교용)
        self.session_dir = Path(f"results/fair_comparison/{self.session_timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 공정한 DQN vs DDPG 비교 학습 초기화")
        print(f"   디바이스: {self.device}")
        print(f"   에피소드: {self.UNIFIED_PARAMS['episodes']}")
        print(f"   세션: {self.session_timestamp}")
        print(f"   📁 결과 디렉토리: {self.session_dir}")
        print()
        print("📊 통일된 하이퍼파라미터:")
        for key, value in self.UNIFIED_PARAMS.items():
            print(f"   {key}: {value}")
    
    def create_agent(self, algorithm: str, environment: str, env) -> Any:
        """통일된 하이퍼파라미터로 에이전트 생성"""
        
        state_dim = env.observation_space.shape[0]
        
        # 공통 파라미터
        common_params = {
            'gamma': self.UNIFIED_PARAMS['gamma'],
            'buffer_size': self.UNIFIED_PARAMS['buffer_size'],
            'batch_size': self.UNIFIED_PARAMS['batch_size'],
            'device': self.device
        }
        
        if environment == "CartPole-v1":
            # CartPole: 이산 행동 공간
            if algorithm == "DQN":
                return DQNAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.n,
                    learning_rate=self.UNIFIED_PARAMS['learning_rate'],
                    epsilon=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=self.UNIFIED_PARAMS['decay_rate'],
                    target_update_freq=1000,  # 중간값으로 통일
                    **common_params
                )
            elif algorithm == "DDPG":
                return DiscreteDDPGAgent(
                    state_dim=state_dim,
                    num_actions=env.action_space.n,
                    actor_lr=self.UNIFIED_PARAMS['learning_rate'],
                    critic_lr=self.UNIFIED_PARAMS['learning_rate'],
                    tau=0.005,  # soft update
                    noise_sigma=0.1,
                    noise_decay=self.UNIFIED_PARAMS['decay_rate'],
                    **common_params
                )
                
        elif environment == "Pendulum-v1":
            # Pendulum: 연속 행동 공간
            if algorithm == "DQN":
                return DiscretizedDQNAgent(
                    state_dim=state_dim,
                    action_bound=2.0,
                    num_actions=21,
                    learning_rate=self.UNIFIED_PARAMS['learning_rate'],
                    epsilon=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=self.UNIFIED_PARAMS['decay_rate'],
                    target_update_freq=1000,
                    **common_params
                )
            elif algorithm == "DDPG":
                return DDPGAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.shape[0],
                    actor_lr=self.UNIFIED_PARAMS['learning_rate'],
                    critic_lr=self.UNIFIED_PARAMS['learning_rate'],
                    tau=0.005,
                    noise_sigma=0.2,
                    noise_decay=self.UNIFIED_PARAMS['decay_rate'],
                    **common_params
                )
        
        raise ValueError(f"지원하지 않는 조합: {algorithm} + {environment}")
    
    def train_combination(self, algorithm: str, environment: str) -> Dict[str, Any]:
        """특정 알고리즘-환경 조합 학습 (통일된 하이퍼파라미터)"""
        
        print(f"\n🚀 {algorithm} + {environment} 학습 시작 (공정 비교)")
        
        # 조합별 디렉토리 생성
        combination_name = f"{algorithm}_{environment}"
        combination_dir = self.session_dir / combination_name
        checkpoints_dir = combination_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   📁 조합 디렉토리: {combination_dir}")
        print(f"   💾 체크포인트: {checkpoints_dir}")
        
        # 환경 생성
        env = self.env_factory.create_env(
            env_name=environment,
            agent_type='auto',
            training=True,
            seed=42
        )
        
        # 에이전트 생성 (통일된 하이퍼파라미터)
        agent = self.create_agent(algorithm, environment, env)
        
        # 학습 메트릭
        episode_rewards = []
        episode_lengths = []
        losses = []
        checkpoint_episodes = []
        
        # 학습 시작
        start_time = time.time()
        best_avg_reward = -float('inf')
        
        for episode in range(self.UNIFIED_PARAMS['episodes']):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            max_steps = 500 if environment == "CartPole-v1" else 200
            
            for step in range(max_steps):
                # 행동 선택
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state)
                else:
                    action = agent.act(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                if hasattr(agent, 'store_transition'):
                    agent.store_transition(state, action, reward, next_state, done)
                else:
                    agent.buffer.push(state, action, reward, next_state, done)
                
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
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_losses:
                losses.append(np.mean([loss.get('loss', 0) for loss in episode_losses]))
            
            # 10 에피소드마다 체크포인트 저장
            if episode % 10 == 0:
                model_path = checkpoints_dir / f"episode_{episode:04d}.pth"
                agent.save(str(model_path))
                checkpoint_episodes.append(episode)
                
                # 메타데이터 저장
                metadata = {
                    'episode': episode,
                    'reward': episode_reward,
                    'timestamp': datetime.now().isoformat(),
                    'avg_reward_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                }
                metadata_path = checkpoints_dir / f"episode_{episode:04d}_meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # 진행상황 출력
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                elapsed = time.time() - start_time
                progress = (episode + 1) / self.UNIFIED_PARAMS['episodes'] * 100
                
                # 현재 탐험률/노이즈 확인
                if hasattr(agent, 'epsilon'):
                    explore_param = f"ε: {agent.epsilon:.4f}"
                elif hasattr(agent, 'noise'):
                    explore_param = f"σ: {agent.noise.sigma:.4f}"
                else:
                    explore_param = "N/A"
                
                print(f"📊 Episode {episode:4d} ({progress:5.1f}%) | "
                      f"Reward: {episode_reward:7.1f} | "
                      f"Avg(100): {avg_reward:7.1f} | "
                      f"Length: {episode_length:3d} | "
                      f"{explore_param} | "
                      f"Time: {elapsed/60:.1f}m")
                
                # 최고 성능 체크
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model_path = combination_dir / "best_model.pth"
                    agent.save(str(best_model_path))
        
        # 학습 완료
        total_time = time.time() - start_time
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        print(f"\n✅ {algorithm} + {environment} 학습 완료 (공정 비교)")
        print(f"📊 최종 평균 보상: {final_avg_reward:.1f}")
        print(f"📊 최고 평균 보상: {best_avg_reward:.1f}")
        print(f"⏰ 총 시간: {total_time/60:.1f}분")
        print(f"💾 체크포인트 개수: {len(checkpoint_episodes)}개")
        
        # 수렴 판정
        converged = False
        if environment == "CartPole-v1":
            converged = final_avg_reward >= 475.0
        elif environment == "Pendulum-v1":
            converged = final_avg_reward >= -200.0
        
        env.close()
        
        result = {
            'algorithm': algorithm,
            'environment': environment,
            'total_episodes': self.UNIFIED_PARAMS['episodes'],
            'final_avg_reward': float(final_avg_reward),
            'best_avg_reward': float(best_avg_reward),
            'max_reward': float(max(episode_rewards)),
            'min_reward': float(min(episode_rewards)),
            'training_time': float(total_time),
            'converged': bool(converged),
            'hyperparameters': self.UNIFIED_PARAMS,
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths],
            'losses': [float(l) for l in losses] if losses else [],
            'checkpoint_episodes': [int(e) for e in checkpoint_episodes],
            'num_checkpoints': len(checkpoint_episodes),
            'combination_dir': str(combination_dir),
            'checkpoints_dir': str(checkpoints_dir)
        }
        
        # 개별 결과 파일 저장
        result_file = combination_dir / "training_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 결과 저장: {result_file}")
        
        return result
    
    def run_all_combinations(self) -> Dict[str, Any]:
        """모든 알고리즘-환경 조합 실행 (공정 비교)"""
        
        combinations = [
            ("DQN", "CartPole-v1"),
            ("DDPG", "CartPole-v1"),
            ("DQN", "Pendulum-v1"),
            ("DDPG", "Pendulum-v1")
        ]
        
        all_results = {}
        summary = {
            'session_timestamp': self.session_timestamp,
            'session_dir': str(self.session_dir),
            'timestamp': datetime.now().isoformat(),
            'total_combinations': len(combinations),
            'episodes_per_combination': self.UNIFIED_PARAMS['episodes'],
            'unified_hyperparameters': self.UNIFIED_PARAMS,
            'device': str(self.device),
            'results': {}
        }
        
        print(f"🎯 공정 비교: {len(combinations)}개 조합 학습 시작")
        print(f"📊 통일된 하이퍼파라미터 적용")
        print()
        
        total_start_time = time.time()
        
        for i, (algorithm, environment) in enumerate(combinations, 1):
            print(f"{'='*60}")
            print(f"진행: {i}/{len(combinations)} - {algorithm} + {environment}")
            print(f"{'='*60}")
            
            # 개별 조합 학습
            result = self.train_combination(algorithm, environment)
            
            # 결과 저장
            combination_key = f"{algorithm}_{environment}"
            all_results[combination_key] = result
            summary['results'][combination_key] = {
                'final_avg_reward': result['final_avg_reward'],
                'best_avg_reward': result['best_avg_reward'],
                'converged': result['converged'],
                'training_time': result['training_time'],
                'num_checkpoints': result['num_checkpoints']
            }
        
        # 종합 결과 저장
        total_time = time.time() - total_start_time
        summary['total_training_time'] = total_time
        
        summary_file = self.session_dir / "fair_comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 공정 비교 분석 출력
        self.print_fair_comparison_analysis(summary)
        
        return summary
    
    def print_fair_comparison_analysis(self, summary: Dict[str, Any]):
        """공정 비교 분석 결과 출력"""
        
        print(f"\n\n{'='*80}")
        print("🎉 공정한 DQN vs DDPG 비교 완료")
        print(f"{'='*80}\n")
        
        results = summary['results']
        
        print("📊 통일된 하이퍼파라미터:")
        for key, value in self.UNIFIED_PARAMS.items():
            print(f"   {key}: {value}")
        
        print("\n📊 환경별 성능 비교 (동일 조건)\n")
        
        # CartPole 비교
        print("🎮 CartPole-v1 환경:")
        dqn_cartpole = results.get('DQN_CartPole-v1', {})
        ddpg_cartpole = results.get('DDPG_CartPole-v1', {})
        
        print(f"   DQN:  {dqn_cartpole.get('final_avg_reward', 0):7.1f} | "
              f"수렴: {dqn_cartpole.get('converged', False)} | "
              f"시간: {dqn_cartpole.get('training_time', 0)/60:.1f}분")
        print(f"   DDPG: {ddpg_cartpole.get('final_avg_reward', 0):7.1f} | "
              f"수렴: {ddpg_cartpole.get('converged', False)} | "
              f"시간: {ddpg_cartpole.get('training_time', 0)/60:.1f}분")
        
        # Pendulum 비교
        print("\n🎯 Pendulum-v1 환경:")
        dqn_pendulum = results.get('DQN_Pendulum-v1', {})
        ddpg_pendulum = results.get('DDPG_Pendulum-v1', {})
        
        print(f"   DQN:  {dqn_pendulum.get('final_avg_reward', 0):7.1f} | "
              f"수렴: {dqn_pendulum.get('converged', False)} | "
              f"시간: {dqn_pendulum.get('training_time', 0)/60:.1f}분")
        print(f"   DDPG: {ddpg_pendulum.get('final_avg_reward', 0):7.1f} | "
              f"수렴: {ddpg_pendulum.get('converged', False)} | "
              f"시간: {ddpg_pendulum.get('training_time', 0)/60:.1f}분")
        
        print(f"\n⏰ 총 학습 시간: {summary['total_training_time']/60:.1f}분")
        print(f"📁 세션 디렉토리: {summary['session_dir']}")
        
        # 주요 인사이트
        print("\n🔍 공정 비교 인사이트:")
        print("   • 모든 알고리즘이 동일한 하이퍼파라미터로 학습됨")
        print("   • 성능 차이는 알고리즘 자체의 특성을 반영")
        print("   • 환경별 적합성을 정확히 평가 가능")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='공정한 DQN vs DDPG 비교 학습')
    parser.add_argument('--device', type=str, default=None, help='디바이스 (cuda/cpu)')
    parser.add_argument('--algorithm', type=str, default=None, 
                       choices=['DQN', 'DDPG'], help='특정 알고리즘만 실행')
    parser.add_argument('--environment', type=str, default=None,
                       choices=['CartPole-v1', 'Pendulum-v1'], help='특정 환경만 실행')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(42)
    
    # 트레이너 초기화
    trainer = FairComparisonTrainer(device=args.device)
    
    if args.algorithm and args.environment:
        # 특정 조합만 실행
        print(f"🎯 단일 조합 공정 비교: {args.algorithm} + {args.environment}")
        result = trainer.train_combination(args.algorithm, args.environment)
        
        # 결과 출력
        print(f"\n✅ 공정 비교 결과:")
        print(f"   최종 평균 보상: {result['final_avg_reward']:.1f}")
        print(f"   수렴 여부: {result['converged']}")
        print(f"   학습 시간: {result['training_time']/60:.1f}분")
        
    else:
        # 모든 조합 실행
        print("🎯 모든 조합 공정 비교 실행")
        summary = trainer.run_all_combinations()


if __name__ == "__main__":
    main()