#!/usr/bin/env python3
"""
종합 환경별 학습 스크립트 - 모든 조합 지원

지원하는 조합:
1. CartPole-v1 + DQN (자연스러운 조합)
2. CartPole-v1 + DDPG (DiscreteDDPG 사용)
3. Pendulum-v1 + DQN (DiscretizedDQN 사용)  
4. Pendulum-v1 + DDPG (자연스러운 조합)

각 조합에 대해 2000 에피소드 학습 후 성능 비교
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


class CrossEnvironmentTrainer:
    """교차 환경 학습 매니저"""
    
    def __init__(self, device: str = None, episodes: int = 2000):
        self.device = device or get_device()
        self.episodes = episodes
        self.config_manager = ConfigManager()
        self.env_factory = SimpleEnvironmentFactory(self.config_manager)
        
        # 결과 저장 디렉토리
        self.results_dir = Path("results/cross_environment")
        self.models_dir = Path("models/cross_environment")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 교차 환경 학습 초기화")
        print(f"   디바이스: {self.device}")
        print(f"   에피소드: {self.episodes}")
    
    def create_agent(self, algorithm: str, environment: str, env) -> Any:
        """환경과 알고리즘에 맞는 에이전트 생성"""
        
        state_dim = env.observation_space.shape[0]
        
        if environment == "CartPole-v1":
            # CartPole: 이산 행동 공간 (2개 행동)
            if algorithm == "DQN":
                return DQNAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.n,
                    learning_rate=0.001,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=0.995,
                    buffer_size=100000,
                    batch_size=64,
                    target_update_freq=100,
                    device=self.device
                )
            elif algorithm == "DDPG":
                return DiscreteDDPGAgent(
                    state_dim=state_dim,
                    num_actions=env.action_space.n,
                    actor_lr=0.0001,  # learning_rate -> actor_lr
                    critic_lr=0.0001,
                    gamma=0.99,
                    tau=0.001,
                    buffer_size=100000,
                    batch_size=128,
                    device=self.device,
                    noise_sigma=0.1,
                    noise_decay=0.999
                )
                
        elif environment == "Pendulum-v1":
            # Pendulum: 연속 행동 공간
            if algorithm == "DQN":
                return DiscretizedDQNAgent(
                    state_dim=state_dim,
                    action_bound=2.0,  # Pendulum 행동 범위
                    num_actions=21,    # 이산화 레벨
                    learning_rate=0.001,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=0.995,
                    buffer_size=100000,
                    batch_size=64,
                    target_update_freq=100,
                    device=self.device
                )
            elif algorithm == "DDPG":
                return DDPGAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.shape[0],
                    actor_lr=0.0001,  # learning_rate -> actor_lr
                    critic_lr=0.0001,
                    gamma=0.99,
                    tau=0.001,
                    buffer_size=100000,
                    batch_size=128,
                    device=self.device,
                    noise_sigma=0.2,
                    noise_decay=0.999
                )
        
        raise ValueError(f"지원하지 않는 조합: {algorithm} + {environment}")
    
    def train_combination(self, algorithm: str, environment: str) -> Dict[str, Any]:
        """특정 알고리즘-환경 조합 학습"""
        
        print(f"\n🚀 {algorithm} + {environment} 학습 시작")
        
        # 환경 생성
        env = self.env_factory.create_env(
            env_name=environment,
            agent_type='auto',  # 자동 감지
            training=True,
            seed=42
        )
        
        # 에이전트 생성
        agent = self.create_agent(algorithm, environment, env)
        
        # 학습 메트릭
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        # 학습 시작
        start_time = time.time()
        best_avg_reward = -float('inf')
        
        for episode in range(self.episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            max_steps = 500 if environment == "CartPole-v1" else 200
            
            for step in range(max_steps):
                # 행동 선택 (인터페이스 통일)
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state)
                else:
                    action = agent.act(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장 (DiscretizedDQN은 특별한 처리 필요)
                if hasattr(agent, 'store_transition'):
                    # DiscretizedDQN의 경우 store_transition 사용 (자동으로 연속->이산 변환)
                    agent.store_transition(state, action, reward, next_state, done)
                else:
                    # 다른 에이전트는 직접 버퍼에 저장
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
            
            # 진행상황 출력
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                elapsed = time.time() - start_time
                
                print(f"📊 Episode {episode:4d} | "\n                      f"Reward: {episode_reward:7.1f} | "\n                      f"Avg(100): {avg_reward:7.1f} | "\n                      f"Length: {episode_length:3d} | "\n                      f"AvgLen: {avg_length:5.1f} | "\n                      f"Time: {elapsed/60:.1f}m")\n                \n                # 최고 성능 체크\n                if avg_reward > best_avg_reward:\n                    best_avg_reward = avg_reward\n                    \n                    # 최고 모델 저장\n                    model_path = self.models_dir / f"{algorithm}_{environment}_best.pth"\n                    agent.save_checkpoint(str(model_path), episode, avg_reward)\n            \n            # 주기적 체크포인트\n            if episode % 500 == 0 and episode > 0:\n                model_path = self.models_dir / f"{algorithm}_{environment}_episode_{episode}.pth"\n                agent.save_checkpoint(str(model_path), episode, episode_reward)\n        \n        # 학습 완료\n        total_time = time.time() - start_time\n        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)\n        \n        print(f"\\n✅ {algorithm} + {environment} 학습 완료")\n        print(f"📊 최종 평균 보상: {final_avg_reward:.1f}")\n        print(f"📊 최고 평균 보상: {best_avg_reward:.1f}")\n        print(f"⏰ 총 시간: {total_time/60:.1f}분")\n        \n        # 수렴 판정\n        converged = False\n        if environment == "CartPole-v1":\n            converged = final_avg_reward >= 475.0  # CartPole 수렴 기준\n        elif environment == "Pendulum-v1":\n            converged = final_avg_reward >= -200.0  # Pendulum 수렴 기준 (높을수록 좋음)\n        \n        env.close()\n        \n        return {\n            'algorithm': algorithm,\n            'environment': environment,\n            'total_episodes': self.episodes,\n            'final_avg_reward': final_avg_reward,\n            'best_avg_reward': best_avg_reward,\n            'max_reward': max(episode_rewards),\n            'min_reward': min(episode_rewards),\n            'training_time': total_time,\n            'converged': converged,\n            'episode_rewards': episode_rewards,\n            'episode_lengths': episode_lengths,\n            'losses': losses if losses else []\n        }\n    \n    def run_all_combinations(self) -> Dict[str, Any]:\n        """모든 알고리즘-환경 조합 실행"""\n        \n        combinations = [\n            ("DQN", "CartPole-v1"),     # 자연스러운 조합\n            ("DDPG", "CartPole-v1"),    # 적응된 조합\n            ("DQN", "Pendulum-v1"),     # 적응된 조합  \n            ("DDPG", "Pendulum-v1")     # 자연스러운 조합\n        ]\n        \n        all_results = {}\n        summary = {\n            'timestamp': datetime.now().isoformat(),\n            'total_combinations': len(combinations),\n            'episodes_per_combination': self.episodes,\n            'device': str(self.device),\n            'results': {}\n        }\n        \n        print(f"🎯 {len(combinations)}개 조합 학습 시작 (총 {len(combinations) * self.episodes} 에피소드)\\n")\n        \n        total_start_time = time.time()\n        \n        for i, (algorithm, environment) in enumerate(combinations, 1):\n            print(f"{'='*60}")\n            print(f"진행: {i}/{len(combinations)} - {algorithm} + {environment}")\n            print(f"{'='*60}")\n            \n            # 개별 조합 학습\n            result = self.train_combination(algorithm, environment)\n            \n            # 결과 저장\n            combination_key = f"{algorithm}_{environment}"\n            all_results[combination_key] = result\n            summary['results'][combination_key] = {\n                'final_avg_reward': result['final_avg_reward'],\n                'best_avg_reward': result['best_avg_reward'],\n                'converged': result['converged'],\n                'training_time': result['training_time']\n            }\n            \n            # 개별 결과 파일 저장\n            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n            result_file = self.results_dir / f"{combination_key}_{timestamp}.json"\n            with open(result_file, 'w') as f:\n                json.dump(result, f, indent=2)\n            \n            print(f"💾 결과 저장: {result_file}")\n        \n        # 종합 결과 저장\n        total_time = time.time() - total_start_time\n        summary['total_training_time'] = total_time\n        \n        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n        summary_file = self.results_dir / f"cross_environment_summary_{timestamp}.json"\n        with open(summary_file, 'w') as f:\n            json.dump(summary, f, indent=2)\n        \n        # 비교 분석 출력\n        self.print_comparison_analysis(summary)\n        \n        return summary\n    \n    def print_comparison_analysis(self, summary: Dict[str, Any]):\n        """비교 분석 결과 출력"""\n        \n        print(f"\\n\\n{'='*80}")\n        print("🎉 교차 환경 학습 완료 - 비교 분석")\n        print(f"{'='*80}\\n")\n        \n        results = summary['results']\n        \n        print("📊 환경별 성능 비교\\n")\n        \n        # CartPole 비교\n        print("🎮 CartPole-v1 환경:")\n        dqn_cartpole = results.get('DQN_CartPole-v1', {})\n        ddpg_cartpole = results.get('DDPG_CartPole-v1', {})\n        \n        print(f"   DQN (자연):  {dqn_cartpole.get('final_avg_reward', 0):7.1f} | "\n              f"수렴: {dqn_cartpole.get('converged', False)} | "\n              f"시간: {dqn_cartpole.get('training_time', 0)/60:.1f}분")\n        print(f"   DDPG (적응): {ddpg_cartpole.get('final_avg_reward', 0):7.1f} | "\n              f"수렴: {ddpg_cartpole.get('converged', False)} | "\n              f"시간: {ddpg_cartpole.get('training_time', 0)/60:.1f}분")\n        \n        # Pendulum 비교\n        print("\\n🎯 Pendulum-v1 환경:")\n        dqn_pendulum = results.get('DQN_Pendulum-v1', {})\n        ddpg_pendulum = results.get('DDPG_Pendulum-v1', {})\n        \n        print(f"   DQN (적응):  {dqn_pendulum.get('final_avg_reward', 0):7.1f} | "\n              f"수렴: {dqn_pendulum.get('converged', False)} | "\n              f"시간: {dqn_pendulum.get('training_time', 0)/60:.1f}분")\n        print(f"   DDPG (자연): {ddpg_pendulum.get('final_avg_reward', 0):7.1f} | "\n              f"수렴: {ddpg_pendulum.get('converged', False)} | "\n              f"시간: {ddpg_pendulum.get('training_time', 0)/60:.1f}분")\n        \n        print(f"\\n⏰ 총 학습 시간: {summary['total_training_time']/60:.1f}분")\n        print(f"💾 결과 저장 위치: {self.results_dir}/")\n        print(f"🎯 모델 저장 위치: {self.models_dir}/")\n        \n        # 주요 인사이트\n        print("\\n🔍 주요 인사이트:")\n        \n        # CartPole에서 어느 것이 더 좋은가?\n        if dqn_cartpole and ddpg_cartpole:\n            dqn_score = dqn_cartpole.get('final_avg_reward', 0)\n            ddpg_score = ddpg_cartpole.get('final_avg_reward', 0)\n            if dqn_score > ddpg_score:\n                print(f"   • CartPole에서 DQN이 DDPG보다 {dqn_score - ddpg_score:.1f}점 더 좋음")\n            else:\n                print(f"   • CartPole에서 DDPG가 DQN보다 {ddpg_score - dqn_score:.1f}점 더 좋음")\n        \n        # Pendulum에서 어느 것이 더 좋은가?\n        if dqn_pendulum and ddpg_pendulum:\n            dqn_score = dqn_pendulum.get('final_avg_reward', 0)\n            ddpg_score = ddpg_pendulum.get('final_avg_reward', 0)\n            if ddpg_score > dqn_score:\n                print(f"   • Pendulum에서 DDPG가 DQN보다 {ddpg_score - dqn_score:.1f}점 더 좋음")\n            else:\n                print(f"   • Pendulum에서 DQN이 DDPG보다 {dqn_score - ddpg_score:.1f}점 더 좋음")\n        \n        print(f"\\n{'='*80}\\n")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='교차 환경 학습 스크립트')\n    parser.add_argument('--episodes', type=int, default=2000, help='에피소드 수 (기본: 2000)')\n    parser.add_argument('--device', type=str, default=None, help='디바이스 (cuda/cpu, 기본: 자동)')\n    parser.add_argument('--algorithm', type=str, default=None, \n                       choices=['DQN', 'DDPG'], help='특정 알고리즘만 실행')\n    parser.add_argument('--environment', type=str, default=None,\n                       choices=['CartPole-v1', 'Pendulum-v1'], help='특정 환경만 실행')\n    \n    args = parser.parse_args()\n    \n    # 시드 설정\n    set_seed(42)\n    \n    # 트레이너 초기화\n    trainer = CrossEnvironmentTrainer(device=args.device, episodes=args.episodes)\n    \n    if args.algorithm and args.environment:\n        # 특정 조합만 실행\n        print(f"🎯 단일 조합 실행: {args.algorithm} + {args.environment}")\n        result = trainer.train_combination(args.algorithm, args.environment)\n        \n        # 결과 출력\n        print(f"\\n✅ 결과:")\n        print(f"   최종 평균 보상: {result['final_avg_reward']:.1f}")\n        print(f"   수렴 여부: {result['converged']}")\n        print(f"   학습 시간: {result['training_time']/60:.1f}분")\n        \n    else:\n        # 모든 조합 실행\n        print("🎯 모든 조합 실행")\n        summary = trainer.run_all_combinations()\n\n\nif __name__ == "__main__":\n    main()