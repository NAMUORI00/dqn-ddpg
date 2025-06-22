#!/usr/bin/env python3
"""
통합 학습 시스템
- 중복 코드 제거
- 모든 학습 시나리오 지원
- 개선된 결과 디렉토리 구조: results/{algorithm}/{environment}/{timestamp}/

지원하는 학습 모드:
1. Standard: 기본 하이퍼파라미터
2. Fair: 공정 비교를 위한 통일 하이퍼파라미터
3. Improved: 개선된 하이퍼파라미터 (DQN CartPole 수렴용)
4. Custom: 사용자 정의 하이퍼파라미터
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
from typing import Dict, List, Any, Optional, Union
import gymnasium as gym
from dataclasses import dataclass, asdict

from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.discrete_ddpg_agent import DiscreteDDPGAgent
from src.agents.improved_dqn_agent import ImprovedDQNAgent
from src.environments.env_factory import SimpleEnvironmentFactory
from src.core.config_manager import ConfigManager
from src.core.utils import get_device, set_seed


@dataclass
class TrainingConfig:
    """학습 설정 데이터클래스"""
    # 기본 설정
    algorithm: str
    environment: str
    episodes: int = 2000
    seed: int = 42
    device: Optional[str] = None
    
    # 하이퍼파라미터
    learning_rate: float = 0.001
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    
    # DQN 전용
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    use_double_dqn: bool = False
    
    # DDPG 전용
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    tau: float = 0.001
    noise_sigma: float = 0.1
    noise_decay: float = 0.999
    
    # 학습 모드
    mode: str = "standard"  # standard, fair, improved, custom
    
    def __post_init__(self):
        """파라미터 자동 설정"""
        # DDPG의 경우 actor/critic lr이 없으면 learning_rate 사용
        if self.algorithm == "DDPG":
            if self.actor_lr is None:
                self.actor_lr = self.learning_rate
            if self.critic_lr is None:
                self.critic_lr = self.learning_rate


class UnifiedTrainingSystem:
    """통합 학습 시스템"""
    
    # 모드별 하이퍼파라미터 프리셋
    PRESETS = {
        "standard": {
            "DQN": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "epsilon_decay": 0.995,
                "target_update_freq": 100
            },
            "DDPG": {
                "learning_rate": 0.0001,
                "batch_size": 128,
                "tau": 0.001,
                "noise_decay": 0.999
            }
        },
        "fair": {
            # 모든 알고리즘 동일
            "ALL": {
                "learning_rate": 0.0001,
                "batch_size": 128,
                "epsilon_decay": 0.9995,
                "noise_decay": 0.9995,
                "target_update_freq": 1000,
                "tau": 0.005
            }
        },
        "improved": {
            "DQN": {
                "learning_rate": 0.0001,
                "batch_size": 128,
                "epsilon_decay": 0.9995,
                "target_update_freq": 10000,
                "use_double_dqn": False
            }
        }
    }
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device or get_device()
        
        # 프리셋 적용
        self._apply_preset()
        
        # 환경 팩토리
        self.config_manager = ConfigManager()
        self.env_factory = SimpleEnvironmentFactory(self.config_manager)
        
        # 결과 디렉토리 구조: results/{algorithm}/{environment}/{timestamp}/
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / config.algorithm / config.environment / self.timestamp
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_config()
    
    def _apply_preset(self):
        """모드에 따른 프리셋 적용"""
        if self.config.mode in self.PRESETS:
            preset = self.PRESETS[self.config.mode]
            
            # 알고리즘별 또는 전체 프리셋 확인
            if self.config.algorithm in preset:
                params = preset[self.config.algorithm]
            elif "ALL" in preset:
                params = preset["ALL"]
            else:
                return
            
            # 프리셋 파라미터 적용
            for key, value in params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # DDPG lr 업데이트
            if self.config.algorithm == "DDPG":
                self.config.actor_lr = self.config.learning_rate
                self.config.critic_lr = self.config.learning_rate
    
    def _print_config(self):
        """설정 출력"""
        print(f"\n{'='*60}")
        print(f"🚀 통합 학습 시스템 초기화")
        print(f"{'='*60}")
        print(f"📋 기본 정보:")
        print(f"   알고리즘: {self.config.algorithm}")
        print(f"   환경: {self.config.environment}")
        print(f"   모드: {self.config.mode}")
        print(f"   에피소드: {self.config.episodes}")
        print(f"   디바이스: {self.device}")
        print(f"\n📁 결과 디렉토리:")
        print(f"   {self.results_dir}")
        print(f"\n📊 하이퍼파라미터:")
        print(f"   Learning Rate: {self.config.learning_rate}")
        print(f"   Batch Size: {self.config.batch_size}")
        print(f"   Buffer Size: {self.config.buffer_size}")
        print(f"   Gamma: {self.config.gamma}")
        
        if self.config.algorithm == "DQN":
            print(f"   Epsilon Decay: {self.config.epsilon_decay}")
            print(f"   Target Update: {self.config.target_update_freq}")
            if self.config.use_double_dqn:
                print(f"   Double DQN: {self.config.use_double_dqn}")
        else:
            print(f"   Tau: {self.config.tau}")
            print(f"   Noise Decay: {self.config.noise_decay}")
        print(f"{'='*60}\n")
    
    def create_agent(self, env) -> Any:
        """에이전트 생성"""
        state_dim = env.observation_space.shape[0]
        
        # 공통 파라미터
        common_params = {
            'gamma': self.config.gamma,
            'buffer_size': self.config.buffer_size,
            'batch_size': self.config.batch_size,
            'device': self.device
        }
        
        # 알고리즘 및 환경별 에이전트 생성
        if self.config.environment == "CartPole-v1":
            if self.config.algorithm == "DQN":
                # Improved DQN 사용 여부
                if self.config.mode == "improved":
                    return ImprovedDQNAgent(
                        state_dim=state_dim,
                        action_dim=env.action_space.n,
                        learning_rate=self.config.learning_rate,
                        epsilon=self.config.epsilon,
                        epsilon_min=self.config.epsilon_min,
                        epsilon_decay=self.config.epsilon_decay,
                        target_update_freq=self.config.target_update_freq,
                        use_double_dqn=self.config.use_double_dqn,
                        **common_params
                    )
                else:
                    return DQNAgent(
                        state_dim=state_dim,
                        action_dim=env.action_space.n,
                        learning_rate=self.config.learning_rate,
                        epsilon=self.config.epsilon,
                        epsilon_min=self.config.epsilon_min,
                        epsilon_decay=self.config.epsilon_decay,
                        target_update_freq=self.config.target_update_freq,
                        **common_params
                    )
            elif self.config.algorithm == "DDPG":
                return DiscreteDDPGAgent(
                    state_dim=state_dim,
                    num_actions=env.action_space.n,
                    actor_lr=self.config.actor_lr,
                    critic_lr=self.config.critic_lr,
                    tau=self.config.tau,
                    noise_sigma=self.config.noise_sigma,
                    noise_decay=self.config.noise_decay,
                    **common_params
                )
                
        elif self.config.environment == "Pendulum-v1":
            if self.config.algorithm == "DQN":
                return DiscretizedDQNAgent(
                    state_dim=state_dim,
                    action_bound=2.0,
                    num_actions=21,
                    learning_rate=self.config.learning_rate,
                    epsilon=self.config.epsilon,
                    epsilon_min=self.config.epsilon_min,
                    epsilon_decay=self.config.epsilon_decay,
                    target_update_freq=self.config.target_update_freq,
                    **common_params
                )
            elif self.config.algorithm == "DDPG":
                return DDPGAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.shape[0],
                    actor_lr=self.config.actor_lr,
                    critic_lr=self.config.critic_lr,
                    tau=self.config.tau,
                    noise_sigma=self.config.noise_sigma * 2,  # Pendulum은 더 큰 노이즈
                    noise_decay=self.config.noise_decay,
                    **common_params
                )
        
        raise ValueError(f"지원하지 않는 조합: {self.config.algorithm} + {self.config.environment}")
    
    def train(self) -> Dict[str, Any]:
        """학습 실행"""
        # 환경 생성
        env = self.env_factory.create_env(
            env_name=self.config.environment,
            agent_type='auto',
            training=True,
            seed=self.config.seed
        )
        
        # 에이전트 생성
        agent = self.create_agent(env)
        
        # 학습 메트릭
        episode_rewards = []
        episode_lengths = []
        losses = []
        checkpoint_episodes = []
        
        # 학습 시작
        start_time = time.time()
        best_avg_reward = -float('inf')
        
        print(f"🎯 학습 시작: {self.config.algorithm} + {self.config.environment}")
        print(f"   모드: {self.config.mode}")
        print()
        
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            max_steps = 500 if self.config.environment == "CartPole-v1" else 200
            
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
            
            # 에피소드 메트릭 저장
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_losses:
                avg_loss = np.mean([loss.get('loss', 0) for loss in episode_losses])
                losses.append(avg_loss)
            
            # 체크포인트 저장 (10 에피소드마다)
            if episode % 10 == 0:
                self._save_checkpoint(agent, episode, episode_reward, episode_rewards)
                checkpoint_episodes.append(episode)
            
            # 진행상황 출력 (100 에피소드마다)
            if episode % 100 == 0:
                metrics = self._calculate_metrics(
                    episode, episode_reward, episode_rewards, 
                    episode_lengths, start_time, agent
                )
                self._print_progress(metrics)
                
                # 최고 성능 모델 저장
                if metrics['avg_reward'] > best_avg_reward:
                    best_avg_reward = metrics['avg_reward']
                    self._save_best_model(agent, metrics)
        
        # 학습 완료
        env.close()
        
        # 최종 결과 계산 및 저장
        result = self._finalize_training(
            episode_rewards, episode_lengths, losses, 
            checkpoint_episodes, best_avg_reward, start_time
        )
        
        return result
    
    def _calculate_metrics(self, episode: int, episode_reward: float,
                          episode_rewards: List[float], episode_lengths: List[int],
                          start_time: float, agent: Any) -> Dict[str, Any]:
        """메트릭 계산"""
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
        elapsed = time.time() - start_time
        progress = (episode + 1) / self.config.episodes * 100
        
        # 탐험 파라미터
        if hasattr(agent, 'epsilon'):
            explore_param = agent.epsilon
            explore_name = "epsilon"
        elif hasattr(agent, 'noise') and hasattr(agent.noise, 'sigma'):
            explore_param = agent.noise.sigma
            explore_name = "sigma"
        else:
            explore_param = None
            explore_name = None
        
        return {
            'episode': episode,
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'elapsed_time': elapsed,
            'progress': progress,
            'explore_param': explore_param,
            'explore_name': explore_name
        }
    
    def _print_progress(self, metrics: Dict[str, Any]):
        """진행상황 출력"""
        episode = metrics['episode']
        progress = metrics['progress']
        episode_reward = metrics['episode_reward']
        avg_reward = metrics['avg_reward']
        elapsed = metrics['elapsed_time']
        
        progress_str = f"📊 Episode {episode:4d} ({progress:5.1f}%) | "
        progress_str += f"Reward: {episode_reward:7.1f} | "
        progress_str += f"Avg(100): {avg_reward:7.1f} | "
        
        if metrics['explore_param'] is not None:
            symbol = "ε" if metrics['explore_name'] == "epsilon" else "σ"
            progress_str += f"{symbol}: {metrics['explore_param']:.4f} | "
        
        progress_str += f"Time: {elapsed/60:.1f}m"
        
        print(progress_str)
    
    def _save_checkpoint(self, agent: Any, episode: int, 
                        episode_reward: float, episode_rewards: List[float]):
        """체크포인트 저장"""
        model_path = self.checkpoints_dir / f"episode_{episode:04d}.pth"
        agent.save(str(model_path))
        
        # 메타데이터 저장
        metadata = {
            'episode': int(episode),
            'reward': float(episode_reward),
            'timestamp': datetime.now().isoformat(),
            'avg_reward_100': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards)),
            'mode': self.config.mode,
            'algorithm': self.config.algorithm,
            'environment': self.config.environment
        }
        
        metadata_path = self.checkpoints_dir / f"episode_{episode:04d}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_best_model(self, agent: Any, metrics: Dict[str, Any]):
        """최고 성능 모델 저장"""
        best_model_path = self.results_dir / "best_model.pth"
        agent.save(str(best_model_path))
        
        # 최고 모델 메타데이터
        best_meta = {
            'episode': metrics['episode'],
            'avg_reward': metrics['avg_reward'],
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.mode
        }
        
        best_meta_path = self.results_dir / "best_model_meta.json"
        with open(best_meta_path, 'w') as f:
            json.dump(best_meta, f, indent=2)
    
    def _finalize_training(self, episode_rewards: List[float], 
                          episode_lengths: List[int], losses: List[float],
                          checkpoint_episodes: List[int], best_avg_reward: float,
                          start_time: float) -> Dict[str, Any]:
        """학습 완료 및 결과 저장"""
        total_time = time.time() - start_time
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # 수렴 판정
        converged = False
        if self.config.environment == "CartPole-v1":
            converged = final_avg_reward >= 475.0
        elif self.config.environment == "Pendulum-v1":
            converged = final_avg_reward >= -200.0
        
        print(f"\n✅ 학습 완료: {self.config.algorithm} + {self.config.environment}")
        print(f"📊 최종 평균 보상: {final_avg_reward:.1f}")
        print(f"📊 최고 평균 보상: {best_avg_reward:.1f}")
        print(f"📊 수렴 여부: {'✅' if converged else '❌'}")
        print(f"⏰ 총 시간: {total_time/60:.1f}분")
        print(f"💾 체크포인트: {len(checkpoint_episodes)}개")
        
        # 결과 딕셔너리
        result = {
            'config': asdict(self.config),
            'results': {
                'final_avg_reward': float(final_avg_reward),
                'best_avg_reward': float(best_avg_reward),
                'converged': bool(converged),
                'max_reward': float(max(episode_rewards)),
                'min_reward': float(min(episode_rewards)),
                'training_time': float(total_time),
                'num_checkpoints': len(checkpoint_episodes)
            },
            'metrics': {
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': [int(l) for l in episode_lengths],
                'losses': [float(l) for l in losses] if losses else []
            },
            'metadata': {
                'timestamp': self.timestamp,
                'results_dir': str(self.results_dir),
                'checkpoints_dir': str(self.checkpoints_dir),
                'checkpoint_episodes': [int(e) for e in checkpoint_episodes]
            }
        }
        
        # 결과 파일 저장
        result_file = self.results_dir / "training_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 결과 저장: {result_file}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='통합 학습 시스템')
    
    # 기본 설정
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['DQN', 'DDPG'], help='학습 알고리즘')
    parser.add_argument('--environment', type=str, required=True,
                       choices=['CartPole-v1', 'Pendulum-v1'], help='환경')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'fair', 'improved', 'custom'],
                       help='학습 모드 (standard/fair/improved/custom)')
    parser.add_argument('--episodes', type=int, default=2000, help='에피소드 수')
    parser.add_argument('--device', type=str, default=None, help='디바이스')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    # 커스텀 하이퍼파라미터 (--mode custom 일 때)
    parser.add_argument('--learning-rate', type=float, help='학습률')
    parser.add_argument('--batch-size', type=int, help='배치 크기')
    parser.add_argument('--epsilon-decay', type=float, help='Epsilon 감소율 (DQN)')
    parser.add_argument('--target-update-freq', type=int, help='타겟 업데이트 주기 (DQN)')
    parser.add_argument('--tau', type=float, help='Soft update 파라미터 (DDPG)')
    parser.add_argument('--noise-decay', type=float, help='노이즈 감소율 (DDPG)')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 설정 생성
    config = TrainingConfig(
        algorithm=args.algorithm,
        environment=args.environment,
        mode=args.mode,
        episodes=args.episodes,
        device=args.device,
        seed=args.seed
    )
    
    # 커스텀 모드일 때 파라미터 적용
    if args.mode == 'custom':
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epsilon_decay is not None:
            config.epsilon_decay = args.epsilon_decay
        if args.target_update_freq is not None:
            config.target_update_freq = args.target_update_freq
        if args.tau is not None:
            config.tau = args.tau
        if args.noise_decay is not None:
            config.noise_decay = args.noise_decay
    
    # 통합 학습 시스템 실행
    trainer = UnifiedTrainingSystem(config)
    result = trainer.train()
    
    # 간단한 결과 출력
    print(f"\n🎯 최종 결과:")
    print(f"   알고리즘: {config.algorithm}")
    print(f"   환경: {config.environment}")
    print(f"   모드: {config.mode}")
    print(f"   최종 평균 보상: {result['results']['final_avg_reward']:.1f}")
    print(f"   수렴 여부: {'✅' if result['results']['converged'] else '❌'}")


if __name__ == "__main__":
    main()