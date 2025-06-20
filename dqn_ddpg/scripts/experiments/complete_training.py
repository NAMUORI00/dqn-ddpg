#!/usr/bin/env python3
"""
GPU 기반 완전학습 파이프라인
DQN과 DDPG의 완전학습을 수행하며 상세한 모니터링과 체크포인트를 제공합니다.
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from collections import deque
from datetime import datetime
import gymnasium as gym

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 모듈 import
from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import (
    set_seed, get_device, get_gpu_memory_info, 
    enable_mixed_precision, set_cuda_options, optimize_gpu_memory
)
from src.core.episode_checkpoint import EpisodeCheckpointManager, EpisodeMetrics
from src.core.gpu_config import GPUConfig, load_gpu_config
from src.core.advanced_convergence import AdvancedConvergenceDetector


class CompleteTrainingPipeline:
    """완전학습 파이프라인
    
    GPU 최적화, 체크포인트 관리, 상세 모니터링을 포함한 완전학습 시스템
    """
    
    def __init__(self, config_path: str, algorithm: str, gpu_config: GPUConfig):
        """
        Args:
            config_path: 알고리즘 설정 파일 경로
            algorithm: 알고리즘 이름 ('dqn' 또는 'ddpg')
            gpu_config: GPU 설정 객체
        """
        self.algorithm = algorithm.lower()
        self.gpu_config = gpu_config
        
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # GPU 최적화 설정 적용
        if self.gpu_config.enabled:
            self._setup_gpu_optimizations()
        
        # 디바이스 설정
        self.device = self.gpu_config.device
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 환경 이름 추출 (설정에서)
        env_name = self.config['environment']['name'].lower()
        
        # 체크포인트 매니저 초기화 (모든 에피소드 저장하도록 설정)
        self.checkpoint_manager = EpisodeCheckpointManager(
            base_path="models",
            algorithm=self.algorithm,
            environment=env_name,  # 환경별 구분
            save_frequency=1,  # 모든 에피소드 저장
            keep_latest=10000,  # 최대한 많이 보관
            keep_best=100,  # 최고 성능 모델들도 더 많이 보관
            auto_cleanup=False  # 자동 정리 비활성화
        )
        
        # 학습 통계
        self.training_start_time = None
        self.episode_times = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=500)  # 더 긴 히스토리 유지
        self.recent_losses = deque(maxlen=500)
        
        # 고급 수렴 감지기
        self.convergence_detector = AdvancedConvergenceDetector(
            min_episodes=500,
            stability_window=200,
            cv_threshold=0.05,
            improvement_threshold=0.01,
            plateau_patience=150,
            confidence_threshold=0.85
        )
        
        # 결과 디렉토리 생성
        self.results_dir = Path("results") / self.algorithm
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_gpu_optimizations(self):
        """GPU 최적화 설정"""
        print("🚀 GPU 최적화 설정 중...")
        
        if torch.cuda.is_available():
            # CUDA 옵션 설정
            cuda_options = self.gpu_config.get_cuda_options()
            set_cuda_options(cuda_options)
            
            # Mixed Precision 활성화
            if self.gpu_config.use_mixed_precision:
                enable_mixed_precision()
                print("  ✓ Mixed Precision Training 활성화")
            
            # GPU 메모리 정보 출력
            gpu_info = get_gpu_memory_info()
            print(f"  ✓ GPU 메모리: {gpu_info.get('memory_used_gb', 0):.1f}GB / {gpu_info.get('memory_total_gb', 0):.1f}GB")
        
        print("  ✓ GPU 최적화 설정 완료")
    
    def create_environment(self):
        """환경 생성"""
        env_name = self.config['environment']['name']
        
        if self.algorithm == 'dqn':
            env = create_dqn_env(env_name)
        elif self.algorithm == 'ddpg':
            env = create_ddpg_env(env_name)
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {self.algorithm}")
        
        return env
    
    def create_agent(self, env):
        """에이전트 생성"""
        # 기본 파라미터
        state_dim = env.observation_space.shape[0]
        
        if self.algorithm == 'dqn':
            action_dim = env.action_space.n
            base_kwargs = {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'learning_rate': self.config['agent']['learning_rate'],
                'gamma': self.config['agent']['gamma'],
                'epsilon': self.config['agent']['epsilon'],
                'epsilon_min': self.config['agent']['epsilon_min'],
                'epsilon_decay': self.config['agent']['epsilon_decay'],
                'buffer_size': self.config['agent']['buffer_size'],
                'batch_size': self.config['agent']['batch_size'],
                'target_update_freq': self.config['agent'].get('target_update_freq', 100)
            }
            
            # GPU 옵션 추가
            agent_kwargs = self.gpu_config.create_agent_kwargs(base_kwargs)
            agent = DQNAgent(**agent_kwargs)
            
        elif self.algorithm == 'ddpg':
            action_dim = env.action_space.shape[0]
            base_kwargs = {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'actor_lr': self.config['agent']['actor_lr'],
                'critic_lr': self.config['agent']['critic_lr'],
                'gamma': self.config['agent']['gamma'],
                'tau': self.config['agent']['tau'],
                'noise_sigma': self.config['agent']['noise_sigma'],
                'buffer_size': self.config['agent']['buffer_size'],
                'batch_size': self.config['agent']['batch_size']
            }
            
            # GPU 옵션 추가
            agent_kwargs = self.gpu_config.create_agent_kwargs(base_kwargs)
            agent = DDPGAgent(**agent_kwargs)
        
        return agent
    
    def calculate_convergence_score(self, recent_rewards: deque, window_size: int = 50) -> float:
        """수렴도 점수 계산 (0-1 사이 값)"""
        if len(recent_rewards) < window_size:
            return 0.0
        
        # 최근 보상들의 분산 계산
        rewards_array = np.array(list(recent_rewards)[-window_size:])
        variance = np.var(rewards_array)
        
        # 분산이 낮을수록 수렴도가 높음
        convergence_score = 1.0 / (1.0 + variance)
        return convergence_score
    
    def run_episode(self, env, agent, episode_id: int) -> EpisodeMetrics:
        """단일 에피소드 실행"""
        episode_start_time = time.time()
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while True:
            # 행동 선택
            if self.algorithm == 'dqn':
                action = agent.select_action(state)
            else:  # ddpg
                action = agent.select_action(state, add_noise=True)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 에이전트 업데이트
            if self.algorithm == 'dqn':
                loss = agent.update()
            else:  # ddpg
                # Warmup 체크
                warmup_steps = self.config.get('training', {}).get('warmup_steps', 1000)
                if len(agent.buffer) > warmup_steps:
                    loss = agent.update()
                else:
                    loss = 0.0
            
            if loss is not None and loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 에피소드 메트릭 생성
        episode_time = time.time() - episode_start_time
        self.episode_times.append(episode_time)
        self.recent_rewards.append(total_reward)
        
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            self.recent_losses.append(avg_loss)
        else:
            avg_loss = 0.0
        
        # 탐험율 계산
        if self.algorithm == 'dqn':
            exploration_rate = agent.epsilon
        else:  # ddpg
            exploration_rate = agent.noise_sigma if hasattr(agent, 'noise_sigma') else 0.1
        
        # 수렴도 계산
        convergence_score = self.calculate_convergence_score(self.recent_rewards)
        
        # 성공률 계산 (환경에 따라 다르게 정의)
        success_rate = 1.0 if total_reward > 0 else 0.0
        
        # GPU 메모리 정보
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            gpu_memory_used = gpu_info.get('memory_used_gb', 0.0)
            gpu_memory_total = gpu_info.get('memory_total_gb', 0.0)
        
        metrics = EpisodeMetrics(
            episode_id=episode_id,
            total_reward=total_reward,
            episode_length=steps,
            average_loss=avg_loss,
            exploration_rate=exploration_rate,
            timestamp=time.time(),
            success_rate=success_rate,
            convergence_score=convergence_score,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            training_time=episode_time,
            extra_metrics={
                'steps_per_second': steps / episode_time if episode_time > 0 else 0,
                'memory_usage_percent': (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
            }
        )
        
        return metrics
    
    def print_progress(self, episode_id: int, metrics: EpisodeMetrics, convergence_metrics=None):
        """진행 상황 출력 (수렴 정보 포함)"""
        if episode_id % 10 == 0:
            # 통계 계산
            avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            avg_time = np.mean(self.episode_times) if self.episode_times else 0
            
            # 진행 상황 출력
            elapsed_time = time.time() - self.training_start_time
            elapsed_hours = elapsed_time / 3600
            
            print(f"\n📊 Episode {episode_id:4d} | "
                  f"Reward: {metrics.total_reward:8.2f} | "
                  f"Avg(500): {avg_reward:7.2f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Steps: {metrics.episode_length:3d} | "
                  f"Explore: {metrics.exploration_rate:.3f}")
            
            print(f"     Time: {avg_time:5.2f}s/ep | "
                  f"Elapsed: {elapsed_hours:5.2f}h | "
                  f"Conv: {metrics.convergence_score:.3f}")
            
            # 고급 수렴 정보 출력
            if convergence_metrics:
                print(f"     🎯 CV: {convergence_metrics.coefficient_of_variation:.4f} | "
                      f"개선율: {convergence_metrics.improvement_rate:.4f} | "
                      f"신뢰도: {convergence_metrics.confidence_level:.3f} | "
                      f"정체: {convergence_metrics.plateau_duration}ep")
                
                if convergence_metrics.is_converged:
                    print(f"     ✅ 수렴 상태: {convergence_metrics.convergence_reason}")
            
            if torch.cuda.is_available():
                print(f"     GPU: {metrics.gpu_memory_used:.1f}GB / {metrics.gpu_memory_total:.1f}GB "
                      f"({metrics.extra_metrics.get('memory_usage_percent', 0):.1f}%)")
        
        # 주요 수렴 상태 변화 시 즉시 출력
        elif convergence_metrics and convergence_metrics.is_converged:
            print(f"🎯 Episode {episode_id}: 수렴 감지! 신뢰도 {convergence_metrics.confidence_level:.3f}")
    
    def train(self, num_episodes: int = 10000, resume_from: int = None):
        """완전학습 실행 (수렴까지)
        
        Args:
            num_episodes: 최대 학습 에피소드 수 (수렴 시 조기 종료)
            resume_from: 재시작할 에피소드 ID (None이면 처음부터)
        """
        print(f"\n🚀 {self.algorithm.upper()} 완전학습 시작")
        print(f"📁 체크포인트 저장 경로: {self.checkpoint_manager.model_dir}")
        print(f"🎯 최대 에피소드: {num_episodes} (수렴 시 조기 종료)")
        print(f"💾 모든 에피소드 체크포인트 저장")
        print(f"🎯 수렴 조건: 200에피소드 창에서 CV < 0.05, 개선율 < 1%")
        print("=" * 80)
        
        self.training_start_time = time.time()
        
        # 환경 및 에이전트 생성
        env = self.create_environment()
        agent = self.create_agent(env)
        
        # 체크포인트에서 재시작
        start_episode = 0
        if resume_from is not None:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(episode_id=resume_from)
            if checkpoint_data is not None:
                agent.load_state_dict(checkpoint_data['model_state_dict'])
                start_episode = resume_from + 1
                print(f"📂 체크포인트에서 재시작: Episode {resume_from}")
        
        # 학습 루프
        try:
            for episode in tqdm(range(start_episode, num_episodes), 
                              desc=f"{self.algorithm.upper()} Training", 
                              initial=start_episode,
                              total=num_episodes):
                
                # 에피소드 실행
                metrics = self.run_episode(env, agent, episode)
                
                # 메트릭 기록
                self.checkpoint_manager.record_episode_metrics(metrics)
                
                # 고급 수렴 감지기 업데이트
                convergence_metrics = self.convergence_detector.update(episode, metrics.total_reward)
                
                # 체크포인트 저장 (모든 에피소드)
                self._save_episode_checkpoint(agent, episode, metrics)
                
                # 진행 상황 출력 (수렴 정보 포함)
                self.print_progress(episode, metrics, convergence_metrics)
                
                # GPU 메모리 최적화 (주기적)
                if self.gpu_config.enabled and episode % 100 == 0:
                    optimize_gpu_memory()
                
                # 고급 수렴 감지
                should_stop, stop_reason = self.convergence_detector.should_stop_training(episode, patience=100)
                if should_stop:
                    print(f"\n🎉 학습 완료! Episode {episode}")
                    print(f"📊 종료 이유: {stop_reason}")
                    break
        
        except KeyboardInterrupt:
            print(f"\n⚠️ 사용자에 의해 학습이 중단되었습니다 (Episode {episode})")
        
        finally:
            env.close()
            
            # 최종 체크포인트 저장
            final_path = self._save_episode_checkpoint(agent, episode, metrics, force_save=True)
            if final_path:
                print(f"💾 최종 모델 저장: {final_path}")
            
            # 수렴 요약 정보
            convergence_summary = self.convergence_detector.get_convergence_summary()
            print(f"\n📊 수렴 분석 요약:")
            print(f"   상태: {convergence_summary['status']}")
            print(f"   총 에피소드: {convergence_summary['total_episodes']}")
            print(f"   수렴 확률: {convergence_summary['convergence_probability']:.3f}")
            
            # 학습 요약
            self._print_training_summary()
            
            # 학습 데이터 내보내기
            self._export_results()
    
    def _check_convergence(self, episode: int, window_size: int = 200, 
                          stability_threshold: float = 0.05, 
                          min_episodes: int = 500) -> bool:
        """수렴 조건 확인 (더 엄격한 조건)"""
        # 최소 에피소드 수 확인
        if episode < min_episodes or len(self.recent_rewards) < window_size:
            return False
        
        # 최근 보상들의 안정성 확인
        rewards_array = np.array(list(self.recent_rewards)[-window_size:])
        
        # 변동 계수 (Coefficient of Variation) 확인
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        
        if mean_reward == 0:
            return False
        
        coefficient_of_variation = std_reward / abs(mean_reward)
        
        # 추가 조건: 최근 구간과 이전 구간의 성능 차이가 미미해야 함
        if len(self.recent_rewards) >= window_size * 2:
            recent_half = rewards_array[-window_size//2:]
            previous_half = rewards_array[-window_size:-window_size//2]
            
            recent_mean = np.mean(recent_half)
            previous_mean = np.mean(previous_half)
            
            # 성능 개선이 1% 미만이어야 수렴으로 판단
            if previous_mean != 0:
                improvement_rate = abs(recent_mean - previous_mean) / abs(previous_mean)
                convergence_by_improvement = improvement_rate < 0.01
            else:
                convergence_by_improvement = True
        else:
            convergence_by_improvement = True
        
        # 두 조건 모두 만족해야 수렴
        convergence_by_cv = coefficient_of_variation < stability_threshold
        
        if convergence_by_cv and convergence_by_improvement:
            print(f"\n🎯 수렴 감지 - CV: {coefficient_of_variation:.4f}, 개선율: {improvement_rate if 'improvement_rate' in locals() else 0:.4f}")
            
        return convergence_by_cv and convergence_by_improvement
    
    def _save_episode_checkpoint(self, agent, episode_id: int, metrics, force_save: bool = False) -> Optional[str]:
        """에피소드 체크포인트 저장 (알고리즘별 모델 구조 고려)"""
        try:
            if self.algorithm == 'dqn':
                # DQN: Q-Network 저장
                if hasattr(agent, 'q_network') and hasattr(agent, 'optimizer'):
                    saved_path = self.checkpoint_manager.save_checkpoint(
                        episode_id=episode_id,
                        model=agent.q_network,
                        optimizer=agent.optimizer,
                        metrics=metrics,
                        config=self.config,
                        force_save=force_save or True  # 모든 에피소드 저장
                    )
                    return saved_path
                    
            elif self.algorithm == 'ddpg':
                # DDPG: Actor와 Critic 모두 저장
                if (hasattr(agent, 'actor') and hasattr(agent, 'critic') and 
                    hasattr(agent, 'actor_optimizer') and hasattr(agent, 'critic_optimizer')):
                    
                    # Actor 저장
                    actor_path = self.checkpoint_manager.save_checkpoint(
                        episode_id=episode_id,
                        model=agent.actor,
                        optimizer=agent.actor_optimizer,
                        metrics=metrics,
                        config=self.config,
                        force_save=force_save or True
                    )
                    
                    # Critic도 별도 저장 (선택사항)
                    if actor_path:
                        # Critic 가중치를 같은 파일에 추가로 저장
                        import torch
                        checkpoint_data = torch.load(actor_path)
                        checkpoint_data['critic_state_dict'] = agent.critic.state_dict()
                        checkpoint_data['critic_optimizer_state_dict'] = agent.critic_optimizer.state_dict()
                        if hasattr(agent, 'target_actor'):
                            checkpoint_data['target_actor_state_dict'] = agent.target_actor.state_dict()
                        if hasattr(agent, 'target_critic'):
                            checkpoint_data['target_critic_state_dict'] = agent.target_critic.state_dict()
                        torch.save(checkpoint_data, actor_path)
                    
                    return actor_path
            
            return None
            
        except Exception as e:
            print(f"⚠️ 체크포인트 저장 오류: {e}")
            return None
    
    def _print_training_summary(self):
        """학습 요약 출력"""
        summary = self.checkpoint_manager.get_training_summary()
        
        print("\n" + "=" * 80)
        print(f"🎯 {self.algorithm.upper()} 학습 완료 요약")
        print("=" * 80)
        
        print(f"📊 총 에피소드: {summary.get('total_episodes', 0)}")
        print(f"💾 저장된 체크포인트: {summary.get('total_checkpoints', 0)}")
        print(f"💿 총 저장 용량: {summary.get('total_storage_mb', 0):.2f}MB")
        
        if 'reward_stats' in summary:
            reward_stats = summary['reward_stats']
            print(f"🏆 보상 통계:")
            print(f"   - 평균: {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
            print(f"   - 최고: {reward_stats['max']:.2f}")
            print(f"   - 최근 100ep 평균: {reward_stats['latest_100_mean']:.2f}")
        
        if 'best_model' in summary:
            best = summary['best_model']
            print(f"🥇 최고 성능 모델:")
            print(f"   - Episode: {best['episode_id']}")
            print(f"   - 보상: {best['reward']:.2f}")
            print(f"   - 경로: {best['path']}")
        
        total_time = time.time() - self.training_start_time
        print(f"⏰ 총 학습 시간: {total_time/3600:.2f}시간")
    
    def _export_results(self):
        """결과 내보내기"""
        # JSON 형태로 내보내기
        json_path = self.checkpoint_manager.export_training_data('json')
        
        # CSV 형태로 내보내기 (pandas가 설치된 경우)
        try:
            csv_path = self.checkpoint_manager.export_training_data('csv')
            print(f"📄 CSV 데이터: {csv_path}")
        except ImportError:
            print("⚠️ pandas가 설치되지 않아 CSV 내보내기를 건너뜁니다.")
        
        print(f"📄 JSON 데이터: {json_path}")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='GPU 기반 DQN/DDPG 완전학습')
    
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ddpg'], required=True,
                       help='학습할 알고리즘 (dqn 또는 ddpg)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='학습할 에피소드 수 (default: 1000)')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로 (default: configs/{algorithm}_config.yaml)')
    parser.add_argument('--resume', type=int, default=None,
                       help='재시작할 에피소드 ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (default: 42)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='사용할 GPU ID')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 사용 비활성화')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 설정 파일 경로
    if args.config is None:
        config_path = f"configs/{args.algorithm}_config.yaml"
    else:
        config_path = args.config
    
    # GPU 설정 로드
    gpu_config = load_gpu_config(
        config_path=config_path,
        device_id=args.gpu_id,
        force_cpu=args.no_gpu
    )
    
    print(f"🔧 GPU 설정: {gpu_config}")
    
    # 학습 파이프라인 생성
    pipeline = CompleteTrainingPipeline(
        config_path=config_path,
        algorithm=args.algorithm,
        gpu_config=gpu_config
    )
    
    # 학습 실행
    pipeline.train(
        num_episodes=args.episodes,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()