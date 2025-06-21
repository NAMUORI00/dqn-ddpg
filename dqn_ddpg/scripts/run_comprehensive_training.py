#!/usr/bin/env python3
"""
종합 훈련 스크립트 - GPU 기반 DQN vs DDPG 환경별 학습
리팩토링된 아키텍처 기반으로 체계적인 학습 및 저장을 수행합니다.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Refactored 컴포넌트 import
from src.core.config_manager import ConfigManager
from src.core.logging_system import setup_logging, get_logger
from src.core.episode_checkpoint import EpisodeCheckpointManager
from src.core.advanced_convergence import AdvancedConvergenceDetector
from src.core.training_monitor import TrainingMonitor
from src.environments.factory import EnvironmentFactory
from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.core.exceptions import DQNDDPGException, TrainingError


class ComprehensiveTrainingManager:
    """종합 훈련 매니저 - 환경별 체계적 학습"""
    
    def __init__(self, config_environment: str = "production"):
        """
        Args:
            config_environment: 설정 환경 ('development', 'production', 'testing')
        """
        # 설정 시스템 초기화
        self.config_manager = ConfigManager(environment=config_environment)
        self.logger = setup_logging(self.config_manager).get_training_logger()
        
        # 환경 팩토리 초기화
        self.env_factory = EnvironmentFactory(self.config_manager)
        
        # 훈련 모니터는 개별 훈련에서 초기화
        
        # 결과 저장 경로 설정
        self.base_results_path = Path("results/comprehensive_training")
        self.base_models_path = Path("models")
        self.base_results_path.mkdir(parents=True, exist_ok=True)
        self.base_models_path.mkdir(parents=True, exist_ok=True)
        
        # 학습 세션 ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"종합 훈련 매니저 초기화 완료", session_id=self.session_id)
    
    def train_environment(self, 
                         env_name: str, 
                         algorithm: str, 
                         max_episodes: int = 1000,
                         convergence_patience: int = 100) -> Dict[str, Any]:
        """환경별 알고리즘 훈련
        
        Args:
            env_name: 환경 이름 ('CartPole-v1', 'ContinuousCartPole', 'Pendulum-v1')
            algorithm: 알고리즘 ('dqn', 'ddpg')
            max_episodes: 최대 에피소드 수
            convergence_patience: 수렴 감지 인내심
            
        Returns:
            훈련 결과 딕셔너리
        """
        self.logger.info(f"환경별 훈련 시작", 
                        env_name=env_name, 
                        algorithm=algorithm,
                        max_episodes=max_episodes)
        
        try:
            # 환경 생성
            env = self._create_environment(env_name, algorithm)
            
            # 에이전트 생성
            agent = self._create_agent(algorithm, env, env_name)
            
            # 체크포인트 매니저 설정
            checkpoint_manager = EpisodeCheckpointManager(
                base_path=self.base_models_path,
                algorithm=algorithm,
                environment=env_name
            )
            
            # 수렴 감지기 설정
            convergence_detector = AdvancedConvergenceDetector(
                min_episodes=50,
                plateau_patience=convergence_patience,
                stability_window=min(100, convergence_patience)
            )
            
            # 훈련 모니터 설정
            training_monitor = TrainingMonitor(
                algorithm=algorithm,
                log_dir=str(self.base_results_path / "logs"),
                enable_gpu_monitoring=True
            )
            
            # 훈련 수행
            training_results = self._execute_training(
                env=env,
                agent=agent,
                checkpoint_manager=checkpoint_manager,
                convergence_detector=convergence_detector,
                training_monitor=training_monitor,
                max_episodes=max_episodes,
                env_name=env_name,
                algorithm=algorithm
            )
            
            # 결과 저장
            self._save_training_results(training_results, env_name, algorithm)
            
            return training_results
            
        except Exception as e:
            import traceback
            error_msg = f"환경 {env_name}, 알고리즘 {algorithm} 훈련 실패: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg, env_name=env_name, algorithm=algorithm, error=str(e), traceback=traceback_str)
            print(f"DEBUG: Detailed error info:\n{traceback_str}")  # 추가 디버깅
            raise TrainingError(error_msg) from e
        
        finally:
            if 'env' in locals():
                env.close()
    
    def _create_environment(self, env_name: str, algorithm: str):
        """환경 생성"""
        self.logger.debug(f"환경 생성", env_name=env_name, algorithm=algorithm)
        
        # 환경별 설정 가져오기 (현재는 팩토리에서 처리)
        env_config = {}
        
        # 환경 생성
        env = self.env_factory.create_training_env(algorithm=algorithm)
        
        return env
    
    def _create_agent(self, algorithm: str, env, env_name: str):
        """에이전트 생성"""
        self.logger.debug(f"에이전트 생성", algorithm=algorithm, env_name=env_name)
        
        # 알고리즘별 설정 가져오기
        full_config = self.config_manager.get_algorithm_config(algorithm)
        agent_config = full_config.get('agent', {})
        gpu_config_data = self.config_manager.get_gpu_config()
        
        # GPU 디바이스 설정
        from src.core.utils import get_device
        device = get_device(gpu_config_data.device_id if gpu_config_data.enabled else 'cpu')
        use_gpu_buffer = gpu_config_data.enabled and gpu_config_data.memory_fraction > 0
        use_mixed_precision = gpu_config_data.enabled and gpu_config_data.mixed_precision
        
        # 환경 정보
        state_dim = env.observation_space.shape[0]
        
        if algorithm == 'dqn':
            action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 11  # Discretized continuous
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                **agent_config,
                device=device,
                use_gpu_buffer=use_gpu_buffer,
                use_mixed_precision=use_mixed_precision
            )
        elif algorithm == 'ddpg':
            action_dim = env.action_space.shape[0]
            action_bound = float(env.action_space.high[0])
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bound=action_bound,
                **agent_config,
                device=device,
                use_gpu_buffer=use_gpu_buffer,
                use_mixed_precision=use_mixed_precision
            )
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {algorithm}")
        
        return agent
    
    def _get_target_performance(self, env_name: str) -> float:
        """환경별 목표 성능 반환"""
        targets = {
            'CartPole-v1': 195.0,
            'ContinuousCartPole': 195.0,
            'Pendulum-v1': -200.0  # 높을수록 좋음 (음수이므로)
        }
        return targets.get(env_name, 195.0)
    
    def _execute_training(self, 
                         env, 
                         agent, 
                         checkpoint_manager,
                         convergence_detector,
                         training_monitor,
                         max_episodes: int,
                         env_name: str,
                         algorithm: str) -> Dict[str, Any]:
        """실제 훈련 수행"""
        
        # 훈련 결과 저장용
        episode_rewards = []
        episode_lengths = []
        episode_losses = []
        convergence_info = {}
        training_start_time = time.time()
        
        self.logger.info(f"훈련 시작", 
                        env_name=env_name, 
                        algorithm=algorithm,
                        max_episodes=max_episodes)
        
        for episode in range(max_episodes):
            episode_start_time = time.time()
            
            # 에피소드 실행
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # gymnasium 형식 처리
            
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            loss_count = 0
            
            done = False
            while not done:
                # 행동 선택
                action = agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장 (action을 적절한 형태로 변환)
                if isinstance(action, dict):
                    action_for_buffer = list(action.values())[0] if action else 0
                else:
                    action_for_buffer = action
                    
                agent.buffer.push(state, action_for_buffer, reward, next_state, done)
                
                # 학습
                if len(agent.buffer) > agent.batch_size:
                    loss_dict = agent.update()
                    if loss_dict and 'loss' in loss_dict:
                        episode_loss += loss_dict['loss']
                        loss_count += 1
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # 에피소드 통계 계산
            avg_episode_loss = episode_loss / max(loss_count, 1)
            episode_duration = time.time() - episode_start_time
            
            # 결과 기록
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_losses.append(avg_episode_loss)
            
            # 로깅
            if episode % 10 == 0:
                recent_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                self.logger.info(f"에피소드 {episode}", 
                               episode=episode,
                               reward=episode_reward,
                               recent_avg_reward=recent_reward,
                               duration=episode_duration,
                               loss=avg_episode_loss)
            
            # 체크포인트 저장 (매 10 에피소드마다)
            if episode % 10 == 0:
                # EpisodeMetrics 생성
                from src.core.episode_checkpoint import EpisodeMetrics
                metrics = EpisodeMetrics(
                    episode_id=episode,
                    total_reward=episode_reward,
                    episode_length=episode_length,
                    average_loss=avg_episode_loss,
                    exploration_rate=getattr(agent, 'epsilon', 0.0),
                    timestamp=time.time()
                )
                
                # 모델과 옵티마이저 선택 (알고리즘별)
                if algorithm == 'dqn':
                    model = agent.q_network
                    optimizer = agent.optimizer
                elif algorithm == 'ddpg':
                    model = agent.actor  # DDPG의 경우 actor 네트워크 저장
                    optimizer = agent.actor_optimizer
                else:
                    # 일반적인 경우 (BaseAgent에서 첫 번째 네트워크와 옵티마이저)
                    model = list(agent.networks.values())[0] if hasattr(agent, 'networks') else None
                    optimizer = list(agent.optimizers.values())[0] if hasattr(agent, 'optimizers') else None
                
                # 체크포인트 저장
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    episode_id=episode,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    config={'algorithm': algorithm, 'env_name': env_name},
                    force_save=True
                )
                if checkpoint_path:
                    self.logger.debug(f"체크포인트 저장", episode=episode, path=checkpoint_path)
            
            # 수렴 감지
            conv_metrics = convergence_detector.update(episode, episode_reward)
            if conv_metrics.is_converged:
                convergence_info = {
                    'converged': True,
                    'convergence_episode': episode,
                    'convergence_reason': conv_metrics.convergence_reason,
                    'final_performance': episode_reward,
                    'cv': conv_metrics.coefficient_of_variation,
                    'stability_score': conv_metrics.stability_score
                }
                self.logger.info(f"수렴 감지됨", **convergence_info)
                break
        
        # 최종 모델 저장
        if episode_rewards:
            final_metrics = EpisodeMetrics(
                episode_id=episode,
                total_reward=episode_rewards[-1],
                episode_length=episode_lengths[-1] if episode_lengths else 0,
                average_loss=episode_losses[-1] if episode_losses else 0,
                exploration_rate=getattr(agent, 'epsilon', 0.0),
                timestamp=time.time()
            )
            
            # 최종 모델과 옵티마이저 선택
            if algorithm == 'dqn':
                final_model = agent.q_network
                final_optimizer = agent.optimizer
            elif algorithm == 'ddpg':
                final_model = agent.actor
                final_optimizer = agent.actor_optimizer
            else:
                final_model = list(agent.networks.values())[0] if hasattr(agent, 'networks') else None
                final_optimizer = list(agent.optimizers.values())[0] if hasattr(agent, 'optimizers') else None
                
            final_model_path = checkpoint_manager.save_checkpoint(
                episode_id=episode,
                model=final_model,
                optimizer=final_optimizer,
                metrics=final_metrics,
                config={'algorithm': algorithm, 'env_name': env_name, 'final_model': True},
                force_save=True
            )
        else:
            final_model_path = None
        
        # 훈련 완료
        training_duration = time.time() - training_start_time
        
        # 결과 정리
        training_results = {
            'env_name': env_name,
            'algorithm': algorithm,
            'session_id': self.session_id,
            'total_episodes': len(episode_rewards),
            'max_episodes': max_episodes,
            'training_duration': training_duration,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_losses': episode_losses,
            'convergence_info': convergence_info,
            'final_model_path': str(final_model_path),
            'final_performance': {
                'avg_reward_last_100': sum(episode_rewards[-100:]) / min(len(episode_rewards), 100),
                'max_reward': max(episode_rewards) if episode_rewards else 0,
                'min_reward': min(episode_rewards) if episode_rewards else 0,
                'final_reward': episode_rewards[-1] if episode_rewards else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"훈련 완료", 
                        env_name=env_name, 
                        algorithm=algorithm,
                        total_episodes=len(episode_rewards),
                        final_avg_reward=training_results['final_performance']['avg_reward_last_100'])
        
        return training_results
    
    def _save_training_results(self, results: Dict[str, Any], env_name: str, algorithm: str):
        """훈련 결과 저장"""
        # 결과 파일 경로
        results_file = self.base_results_path / f"{algorithm}_{env_name}_{self.session_id}.json"
        
        # JSON 직렬화 가능하도록 처리
        serializable_results = self._make_json_serializable(results)
        
        # 파일 저장
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"훈련 결과 저장", 
                        results_file=str(results_file),
                        env_name=env_name,
                        algorithm=algorithm)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return str(obj)
    
    def run_comprehensive_training(self, 
                                  environments: List[str] = None,
                                  algorithms: List[str] = None,
                                  max_episodes: int = 1000) -> Dict[str, Any]:
        """종합 훈련 실행
        
        Args:
            environments: 훈련할 환경 목록
            algorithms: 사용할 알고리즘 목록  
            max_episodes: 최대 에피소드 수
            
        Returns:
            모든 훈련 결과
        """
        if environments is None:
            environments = ['CartPole-v1', 'ContinuousCartPole']
        
        if algorithms is None:
            algorithms = ['dqn', 'ddpg']
        
        all_results = {}
        
        self.logger.info(f"종합 훈련 시작", 
                        environments=environments,
                        algorithms=algorithms,
                        max_episodes=max_episodes)
        
        for env_name in environments:
            all_results[env_name] = {}
            
            for algorithm in algorithms:
                try:
                    self.logger.info(f"훈련 시작: {algorithm} on {env_name}")
                    
                    result = self.train_environment(
                        env_name=env_name,
                        algorithm=algorithm,
                        max_episodes=max_episodes
                    )
                    
                    all_results[env_name][algorithm] = result
                    
                except Exception as e:
                    error_result = {
                        'error': str(e),
                        'env_name': env_name,
                        'algorithm': algorithm,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results[env_name][algorithm] = error_result
                    self.logger.error(f"훈련 실패", env_name=env_name, algorithm=algorithm, error=str(e))
        
        # 종합 결과 저장
        summary_file = self.base_results_path / f"comprehensive_summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(all_results), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"종합 훈련 완료", summary_file=str(summary_file))
        
        return all_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='DQN vs DDPG 종합 훈련')
    parser.add_argument('--environments', nargs='+', 
                       default=['CartPole-v1', 'ContinuousCartPole'],
                       help='훈련할 환경 목록')
    parser.add_argument('--algorithms', nargs='+',
                       default=['dqn', 'ddpg'],
                       help='사용할 알고리즘 목록')
    parser.add_argument('--max-episodes', type=int, default=1000,
                       help='최대 에피소드 수')
    parser.add_argument('--config-env', default='production',
                       choices=['development', 'production', 'testing'],
                       help='설정 환경')
    
    args = parser.parse_args()
    
    # 훈련 매니저 생성
    trainer = ComprehensiveTrainingManager(config_environment=args.config_env)
    
    # 종합 훈련 실행
    results = trainer.run_comprehensive_training(
        environments=args.environments,
        algorithms=args.algorithms,
        max_episodes=args.max_episodes
    )
    
    # 결과 요약 출력
    print("\n=== 훈련 결과 요약 ===")
    for env_name, env_results in results.items():
        print(f"\n환경: {env_name}")
        for algorithm, result in env_results.items():
            if 'error' in result:
                print(f"  {algorithm}: 실패 - {result['error']}")
            else:
                final_perf = result.get('final_performance', {})
                print(f"  {algorithm}: 성공 - 평균 보상: {final_perf.get('avg_reward_last_100', 0):.2f}")


if __name__ == "__main__":
    main()