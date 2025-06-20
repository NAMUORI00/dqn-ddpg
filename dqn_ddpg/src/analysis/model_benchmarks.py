"""
모델 성능 벤치마킹 시스템
저장된 모델들의 성능을 평가하고 비교 분석을 수행합니다.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import gymnasium as gym
from collections import defaultdict

from ..core.episode_checkpoint import EpisodeCheckpointManager
from ..environments.wrappers import create_dqn_env, create_ddpg_env
from ..agents import DQNAgent, DDPGAgent
from ..core.utils import get_device, set_seed


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    model_path: str
    episode_id: int
    algorithm: str
    
    # 성능 지표
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    success_rate: float
    
    # 에피소드 통계
    mean_episode_length: float
    std_episode_length: float
    
    # 효율성 지표
    reward_per_step: float
    convergence_time: float
    
    # 평가 메타데이터
    num_eval_episodes: int
    evaluation_time: float
    timestamp: float
    
    # 추가 메트릭
    extra_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.extra_metrics is None:
            self.extra_metrics = {}
    
    @property
    def performance_score(self) -> float:
        """종합 성능 점수 계산"""
        # 보상과 안정성을 고려한 점수
        stability_penalty = self.std_reward / max(abs(self.mean_reward), 1e-6)
        return self.mean_reward - (stability_penalty * 0.1)


@dataclass
class ComparisonResult:
    """모델 비교 결과"""
    model_a: BenchmarkResult
    model_b: BenchmarkResult
    
    # 통계적 비교
    reward_difference: float
    statistical_significance: bool
    p_value: float
    
    # 효율성 비교
    efficiency_ratio: float
    stability_comparison: str  # 'better', 'worse', 'similar'
    
    # 권장사항
    recommended_model: str
    recommendation_reason: str


class ModelBenchmarkSystem:
    """모델 성능 벤치마킹 시스템
    
    저장된 모델들을 로드하여 표준화된 환경에서 성능을 평가합니다.
    """
    
    def __init__(self, 
                 checkpoint_manager: EpisodeCheckpointManager,
                 num_eval_episodes: int = 50,
                 eval_deterministic: bool = True,
                 seed: int = 42):
        """
        Args:
            checkpoint_manager: 체크포인트 매니저
            num_eval_episodes: 평가 에피소드 수
            eval_deterministic: 결정론적 평가 여부
            seed: 평가용 시드
        """
        self.checkpoint_manager = checkpoint_manager
        self.num_eval_episodes = num_eval_episodes
        self.eval_deterministic = eval_deterministic
        self.seed = seed
        self.algorithm = checkpoint_manager.algorithm
        
        # 결과 저장
        self.benchmark_results: List[BenchmarkResult] = []
        self.comparison_cache: Dict[Tuple[int, int], ComparisonResult] = {}
        
        # 결과 디렉토리
        self.results_dir = Path("results") / "benchmarks" / self.algorithm
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 모델 벤치마크 시스템 초기화: {self.algorithm}")
        print(f"📊 평가 에피소드: {num_eval_episodes}")
        print(f"📁 결과 저장 경로: {self.results_dir}")
    
    def evaluate_model(self, 
                      episode_id: int,
                      config: Dict[str, Any],
                      verbose: bool = True) -> Optional[BenchmarkResult]:
        """특정 에피소드의 모델 평가
        
        Args:
            episode_id: 평가할 에피소드 ID
            config: 모델 설정
            verbose: 상세 출력 여부
            
        Returns:
            벤치마크 결과 (모델이 없으면 None)
        """
        # 체크포인트 로드
        checkpoint_data = self.checkpoint_manager.load_checkpoint(episode_id=episode_id)
        if checkpoint_data is None:
            if verbose:
                print(f"⚠️ Episode {episode_id} 체크포인트를 찾을 수 없습니다")
            return None
        
        if verbose:
            print(f"🔍 Episode {episode_id} 모델 평가 시작...")
        
        # 환경 생성
        set_seed(self.seed)
        env = self._create_evaluation_environment(config)
        
        # 에이전트 생성 및 로드
        agent = self._create_agent(env, config)
        agent.load_state_dict(checkpoint_data['model_state_dict'])
        agent.eval() if hasattr(agent, 'eval') else None
        
        # 성능 평가
        eval_start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for eval_episode in range(self.num_eval_episodes):
            set_seed(self.seed + eval_episode)  # 재현 가능한 평가
            
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # 결정론적 행동 선택
                if self.algorithm == 'dqn':
                    action = agent.select_action(state, deterministic=self.eval_deterministic)
                else:  # ddpg
                    action = agent.select_action(state, add_noise=not self.eval_deterministic)
                
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # 성공 기준 (환경에 따라 조정 가능)
            if total_reward > 0:  # 간단한 성공 기준
                success_count += 1
        
        env.close()
        evaluation_time = time.time() - eval_start_time
        
        # 통계 계산
        rewards_array = np.array(episode_rewards)
        lengths_array = np.array(episode_lengths)
        
        mean_reward = float(np.mean(rewards_array))
        std_reward = float(np.std(rewards_array))
        min_reward = float(np.min(rewards_array))
        max_reward = float(np.max(rewards_array))
        success_rate = success_count / self.num_eval_episodes
        
        mean_length = float(np.mean(lengths_array))
        std_length = float(np.std(lengths_array))
        
        reward_per_step = mean_reward / mean_length if mean_length > 0 else 0
        
        # 수렴 시간 추정 (체크포인트 메타데이터에서)
        convergence_time = checkpoint_data.get('metrics', {}).get('training_time', 0.0)
        
        # 벤치마크 결과 생성
        result = BenchmarkResult(
            model_path=self.checkpoint_manager.model_dir / f"episode_{episode_id:04d}.pth",
            episode_id=episode_id,
            algorithm=self.algorithm,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            success_rate=success_rate,
            mean_episode_length=mean_length,
            std_episode_length=std_length,
            reward_per_step=reward_per_step,
            convergence_time=convergence_time,
            num_eval_episodes=self.num_eval_episodes,
            evaluation_time=evaluation_time,
            timestamp=time.time(),
            extra_metrics={
                'reward_range': max_reward - min_reward,
                'coefficient_of_variation': std_reward / abs(mean_reward) if mean_reward != 0 else float('inf'),
                'episodes_per_second': self.num_eval_episodes / evaluation_time
            }
        )
        
        self.benchmark_results.append(result)
        
        if verbose:
            print(f"  ✓ 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  ✓ 성공률: {success_rate:.1%}")
            print(f"  ✓ 평가 시간: {evaluation_time:.2f}초")
        
        return result
    
    def evaluate_all_checkpoints(self, config: Dict[str, Any], max_models: int = None) -> List[BenchmarkResult]:
        """모든 체크포인트 평가
        
        Args:
            config: 모델 설정
            max_models: 최대 평가 모델 수 (None이면 전체)
            
        Returns:
            벤치마크 결과 리스트
        """
        checkpoints = self.checkpoint_manager.checkpoints
        if not checkpoints:
            print("⚠️ 평가할 체크포인트가 없습니다")
            return []
        
        # 최신 순으로 정렬
        checkpoints_sorted = sorted(checkpoints, key=lambda cp: cp.episode_id, reverse=True)
        
        if max_models:
            checkpoints_sorted = checkpoints_sorted[:max_models]
        
        print(f"🔬 {len(checkpoints_sorted)}개 모델 평가 시작...")
        
        results = []
        for i, checkpoint in enumerate(checkpoints_sorted, 1):
            print(f"\n[{i}/{len(checkpoints_sorted)}] Episode {checkpoint.episode_id} 평가 중...")
            
            result = self.evaluate_model(checkpoint.episode_id, config, verbose=True)
            if result:
                results.append(result)
        
        # 결과 저장
        self._save_benchmark_results()
        
        print(f"\n✅ 벤치마크 완료: {len(results)}개 모델 평가됨")
        return results
    
    def compare_models(self, episode_a: int, episode_b: int) -> Optional[ComparisonResult]:
        """두 모델 비교
        
        Args:
            episode_a: 첫 번째 모델 에피소드 ID
            episode_b: 두 번째 모델 에피소드 ID
            
        Returns:
            비교 결과
        """
        # 캐시 확인
        cache_key = (min(episode_a, episode_b), max(episode_a, episode_b))
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        # 벤치마크 결과 찾기
        result_a = None
        result_b = None
        
        for result in self.benchmark_results:
            if result.episode_id == episode_a:
                result_a = result
            elif result.episode_id == episode_b:
                result_b = result
        
        if not result_a or not result_b:
            print(f"⚠️ 비교할 모델의 벤치마크 결과를 찾을 수 없습니다")
            return None
        
        # 통계적 유의성 검정 (간단한 t-test 근사)
        reward_diff = result_a.mean_reward - result_b.mean_reward
        pooled_std = np.sqrt((result_a.std_reward**2 + result_b.std_reward**2) / 2)
        t_stat = abs(reward_diff) / (pooled_std / np.sqrt(self.num_eval_episodes))
        p_value = 2 * (1 - self._t_cdf(t_stat, self.num_eval_episodes - 1))
        significant = p_value < 0.05
        
        # 효율성 비교
        efficiency_a = result_a.reward_per_step
        efficiency_b = result_b.reward_per_step
        efficiency_ratio = efficiency_a / efficiency_b if efficiency_b != 0 else float('inf')
        
        # 안정성 비교
        cv_a = result_a.extra_metrics.get('coefficient_of_variation', float('inf'))
        cv_b = result_b.extra_metrics.get('coefficient_of_variation', float('inf'))
        
        if cv_a < cv_b * 0.9:
            stability_comp = 'better'
        elif cv_a > cv_b * 1.1:
            stability_comp = 'worse'
        else:
            stability_comp = 'similar'
        
        # 권장 모델 결정
        if significant and reward_diff > 0:
            recommended = f"episode_{episode_a}"
            reason = f"통계적으로 유의한 성능 향상 ({reward_diff:.2f})"
        elif significant and reward_diff < 0:
            recommended = f"episode_{episode_b}"
            reason = f"통계적으로 유의한 성능 향상 ({-reward_diff:.2f})"
        elif result_a.performance_score > result_b.performance_score:
            recommended = f"episode_{episode_a}"
            reason = "종합 성능 점수가 높음"
        else:
            recommended = f"episode_{episode_b}"
            reason = "종합 성능 점수가 높음"
        
        comparison = ComparisonResult(
            model_a=result_a,
            model_b=result_b,
            reward_difference=reward_diff,
            statistical_significance=significant,
            p_value=p_value,
            efficiency_ratio=efficiency_ratio,
            stability_comparison=stability_comp,
            recommended_model=recommended,
            recommendation_reason=reason
        )
        
        # 캐시에 저장
        self.comparison_cache[cache_key] = comparison
        
        return comparison
    
    def get_top_models(self, 
                      criterion: str = 'performance_score', 
                      top_k: int = 10) -> List[BenchmarkResult]:
        """상위 성능 모델들 반환
        
        Args:
            criterion: 정렬 기준 ('mean_reward', 'performance_score', 'success_rate')
            top_k: 반환할 모델 수
            
        Returns:
            상위 모델 리스트
        """
        if not self.benchmark_results:
            return []
        
        if criterion == 'mean_reward':
            key_func = lambda r: r.mean_reward
        elif criterion == 'performance_score':
            key_func = lambda r: r.performance_score
        elif criterion == 'success_rate':
            key_func = lambda r: r.success_rate
        elif criterion == 'reward_per_step':
            key_func = lambda r: r.reward_per_step
        else:
            key_func = lambda r: r.mean_reward
        
        sorted_results = sorted(self.benchmark_results, key=key_func, reverse=True)
        return sorted_results[:top_k]
    
    def _create_evaluation_environment(self, config: Dict[str, Any]):
        """평가용 환경 생성"""
        env_name = config['environment']['name']
        
        if self.algorithm == 'dqn':
            return create_dqn_env(env_name)
        else:  # ddpg
            return create_ddpg_env(env_name)
    
    def _create_agent(self, env, config: Dict[str, Any]):
        """평가용 에이전트 생성"""
        state_dim = env.observation_space.shape[0]
        
        if self.algorithm == 'dqn':
            action_dim = env.action_space.n
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config['agent']['learning_rate'],
                gamma=config['agent']['gamma'],
                device=get_device()
            )
        else:  # ddpg
            action_dim = env.action_space.shape[0]
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_lr=config['agent']['actor_lr'],
                critic_lr=config['agent']['critic_lr'],
                gamma=config['agent']['gamma'],
                device=get_device()
            )
        
        return agent
    
    def _t_cdf(self, t: float, df: int) -> float:
        """t-분포 누적분포함수 근사"""
        # 간단한 근사 (정확한 구현은 scipy.stats.t.cdf 사용)
        return 0.5 + 0.5 * np.tanh(t / np.sqrt(df))
    
    def _save_benchmark_results(self):
        """벤치마크 결과 저장"""
        results_file = self.results_dir / f"{self.algorithm}_benchmark_results.json"
        
        data = {
            'algorithm': self.algorithm,
            'num_eval_episodes': self.num_eval_episodes,
            'eval_deterministic': self.eval_deterministic,
            'seed': self.seed,
            'last_updated': time.time(),
            'results': [asdict(result) for result in self.benchmark_results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 벤치마크 결과 저장: {results_file}")
    
    def load_benchmark_results(self) -> bool:
        """저장된 벤치마크 결과 로드"""
        results_file = self.results_dir / f"{self.algorithm}_benchmark_results.json"
        
        if not results_file.exists():
            return False
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            self.benchmark_results = []
            for result_data in data.get('results', []):
                result = BenchmarkResult(**result_data)
                self.benchmark_results.append(result)
            
            print(f"📂 벤치마크 결과 로드: {len(self.benchmark_results)}개 결과")
            return True
            
        except Exception as e:
            print(f"⚠️ 벤치마크 결과 로드 실패: {e}")
            return False
    
    def generate_benchmark_report(self) -> str:
        """벤치마크 리포트 생성"""
        if not self.benchmark_results:
            print("⚠️ 생성할 벤치마크 결과가 없습니다")
            return ""
        
        report_file = self.results_dir / f"{self.algorithm}_benchmark_report.json"
        
        # 통계 계산
        all_rewards = [r.mean_reward for r in self.benchmark_results]
        all_scores = [r.performance_score for r in self.benchmark_results]
        
        # 상위 모델들
        top_by_reward = self.get_top_models('mean_reward', 5)
        top_by_score = self.get_top_models('performance_score', 5)
        
        # 리포트 데이터
        report = {
            'algorithm': self.algorithm,
            'generation_time': time.time(),
            'summary': {
                'total_models_evaluated': len(self.benchmark_results),
                'reward_statistics': {
                    'mean': float(np.mean(all_rewards)),
                    'std': float(np.std(all_rewards)),
                    'min': float(np.min(all_rewards)),
                    'max': float(np.max(all_rewards))
                },
                'performance_score_statistics': {
                    'mean': float(np.mean(all_scores)),
                    'std': float(np.std(all_scores)),
                    'min': float(np.min(all_scores)),
                    'max': float(np.max(all_scores))
                }
            },
            'top_models': {
                'by_reward': [
                    {
                        'episode_id': r.episode_id,
                        'mean_reward': r.mean_reward,
                        'success_rate': r.success_rate
                    }
                    for r in top_by_reward
                ],
                'by_performance_score': [
                    {
                        'episode_id': r.episode_id,
                        'performance_score': r.performance_score,
                        'mean_reward': r.mean_reward
                    }
                    for r in top_by_score
                ]
            },
            'detailed_results': [asdict(result) for result in self.benchmark_results]
        }
        
        # 파일 저장
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 벤치마크 리포트 생성: {report_file}")
        return str(report_file)