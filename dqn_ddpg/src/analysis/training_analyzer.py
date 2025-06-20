"""
학습 결과 분석 및 비교 도구
학습 데이터를 분석하고 상세한 리포트를 생성합니다.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import scipy.stats as stats
from datetime import datetime

from .model_benchmarks import ModelBenchmarkSystem, BenchmarkResult
from ..core.episode_checkpoint import EpisodeCheckpointManager


@dataclass
class AlgorithmComparison:
    """알고리즘 간 비교 결과"""
    dqn_summary: Dict[str, Any]
    ddpg_summary: Dict[str, Any]
    
    # 성능 비교
    performance_winner: str
    performance_difference: float
    statistical_significance: bool
    
    # 효율성 비교
    efficiency_winner: str
    convergence_comparison: Dict[str, Any]
    
    # 안정성 비교
    stability_winner: str
    stability_analysis: Dict[str, Any]
    
    # 권장사항
    recommendation: str
    recommendation_reason: str


class TrainingAnalyzer:
    """학습 결과 분석 시스템
    
    학습 데이터를 로드하여 상세한 분석과 시각화를 제공합니다.
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 output_dir: str = "output/analysis"):
        """
        Args:
            results_dir: 결과 데이터 디렉토리
            output_dir: 분석 결과 출력 디렉토리
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print(f"📊 학습 분석 시스템 초기화")
        print(f"📁 결과 디렉토리: {self.results_dir}")
        print(f"📁 출력 디렉토리: {self.output_dir}")
    
    def analyze_algorithm_training(self, algorithm: str) -> Optional[Dict[str, Any]]:
        """특정 알고리즘의 학습 결과 분석
        
        Args:
            algorithm: 분석할 알고리즘 ('dqn' 또는 'ddpg')
            
        Returns:
            분석 결과 딕셔너리
        """
        print(f"\n🔍 {algorithm.upper()} 학습 분석 시작...")
        
        # 환경 이름 결정 (알고리즘별 기본값)
        default_envs = {
            'dqn': 'cartpole-v1',
            'ddpg': 'pendulum-v1'
        }
        env_name = default_envs.get(algorithm, 'unknown')
        
        # 체크포인트 매니저 로드
        checkpoint_manager = EpisodeCheckpointManager(
            base_path="models",
            algorithm=algorithm,
            environment=env_name
        )
        
        if not checkpoint_manager.training_history:
            print(f"⚠️ {algorithm} 학습 데이터를 찾을 수 없습니다")
            return None
        
        # 학습 요약 정보
        summary = checkpoint_manager.get_training_summary()
        
        # 학습 곡선 분석
        learning_analysis = self._analyze_learning_curves(checkpoint_manager.training_history)
        
        # 수렴 분석
        convergence_analysis = self._analyze_convergence(checkpoint_manager.training_history)
        
        # 성능 트렌드 분석
        trend_analysis = self._analyze_performance_trends(checkpoint_manager.training_history)
        
        # 벤치마크 결과 로드 (있는 경우)
        benchmark_analysis = None
        benchmark_system = ModelBenchmarkSystem(checkpoint_manager)
        if benchmark_system.load_benchmark_results():
            benchmark_analysis = self._analyze_benchmark_results(benchmark_system.benchmark_results)
        
        analysis_result = {
            'algorithm': algorithm,
            'analysis_time': datetime.now().isoformat(),
            'training_summary': summary,
            'learning_curves': learning_analysis,
            'convergence': convergence_analysis,
            'performance_trends': trend_analysis,
            'benchmark_analysis': benchmark_analysis
        }
        
        # 결과 저장
        self._save_analysis_result(algorithm, analysis_result)
        
        # 시각화 생성
        self._create_algorithm_visualizations(algorithm, checkpoint_manager.training_history, analysis_result)
        
        print(f"✅ {algorithm.upper()} 분석 완료")
        return analysis_result
    
    def compare_algorithms(self, 
                          dqn_analysis: Dict[str, Any] = None, 
                          ddpg_analysis: Dict[str, Any] = None) -> AlgorithmComparison:
        """DQN과 DDPG 알고리즘 비교
        
        Args:
            dqn_analysis: DQN 분석 결과 (None이면 자동 로드)
            ddpg_analysis: DDPG 분석 결과 (None이면 자동 로드)
            
        Returns:
            알고리즘 비교 결과
        """
        print("\n🔄 DQN vs DDPG 비교 분석 시작...")
        
        # 분석 결과 로드
        if dqn_analysis is None:
            dqn_analysis = self.analyze_algorithm_training('dqn')
        if ddpg_analysis is None:
            ddpg_analysis = self.analyze_algorithm_training('ddpg')
        
        if not dqn_analysis or not ddpg_analysis:
            print("⚠️ 두 알고리즘의 분석 결과가 모두 필요합니다")
            return None
        
        # 성능 비교
        dqn_reward = dqn_analysis['training_summary']['reward_stats']['latest_100_mean']
        ddpg_reward = ddpg_analysis['training_summary']['reward_stats']['latest_100_mean']
        
        performance_diff = dqn_reward - ddpg_reward
        performance_winner = 'DQN' if performance_diff > 0 else 'DDPG'
        
        # 통계적 유의성 검정 (간단한 근사)
        dqn_std = dqn_analysis['training_summary']['reward_stats']['std']
        ddpg_std = ddpg_analysis['training_summary']['reward_stats']['std']
        pooled_std = np.sqrt((dqn_std**2 + ddpg_std**2) / 2)
        t_stat = abs(performance_diff) / (pooled_std / np.sqrt(100))  # 최근 100에피소드 기준
        p_value = 2 * (1 - stats.t.cdf(t_stat, 198))  # df = 100 + 100 - 2
        significant = p_value < 0.05
        
        # 효율성 비교 (수렴 속도)
        dqn_convergence = dqn_analysis['convergence']['estimated_convergence_episode']
        ddpg_convergence = ddpg_analysis['convergence']['estimated_convergence_episode']
        
        if dqn_convergence and ddpg_convergence:
            efficiency_winner = 'DQN' if dqn_convergence < ddpg_convergence else 'DDPG'
        else:
            efficiency_winner = 'Unknown'
        
        convergence_comp = {
            'dqn_convergence_episode': dqn_convergence,
            'ddpg_convergence_episode': ddpg_convergence,
            'convergence_difference': (ddpg_convergence - dqn_convergence) if dqn_convergence and ddpg_convergence else None
        }
        
        # 안정성 비교
        dqn_stability = dqn_analysis['learning_curves']['reward_stability_score']
        ddpg_stability = ddpg_analysis['learning_curves']['reward_stability_score']
        
        stability_winner = 'DQN' if dqn_stability > ddpg_stability else 'DDPG'
        stability_analysis = {
            'dqn_stability_score': dqn_stability,
            'ddpg_stability_score': ddpg_stability,
            'stability_difference': dqn_stability - ddpg_stability
        }
        
        # 권장사항 결정
        factors = []
        dqn_score = 0
        ddpg_score = 0
        
        # 성능 점수
        if significant:
            if performance_diff > 0:
                dqn_score += 3
                factors.append("DQN이 통계적으로 유의한 성능 우위")
            else:
                ddpg_score += 3
                factors.append("DDPG가 통계적으로 유의한 성능 우위")
        
        # 효율성 점수
        if efficiency_winner == 'DQN':
            dqn_score += 2
            factors.append("DQN이 더 빠른 수렴")
        elif efficiency_winner == 'DDPG':
            ddpg_score += 2
            factors.append("DDPG가 더 빠른 수렴")
        
        # 안정성 점수
        if stability_winner == 'DQN':
            dqn_score += 1
            factors.append("DQN이 더 안정적")
        else:
            ddpg_score += 1
            factors.append("DDPG가 더 안정적")
        
        if dqn_score > ddpg_score:
            recommendation = 'DQN'
            reason = f"종합 점수 {dqn_score}:{ddpg_score}. " + "; ".join(factors)
        elif ddpg_score > dqn_score:
            recommendation = 'DDPG'
            reason = f"종합 점수 {ddpg_score}:{dqn_score}. " + "; ".join(factors)
        else:
            recommendation = 'Both'
            reason = "두 알고리즘이 비슷한 성능을 보임"
        
        comparison = AlgorithmComparison(
            dqn_summary=dqn_analysis['training_summary'],
            ddpg_summary=ddpg_analysis['training_summary'],
            performance_winner=performance_winner,
            performance_difference=performance_diff,
            statistical_significance=significant,
            efficiency_winner=efficiency_winner,
            convergence_comparison=convergence_comp,
            stability_winner=stability_winner,
            stability_analysis=stability_analysis,
            recommendation=recommendation,
            recommendation_reason=reason
        )
        
        # 비교 결과 저장
        self._save_comparison_result(comparison)
        
        # 비교 시각화 생성
        self._create_comparison_visualizations(dqn_analysis, ddpg_analysis, comparison)
        
        print("✅ 알고리즘 비교 분석 완료")
        return comparison
    
    def _analyze_learning_curves(self, training_history: List) -> Dict[str, Any]:
        """학습 곡선 분석"""
        if not training_history:
            return {}
        
        rewards = [episode.total_reward for episode in training_history]
        lengths = [episode.episode_length for episode in training_history]
        losses = [episode.average_loss for episode in training_history if episode.average_loss > 0]
        
        # 이동 평균 계산
        window_size = min(50, len(rewards) // 10)
        if window_size > 1:
            reward_ma = pd.Series(rewards).rolling(window=window_size).mean().tolist()
        else:
            reward_ma = rewards
        
        # 학습 안정성 점수 (최근 구간의 분산 기반)
        recent_rewards = rewards[-min(100, len(rewards)):]
        stability_score = 1.0 / (1.0 + np.var(recent_rewards)) if recent_rewards else 0
        
        # 개선 추세 분석
        if len(rewards) >= 100:
            early_mean = np.mean(rewards[:50])
            late_mean = np.mean(rewards[-50:])
            improvement_rate = (late_mean - early_mean) / abs(early_mean) if early_mean != 0 else 0
        else:
            improvement_rate = 0
        
        return {
            'total_episodes': len(rewards),
            'reward_statistics': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            },
            'reward_moving_average': reward_ma,
            'reward_stability_score': stability_score,
            'improvement_rate': improvement_rate,
            'loss_statistics': {
                'mean': np.mean(losses) if losses else 0,
                'std': np.std(losses) if losses else 0,
                'count': len(losses)
            } if losses else None
        }
    
    def _analyze_convergence(self, training_history: List) -> Dict[str, Any]:
        """수렴 분석"""
        if not training_history:
            return {}
        
        rewards = [episode.total_reward for episode in training_history]
        
        # 수렴 감지
        convergence_episode = None
        stability_threshold = 0.1  # 변동 계수 임계값
        window_size = 50
        
        for i in range(window_size, len(rewards)):
            window_rewards = rewards[i-window_size:i]
            mean_reward = np.mean(window_rewards)
            std_reward = np.std(window_rewards)
            
            if mean_reward != 0:
                cv = std_reward / abs(mean_reward)
                if cv < stability_threshold:
                    convergence_episode = i
                    break
        
        # 학습 단계 분석
        total_episodes = len(rewards)
        if total_episodes >= 100:
            exploration_phase = rewards[:total_episodes//3]
            learning_phase = rewards[total_episodes//3:2*total_episodes//3]
            convergence_phase = rewards[2*total_episodes//3:]
            
            phase_analysis = {
                'exploration_mean': np.mean(exploration_phase),
                'learning_mean': np.mean(learning_phase),
                'convergence_mean': np.mean(convergence_phase),
                'learning_improvement': np.mean(learning_phase) - np.mean(exploration_phase),
                'convergence_improvement': np.mean(convergence_phase) - np.mean(learning_phase)
            }
        else:
            phase_analysis = None
        
        return {
            'estimated_convergence_episode': convergence_episode,
            'convergence_achieved': convergence_episode is not None,
            'convergence_percentage': (convergence_episode / total_episodes * 100) if convergence_episode else None,
            'phase_analysis': phase_analysis,
            'final_stability': self._calculate_stability(rewards[-min(100, len(rewards)):])
        }
    
    def _analyze_performance_trends(self, training_history: List) -> Dict[str, Any]:
        """성능 트렌드 분석"""
        if not training_history:
            return {}
        
        rewards = [episode.total_reward for episode in training_history]
        episode_lengths = [episode.episode_length for episode in training_history]
        exploration_rates = [episode.exploration_rate for episode in training_history]
        
        # 선형 회귀를 통한 트렌드 분석
        episodes = np.arange(len(rewards))
        
        # 보상 트렌드
        reward_slope, reward_intercept, reward_r, _, _ = stats.linregress(episodes, rewards)
        
        # 에피소드 길이 트렌드
        length_slope, length_intercept, length_r, _, _ = stats.linregress(episodes, episode_lengths)
        
        # 성능 지표
        efficiency_trend = []
        for i in range(len(rewards)):
            if episode_lengths[i] > 0:
                efficiency_trend.append(rewards[i] / episode_lengths[i])
            else:
                efficiency_trend.append(0)
        
        efficiency_slope, _, efficiency_r, _, _ = stats.linregress(episodes, efficiency_trend)
        
        return {
            'reward_trend': {
                'slope': reward_slope,
                'correlation': reward_r,
                'trend_direction': 'improving' if reward_slope > 0 else 'declining',
                'trend_strength': abs(reward_r)
            },
            'episode_length_trend': {
                'slope': length_slope,
                'correlation': length_r,
                'trend_direction': 'increasing' if length_slope > 0 else 'decreasing'
            },
            'efficiency_trend': {
                'slope': efficiency_slope,
                'correlation': efficiency_r,
                'trend_direction': 'improving' if efficiency_slope > 0 else 'declining'
            },
            'exploration_decay': {
                'initial_rate': exploration_rates[0] if exploration_rates else 0,
                'final_rate': exploration_rates[-1] if exploration_rates else 0,
                'decay_rate': (exploration_rates[0] - exploration_rates[-1]) / len(exploration_rates) if exploration_rates else 0
            }
        }
    
    def _analyze_benchmark_results(self, benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """벤치마크 결과 분석"""
        if not benchmark_results:
            return {}
        
        rewards = [r.mean_reward for r in benchmark_results]
        success_rates = [r.success_rate for r in benchmark_results]
        performance_scores = [r.performance_score for r in benchmark_results]
        
        # 최고 성능 모델들
        top_models = sorted(benchmark_results, key=lambda r: r.performance_score, reverse=True)[:5]
        
        return {
            'total_benchmarked_models': len(benchmark_results),
            'performance_statistics': {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'max_reward': np.max(rewards),
                'mean_success_rate': np.mean(success_rates),
                'max_success_rate': np.max(success_rates)
            },
            'top_models': [
                {
                    'episode_id': model.episode_id,
                    'mean_reward': model.mean_reward,
                    'success_rate': model.success_rate,
                    'performance_score': model.performance_score
                }
                for model in top_models
            ],
            'model_consistency': {
                'reward_coefficient_of_variation': np.std(rewards) / np.mean(rewards) if np.mean(rewards) != 0 else float('inf'),
                'performance_range': np.max(performance_scores) - np.min(performance_scores)
            }
        }
    
    def _calculate_stability(self, values: List[float]) -> float:
        """안정성 점수 계산"""
        if not values or len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        # 변동 계수의 역수로 안정성 계산
        cv = std_val / abs(mean_val)
        stability = 1.0 / (1.0 + cv)
        
        return stability
    
    def _save_analysis_result(self, algorithm: str, analysis: Dict[str, Any]):
        """분석 결과 저장"""
        output_file = self.output_dir / f"{algorithm}_analysis.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"💾 {algorithm.upper()} 분석 결과 저장: {output_file}")
    
    def _save_comparison_result(self, comparison: AlgorithmComparison):
        """비교 결과 저장"""
        output_file = self.output_dir / "algorithm_comparison.json"
        
        # dataclass를 딕셔너리로 변환
        comparison_dict = {
            'dqn_summary': comparison.dqn_summary,
            'ddpg_summary': comparison.ddpg_summary,
            'performance_winner': comparison.performance_winner,
            'performance_difference': comparison.performance_difference,
            'statistical_significance': comparison.statistical_significance,
            'efficiency_winner': comparison.efficiency_winner,
            'convergence_comparison': comparison.convergence_comparison,
            'stability_winner': comparison.stability_winner,
            'stability_analysis': comparison.stability_analysis,
            'recommendation': comparison.recommendation,
            'recommendation_reason': comparison.recommendation_reason,
            'analysis_time': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison_dict, f, indent=2, default=str)
        
        print(f"💾 알고리즘 비교 결과 저장: {output_file}")
    
    def _create_algorithm_visualizations(self, algorithm: str, training_history: List, analysis: Dict[str, Any]):
        """알고리즘별 시각화 생성"""
        if not training_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algorithm.upper()} 학습 분석', fontsize=16)
        
        # 데이터 준비
        episodes = list(range(len(training_history)))
        rewards = [ep.total_reward for ep in training_history]
        lengths = [ep.episode_length for ep in training_history]
        losses = [ep.average_loss for ep in training_history if ep.average_loss > 0]
        exploration_rates = [ep.exploration_rate for ep in training_history]
        
        # 1. 보상 곡선
        axes[0, 0].plot(episodes, rewards, alpha=0.6, label='Episode Reward')
        if len(rewards) > 10:
            ma = pd.Series(rewards).rolling(window=min(50, len(rewards)//10)).mean()
            axes[0, 0].plot(episodes, ma, 'r-', linewidth=2, label='Moving Average')
        axes[0, 0].set_title('Reward Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 에피소드 길이
        axes[0, 1].plot(episodes, lengths, 'g-', alpha=0.7)
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # 3. 손실 (있는 경우)
        if losses:
            loss_episodes = [i for i, ep in enumerate(training_history) if ep.average_loss > 0]
            axes[1, 0].plot(loss_episodes, losses, 'orange', alpha=0.7)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Loss (No Data)')
        
        # 4. 탐험율
        axes[1, 1].plot(episodes, exploration_rates, 'purple', alpha=0.7)
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Exploration Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_file = self.output_dir / f"{algorithm}_training_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 {algorithm.upper()} 시각화 저장: {output_file}")
    
    def _create_comparison_visualizations(self, dqn_analysis: Dict, ddpg_analysis: Dict, comparison: AlgorithmComparison):
        """비교 시각화 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN vs DDPG 비교 분석', fontsize=16)
        
        # 1. 성능 비교 바 차트
        algorithms = ['DQN', 'DDPG']
        rewards = [
            dqn_analysis['training_summary']['reward_stats']['latest_100_mean'],
            ddpg_analysis['training_summary']['reward_stats']['latest_100_mean']
        ]
        
        bars = axes[0, 0].bar(algorithms, rewards, color=['blue', 'red'], alpha=0.7)
        axes[0, 0].set_title('최종 성능 비교 (최근 100에피소드 평균)')
        axes[0, 0].set_ylabel('Average Reward')
        
        # 성능 차이 표시
        for i, (bar, reward) in enumerate(zip(bars, rewards)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{reward:.2f}', ha='center', va='bottom')
        
        # 2. 수렴 비교
        convergence_data = comparison.convergence_comparison
        if convergence_data['dqn_convergence_episode'] and convergence_data['ddpg_convergence_episode']:
            conv_episodes = [convergence_data['dqn_convergence_episode'], convergence_data['ddpg_convergence_episode']]
            bars = axes[0, 1].bar(algorithms, conv_episodes, color=['blue', 'red'], alpha=0.7)
            axes[0, 1].set_title('수렴 속도 비교')
            axes[0, 1].set_ylabel('Convergence Episode')
            
            for i, (bar, episode) in enumerate(zip(bars, conv_episodes)):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{episode}', ha='center', va='bottom')
        else:
            axes[0, 1].text(0.5, 0.5, 'Convergence data not available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('수렴 속도 비교 (데이터 없음)')
        
        # 3. 안정성 비교
        stability_scores = [
            comparison.stability_analysis['dqn_stability_score'],
            comparison.stability_analysis['ddpg_stability_score']
        ]
        
        bars = axes[1, 0].bar(algorithms, stability_scores, color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_title('학습 안정성 비교')
        axes[1, 0].set_ylabel('Stability Score')
        
        for i, (bar, score) in enumerate(zip(bars, stability_scores)):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 종합 점수
        axes[1, 1].text(0.5, 0.7, f"권장 알고리즘: {comparison.recommendation}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold')
        axes[1, 1].text(0.5, 0.5, f"성능 우위: {comparison.performance_winner}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.5, 0.3, f"효율성 우위: {comparison.efficiency_winner}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.5, 0.1, f"안정성 우위: {comparison.stability_winner}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('종합 비교 결과')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        
        # 저장
        output_file = self.output_dir / "algorithm_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 비교 시각화 저장: {output_file}")
    
    def generate_comprehensive_report(self) -> str:
        """종합 리포트 생성"""
        print("\n📋 종합 리포트 생성 중...")
        
        # 알고리즘별 분석
        dqn_analysis = self.analyze_algorithm_training('dqn')
        ddpg_analysis = self.analyze_algorithm_training('ddpg')
        
        # 비교 분석
        comparison = None
        if dqn_analysis and ddpg_analysis:
            comparison = self.compare_algorithms(dqn_analysis, ddpg_analysis)
        
        # HTML 리포트 생성
        html_content = self._generate_html_report(dqn_analysis, ddpg_analysis, comparison)
        
        report_file = self.output_dir / "comprehensive_training_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 종합 리포트 생성 완료: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, dqn_analysis: Dict, ddpg_analysis: Dict, comparison: AlgorithmComparison) -> str:
        """HTML 리포트 생성"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DQN vs DDPG 학습 분석 리포트</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; color: #333; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; }
                .winner { color: #28a745; font-weight: bold; }
                .loser { color: #dc3545; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DQN vs DDPG 학습 분석 리포트</h1>
                <p>생성 시간: {}</p>
            </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # DQN 분석 섹션
        if dqn_analysis:
            html += self._generate_algorithm_section("DQN", dqn_analysis)
        
        # DDPG 분석 섹션
        if ddpg_analysis:
            html += self._generate_algorithm_section("DDPG", ddpg_analysis)
        
        # 비교 섹션
        if comparison:
            html += self._generate_comparison_section(comparison)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_algorithm_section(self, algorithm: str, analysis: Dict) -> str:
        """알고리즘 섹션 HTML 생성"""
        summary = analysis['training_summary']
        
        html = f"""
        <div class="section">
            <h2>{algorithm} 분석 결과</h2>
            <div class="metric">총 에피소드: {summary['total_episodes']}</div>
            <div class="metric">최종 평균 보상: {summary['reward_stats']['latest_100_mean']:.2f}</div>
            <div class="metric">최고 보상: {summary['reward_stats']['max']:.2f}</div>
            <div class="metric">보상 표준편차: {summary['reward_stats']['std']:.2f}</div>
        """
        
        if 'convergence' in analysis and analysis['convergence']['convergence_achieved']:
            conv_episode = analysis['convergence']['estimated_convergence_episode']
            html += f'<div class="metric">수렴 에피소드: {conv_episode}</div>'
        
        html += "</div>"
        
        return html
    
    def _generate_comparison_section(self, comparison: AlgorithmComparison) -> str:
        """비교 섹션 HTML 생성"""
        html = f"""
        <div class="section">
            <h2>알고리즘 비교 결과</h2>
            <table>
                <tr>
                    <th>항목</th>
                    <th>우위 알고리즘</th>
                    <th>설명</th>
                </tr>
                <tr>
                    <td>성능</td>
                    <td class="winner">{comparison.performance_winner}</td>
                    <td>차이: {comparison.performance_difference:.2f} (p={comparison.statistical_significance})</td>
                </tr>
                <tr>
                    <td>효율성</td>
                    <td class="winner">{comparison.efficiency_winner}</td>
                    <td>수렴 속도 기준</td>
                </tr>
                <tr>
                    <td>안정성</td>
                    <td class="winner">{comparison.stability_winner}</td>
                    <td>학습 곡선 안정성 기준</td>
                </tr>
            </table>
            
            <h3>최종 권장사항</h3>
            <p><strong>{comparison.recommendation}</strong></p>
            <p>{comparison.recommendation_reason}</p>
        </div>
        """
        
        return html