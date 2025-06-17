"""
동일 환경 DQN vs DDPG 비교 실험

ContinuousCartPole 환경에서 DiscretizedDQN과 DDPG를 직접 비교합니다.
이를 통해 환경의 차이를 배제하고 순수한 알고리즘 특성만을 비교할 수 있습니다.

핵심 비교 포인트:
1. 학습 성능 및 수렴 속도
2. 결정적 정책의 구현 방식 차이
3. 탐험 전략의 효과성
4. 행동 선택의 일관성 및 안정성
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 새로운 시각화 모듈 import
try:
    from src.visualization.charts.learning_curves import LearningCurveVisualizer
    from src.visualization.charts.comparison import ComparisonChartVisualizer
    from src.visualization.charts.policy_analysis import PolicyAnalysisVisualizer
    from src.visualization.core.config import VisualizationConfig
    NEW_VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: 새로운 시각화 모듈을 가져올 수 없습니다. 기본 시각화 사용.")
    NEW_VISUALIZATION_AVAILABLE = False

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.continuous_cartpole import create_continuous_cartpole_env
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.core.utils import set_seed


class SameEnvironmentComparison:
    """동일 환경에서 DQN vs DDPG 비교 실험 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 실험 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.results = {
            'dqn': {'scores': [], 'losses': [], 'metrics': []},
            'ddpg': {'scores': [], 'losses': [], 'metrics': []}
        }
        
        # 환경 설정
        self.env = create_continuous_cartpole_env()
        self.state_dim = self.env.observation_space.shape[0]
        
        print("=== 동일 환경 DQN vs DDPG 비교 실험 ===")
        print(f"환경: ContinuousCartPole-v0")
        print(f"상태 차원: {self.state_dim}")
        print(f"행동 공간: {self.env.action_space}")
        print()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 실험 설정"""
        return {
            'max_episodes': 500,
            'max_steps_per_episode': 500,
            'evaluation_frequency': 50,
            'evaluation_episodes': 10,
            'success_threshold': 450.0,  # CartPole 기준
            'random_seed': 42,
            
            # DQN 설정
            'dqn': {
                'action_bound': 1.0,
                'num_actions': 21,
                'learning_rate': 1e-3,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'gamma': 0.99,
                'buffer_size': 50000,
                'batch_size': 64,
                'target_update_freq': 100
            },
            
            # DDPG 설정
            'ddpg': {
                'action_bound': 1.0,
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.005,
                'noise_sigma': 0.1,
                'noise_decay': 0.999,
                'buffer_size': 50000,
                'batch_size': 64
            }
        }
    
    def create_agents(self) -> Tuple[DiscretizedDQNAgent, DDPGAgent]:
        """에이전트 생성"""
        # DiscretizedDQN 에이전트
        dqn_agent = DiscretizedDQNAgent(
            state_dim=self.state_dim,
            **self.config['dqn']
        )
        
        # DDPG 에이전트
        ddpg_agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=1,  # 1차원 연속 행동
            **self.config['ddpg']
        )
        
        return dqn_agent, ddpg_agent
    
    def train_agent(self, agent, agent_name: str) -> Dict[str, List]:
        """개별 에이전트 훈련
        
        Args:
            agent: 훈련할 에이전트
            agent_name: 에이전트 이름 ('dqn' or 'ddpg')
            
        Returns:
            훈련 결과 딕셔너리
        """
        print(f"\\n=== {agent_name.upper()} 에이전트 훈련 시작 ===")
        
        scores = []
        losses = []
        metrics = []
        best_score = -np.inf
        
        for episode in range(self.config['max_episodes']):
            state, _ = self.env.reset()
            episode_score = 0
            episode_losses = []
            
            for step in range(self.config['max_steps_per_episode']):
                # 행동 선택
                if agent_name == 'dqn':
                    action = agent.select_action(state)
                else:  # ddpg
                    action = agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                agent.store_transition(state, action, reward, next_state, done)
                
                # 에이전트 업데이트
                if agent.buffer.is_ready(agent.batch_size):
                    loss_info = agent.update()
                    if loss_info:
                        episode_losses.append(loss_info)
                
                episode_score += reward
                state = next_state
                
                if done:
                    break
            
            scores.append(episode_score)
            if episode_losses:
                losses.append(np.mean([loss['loss'] if 'loss' in loss else 
                                     loss.get('critic_loss', 0) for loss in episode_losses]))
            
            # 주기적 평가
            if (episode + 1) % self.config['evaluation_frequency'] == 0:
                eval_score = self.evaluate_agent(agent, agent_name)
                metrics.append({
                    'episode': episode + 1,
                    'eval_score': eval_score,
                    'train_score': np.mean(scores[-50:]),
                    'epsilon': getattr(agent, 'epsilon', 0),
                    'noise_sigma': getattr(agent.noise, 'sigma', 0) if hasattr(agent, 'noise') else 0
                })
                
                if eval_score > best_score:
                    best_score = eval_score
                
                print(f"{agent_name.upper()} Episode {episode+1}: "
                      f"Train={np.mean(scores[-50:]):.1f}, "
                      f"Eval={eval_score:.1f}, "
                      f"Best={best_score:.1f}")
        
        return {
            'scores': scores,
            'losses': losses,
            'metrics': metrics,
            'best_score': best_score
        }
    
    def evaluate_agent(self, agent, agent_name: str, num_episodes: int = None) -> float:
        """에이전트 평가"""
        if num_episodes is None:
            num_episodes = self.config['evaluation_episodes']
        
        total_score = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_score = 0
            
            for _ in range(self.config['max_steps_per_episode']):
                # 결정적 행동 선택
                if agent_name == 'dqn':
                    action = agent.select_action(state, deterministic=True)
                else:  # ddpg
                    action = agent.get_deterministic_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_score += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            total_score += episode_score
        
        return total_score / num_episodes
    
    def analyze_deterministic_policies(self, dqn_agent, ddpg_agent) -> Dict[str, Any]:
        """결정적 정책 특성 분석"""
        print("\\n=== 결정적 정책 분석 ===")
        
        # 테스트 상태 생성
        test_states = []
        for _ in range(20):
            state, _ = self.env.reset()
            test_states.append(state)
        test_states = np.array(test_states)
        
        # DQN 분석
        dqn_analysis = dqn_agent.analyze_policy_determinism(test_states)
        
        # DDPG 분석
        ddpg_analysis = self._analyze_ddpg_determinism(ddpg_agent, test_states)
        
        # 행동 비교
        action_comparison = self._compare_action_selections(dqn_agent, ddpg_agent, test_states)
        
        analysis_results = {
            'dqn_determinism': dqn_analysis,
            'ddpg_determinism': ddpg_analysis,
            'action_comparison': action_comparison
        }
        
        self._print_determinism_analysis(analysis_results)
        
        return analysis_results
    
    def _analyze_ddpg_determinism(self, agent, states: np.ndarray, num_samples: int = 100) -> Dict[str, float]:
        """DDPG 결정성 분석"""
        determinism_scores = []
        output_variances = []
        
        for state in states:
            actions = []
            
            for _ in range(num_samples):
                action = agent.get_deterministic_action(state)
                actions.append(action[0])
            
            # 결정성 점수 (분산이 0에 가까우면 결정적)
            action_variance = np.var(actions)
            determinism_score = 1.0 if action_variance < 1e-8 else 0.0
            
            determinism_scores.append(determinism_score)
            output_variances.append(action_variance)
        
        return {
            'determinism_score': np.mean(determinism_scores),
            'output_variance': np.mean(output_variances),
            'consistency_rate': np.mean(determinism_scores),
            'noise_sensitivity': agent.noise.sigma if hasattr(agent, 'noise') else 0.0
        }
    
    def _compare_action_selections(self, dqn_agent, ddpg_agent, states: np.ndarray) -> Dict[str, Any]:
        """두 에이전트의 행동 선택 비교"""
        dqn_actions = []
        ddpg_actions = []
        
        for state in states:
            dqn_action = dqn_agent.get_deterministic_action(state)[0]
            ddpg_action = ddpg_agent.get_deterministic_action(state)[0]
            
            dqn_actions.append(dqn_action)
            ddpg_actions.append(ddpg_action)
        
        dqn_actions = np.array(dqn_actions)
        ddpg_actions = np.array(ddpg_actions)
        
        # 행동 차이 분석
        action_differences = np.abs(dqn_actions - ddpg_actions)
        
        return {
            'dqn_actions': dqn_actions.tolist(),
            'ddpg_actions': ddpg_actions.tolist(),
            'mean_difference': np.mean(action_differences),
            'max_difference': np.max(action_differences),
            'correlation': np.corrcoef(dqn_actions, ddpg_actions)[0, 1],
            'dqn_action_range': [float(np.min(dqn_actions)), float(np.max(dqn_actions))],
            'ddpg_action_range': [float(np.min(ddpg_actions)), float(np.max(ddpg_actions))]
        }
    
    def _print_determinism_analysis(self, analysis: Dict[str, Any]):
        """결정성 분석 결과 출력"""
        print("DQN (이산화된 결정적 정책):")
        dqn = analysis['dqn_determinism']
        print(f"  - 결정성 점수: {dqn['determinism_score']:.3f}")
        print(f"  - 일관성률: {dqn['consistency_rate']:.3f}")
        print(f"  - Q-값 안정성: {dqn['q_value_stability']:.6f}")
        
        print("\\nDDPG (연속 결정적 정책):")
        ddpg = analysis['ddpg_determinism']
        print(f"  - 결정성 점수: {ddpg['determinism_score']:.3f}")
        print(f"  - 일관성률: {ddpg['consistency_rate']:.3f}")
        print(f"  - 출력 분산: {ddpg['output_variance']:.6f}")
        
        print("\\n행동 선택 비교:")
        comp = analysis['action_comparison']
        print(f"  - 평균 차이: {comp['mean_difference']:.3f}")
        print(f"  - 최대 차이: {comp['max_difference']:.3f}")
        print(f"  - 상관관계: {comp['correlation']:.3f}")
        print(f"  - DQN 행동 범위: {comp['dqn_action_range']}")
        print(f"  - DDPG 행동 범위: {comp['ddpg_action_range']}")
    
    def run_comparison(self) -> Dict[str, Any]:
        """전체 비교 실험 실행"""
        set_seed(self.config['random_seed'])
        
        # 에이전트 생성
        dqn_agent, ddpg_agent = self.create_agents()
        
        # 개별 훈련
        print("\\n1단계: 개별 에이전트 훈련")
        dqn_results = self.train_agent(dqn_agent, 'dqn')
        ddpg_results = self.train_agent(ddpg_agent, 'ddpg')
        
        self.results['dqn'] = dqn_results
        self.results['ddpg'] = ddpg_results
        
        # 최종 평가
        print("\\n2단계: 최종 성능 평가")
        final_dqn_score = self.evaluate_agent(dqn_agent, 'dqn', 20)
        final_ddpg_score = self.evaluate_agent(ddpg_agent, 'ddpg', 20)
        
        print(f"최종 DQN 점수: {final_dqn_score:.2f}")
        print(f"최종 DDPG 점수: {final_ddpg_score:.2f}")
        
        # 결정적 정책 분석
        print("\\n3단계: 결정적 정책 분석")
        determinism_analysis = self.analyze_deterministic_policies(dqn_agent, ddpg_agent)
        
        # 결과 컴파일
        comparison_results = {
            'experiment_config': self.config,
            'training_results': self.results,
            'final_scores': {
                'dqn': final_dqn_score,
                'ddpg': final_ddpg_score
            },
            'determinism_analysis': determinism_analysis,
            'agent_info': {
                'dqn': dqn_agent.get_info(),
                'ddpg': ddpg_agent.get_info() if hasattr(ddpg_agent, 'get_info') else {
                    'agent_type': 'DDPG',
                    'policy_type': 'explicit_deterministic',
                    'action_space': 'continuous'
                }
            },
            'environment_info': {
                'name': 'ContinuousCartPole-v0',
                'state_dim': self.state_dim,
                'action_space': 'Box([-1, 1])',
                'physics': 'CartPole-v1_identical'
            }
        }
        
        return comparison_results
    
    def visualize_results(self, results: Dict[str, Any]):
        """결과 시각화 (새로운 시각화 시스템 사용)"""
        if NEW_VISUALIZATION_AVAILABLE:
            # 새로운 시각화 시스템 사용
            viz_config = VisualizationConfig()
            
            # 학습 곡선 시각화
            with LearningCurveVisualizer(output_dir=".", config=viz_config) as viz:
                fig = viz.plot_comprehensive_learning_curves(
                    results['training_results']['dqn'], 
                    results['training_results']['ddpg'],
                    save_filename="same_env_learning_curves.png"
                )
            
            # 비교 차트 시각화
            with ComparisonChartVisualizer(output_dir=".", config=viz_config) as viz:
                comparison_data = {
                    'dqn': results['training_results']['dqn'],
                    'ddpg': results['training_results']['ddpg']
                }
                fig = viz.plot_performance_comparison(
                    comparison_data['dqn'], comparison_data['ddpg'],
                    save_filename="same_env_performance_comparison.png"
                )
            
            return fig
        else:
            # 기본 시각화 (기존 코드)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('동일 환경 DQN vs DDPG 비교 결과', fontsize=16)
            
            # 학습 곡선
            axes[0, 0].plot(results['training_results']['dqn']['scores'], 
                           label='DQN', alpha=0.7, color='blue')
            axes[0, 0].plot(results['training_results']['ddpg']['scores'], 
                           label='DDPG', alpha=0.7, color='red')
            axes[0, 0].set_title('학습 성능 비교')
            axes[0, 0].set_xlabel('에피소드')
            axes[0, 0].set_ylabel('점수')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 평가 점수
            dqn_eval_episodes = [m['episode'] for m in results['training_results']['dqn']['metrics']]
            dqn_eval_scores = [m['eval_score'] for m in results['training_results']['dqn']['metrics']]
            ddpg_eval_episodes = [m['episode'] for m in results['training_results']['ddpg']['metrics']]
            ddpg_eval_scores = [m['eval_score'] for m in results['training_results']['ddpg']['metrics']]
            
            axes[0, 1].plot(dqn_eval_episodes, dqn_eval_scores, 
                           'o-', label='DQN', color='blue')
            axes[0, 1].plot(ddpg_eval_episodes, ddpg_eval_scores, 
                           'o-', label='DDPG', color='red')
            axes[0, 1].set_title('평가 성능 비교')
            axes[0, 1].set_xlabel('에피소드')
            axes[0, 1].set_ylabel('평가 점수')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 행동 비교
            comp = results['determinism_analysis']['action_comparison']
            axes[1, 0].scatter(comp['dqn_actions'], comp['ddpg_actions'], alpha=0.6)
            axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
            axes[1, 0].set_title('행동 선택 비교')
            axes[1, 0].set_xlabel('DQN 행동')
            axes[1, 0].set_ylabel('DDPG 행동')
            axes[1, 0].grid(True)
            
            # 결정성 비교
            dqn_det = results['determinism_analysis']['dqn_determinism']['determinism_score']
            ddpg_det = results['determinism_analysis']['ddpg_determinism']['determinism_score']
            
            axes[1, 1].bar(['DQN', 'DDPG'], [dqn_det, ddpg_det], 
                          color=['blue', 'red'], alpha=0.7)
            axes[1, 1].set_title('결정성 점수 비교')
            axes[1, 1].set_ylabel('결정성 점수')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            return fig
    
    def save_results(self, results: Dict[str, Any], save_dir: str = 'results/same_environment_comparison'):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 직렬화를 위한 결과 변환
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        # JSON 결과 저장
        json_path = os.path.join(save_dir, f'comparison_results_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 시각화 저장
        fig = self.visualize_results(results)
        fig_path = os.path.join(save_dir, f'comparison_plots_{timestamp}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\\n결과 저장 완료:")
        print(f"  - JSON: {json_path}")
        print(f"  - 그래프: {fig_path}")
        
        return json_path, fig_path


def main():
    """메인 실험 실행"""
    print("동일 환경 DQN vs DDPG 비교 실험 시작")
    
    # 실험 설정 (기본 설정 사용하되 일부 오버라이드)
    config = None  # 기본 설정 사용
    
    # 실험 실행
    comparison = SameEnvironmentComparison(config)
    results = comparison.run_comparison()
    
    # 결과 저장
    json_path, fig_path = comparison.save_results(results)
    
    # 요약 출력
    print("\\n" + "="*60)
    print("실험 결과 요약")
    print("="*60)
    print(f"환경: ContinuousCartPole-v0 (연속 행동 공간)")
    print(f"DQN 최종 점수: {results['final_scores']['dqn']:.2f}")
    print(f"DDPG 최종 점수: {results['final_scores']['ddpg']:.2f}")
    
    dqn_det = results['determinism_analysis']['dqn_determinism']['determinism_score']
    ddpg_det = results['determinism_analysis']['ddpg_determinism']['determinism_score']
    print(f"DQN 결정성: {dqn_det:.3f}")
    print(f"DDPG 결정성: {ddpg_det:.3f}")
    
    print("\\n실험 완료!")


if __name__ == "__main__":
    main()