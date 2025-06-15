#!/usr/bin/env python3
"""
결정적 정책 심층 분석 도구

DQN과 DDPG의 결정적 정책 특성을 다각도로 분석합니다:
- DQN: Q-값 기반 암묵적 결정성 (argmax 일관성)
- DDPG: 액터 출력 명시적 결정성 (출력 일관성)
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import set_seed


class DeterministicPolicyAnalyzer:
    """결정적 정책 분석기"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.analysis_results = {}
        
    def analyze_dqn_determinism(self, agent: DQNAgent, env, num_tests: int = 100) -> Dict:
        """DQN의 결정적 정책 특성 분석"""
        print("DQN 결정적 정책 분석 중...")
        
        results = {
            'q_value_consistency': [],
            'action_consistency': [],
            'q_value_differences': [],
            'argmax_confidence': [],
            'state_action_mapping': {},
            'epsilon_effect': {}
        }
        
        # 테스트 상태들 생성
        test_states = []
        for _ in range(num_tests):
            state = env.observation_space.sample()
            test_states.append(state)
        
        # 1. Q-값 일관성 테스트 (동일 상태에 대한 반복 Q-값 계산)
        print("  - Q-값 일관성 테스트")
        for i, state in enumerate(tqdm(test_states[:20], desc="Q-값 일관성")):
            q_values_repeated = []
            actions_repeated = []
            
            # 동일 상태에 대해 10번 반복 계산
            for _ in range(10):
                q_values = agent.get_q_values(state)
                if isinstance(q_values, torch.Tensor):
                    q_values_np = q_values.detach().cpu().numpy()
                    action = torch.argmax(q_values).item()
                else:
                    q_values_np = q_values
                    action = np.argmax(q_values)
                
                q_values_repeated.append(q_values_np)
                actions_repeated.append(action)
            
            # Q-값 분산 계산 (일관성 측정)
            q_values_array = np.array(q_values_repeated)
            q_value_std = np.std(q_values_array, axis=0)
            
            # 행동 일관성 계산 (모든 반복에서 같은 행동을 선택했는가?)
            action_consistency = len(set(actions_repeated)) == 1
            
            results['q_value_consistency'].append(q_value_std.mean())
            results['action_consistency'].append(action_consistency)
            
            # 상태-행동 매핑 저장
            results['state_action_mapping'][f'state_{i}'] = {
                'q_values': q_values_repeated[0].tolist(),
                'selected_action': int(actions_repeated[0]),
                'action_consistency': action_consistency
            }
        
        # 2. Q-값 차이 분석 (최고 Q-값과 두 번째 Q-값의 차이)
        print("  - Q-값 차이 분석")
        for state in tqdm(test_states, desc="Q-값 차이"):
            q_values = agent.get_q_values(state)
            if isinstance(q_values, torch.Tensor):
                q_values = q_values.detach().cpu().numpy()
            sorted_q = np.sort(q_values)[::-1]
            
            if len(sorted_q) > 1:
                q_diff = sorted_q[0] - sorted_q[1]
                results['q_value_differences'].append(q_diff)
                
                # argmax 신뢰도 (차이가 클수록 결정이 확실함)
                confidence = q_diff / (np.abs(sorted_q[0]) + 1e-8)
                results['argmax_confidence'].append(confidence)
        
        # 3. 엡실론 효과 분석
        print("  - 엡실론 효과 분석")
        original_epsilon = agent.epsilon
        test_state = test_states[0]
        
        epsilon_values = [0.0, 0.1, 0.5, 1.0]
        for eps in epsilon_values:
            agent.epsilon = eps
            actions = []
            for _ in range(100):
                action = agent.select_action(test_state)
                actions.append(action)
            
            # 행동 다양성 측정
            unique_actions = len(set(actions))
            action_distribution = np.bincount(actions, minlength=env.action_space.n)
            
            results['epsilon_effect'][f'eps_{eps}'] = {
                'unique_actions': unique_actions,
                'action_distribution': action_distribution.tolist(),
                'entropy': -np.sum(action_distribution / 100 * np.log(action_distribution / 100 + 1e-8))
            }
        
        agent.epsilon = original_epsilon
        
        # 통계 계산
        results['statistics'] = {
            'mean_q_consistency': np.mean(results['q_value_consistency']),
            'action_consistency_rate': np.mean(results['action_consistency']),
            'mean_q_difference': np.mean(results['q_value_differences']),
            'mean_argmax_confidence': np.mean(results['argmax_confidence'])
        }
        
        return results
    
    def analyze_ddpg_determinism(self, agent: DDPGAgent, env, num_tests: int = 100) -> Dict:
        """DDPG의 결정적 정책 특성 분석"""
        print("DDPG 결정적 정책 분석 중...")
        
        results = {
            'action_consistency': [],
            'action_variance': [],
            'noise_effect': {},
            'state_action_mapping': {},
            'deterministic_vs_noisy': []
        }
        
        # 테스트 상태들 생성
        test_states = []
        for _ in range(num_tests):
            state = env.observation_space.sample()
            test_states.append(state)
        
        # 1. 액터 출력 일관성 테스트
        print("  - 액터 출력 일관성 테스트")
        for i, state in enumerate(tqdm(test_states[:20], desc="액터 일관성")):
            actions_repeated = []
            
            # 동일 상태에 대해 10번 반복 (노이즈 없이)
            for _ in range(10):
                action = agent.get_deterministic_action(state)
                actions_repeated.append(action)
            
            actions_array = np.array(actions_repeated)
            
            # 액션 분산 계산 (낮을수록 결정적)
            action_variance = np.var(actions_array, axis=0)
            action_std = np.std(actions_array, axis=0)
            
            # 완전한 일관성인지 확인 (분산이 0에 가까운가?)
            is_consistent = np.allclose(action_variance, 0, atol=1e-6)
            
            results['action_consistency'].append(is_consistent)
            results['action_variance'].append(action_variance.mean())
            
            # 상태-행동 매핑 저장
            results['state_action_mapping'][f'state_{i}'] = {
                'deterministic_action': actions_repeated[0].tolist(),
                'action_std': action_std.tolist(),
                'is_consistent': is_consistent
            }
        
        # 2. 노이즈 효과 분석
        print("  - 노이즈 효과 분석")
        test_state = test_states[0]
        
        # 노이즈 없는 행동 vs 노이즈 있는 행동
        deterministic_actions = []
        noisy_actions = []
        
        for _ in range(50):
            det_action = agent.select_action(test_state, add_noise=False)
            noisy_action = agent.select_action(test_state, add_noise=True)
            
            deterministic_actions.append(det_action)
            noisy_actions.append(noisy_action)
        
        det_actions_array = np.array(deterministic_actions)
        noisy_actions_array = np.array(noisy_actions)
        
        # 노이즈로 인한 변화량 측정
        action_differences = []
        for det, noisy in zip(deterministic_actions, noisy_actions):
            diff = np.linalg.norm(np.array(noisy) - np.array(det))
            action_differences.append(diff)
        
        results['noise_effect'] = {
            'deterministic_std': np.std(det_actions_array, axis=0).tolist(),
            'noisy_std': np.std(noisy_actions_array, axis=0).tolist(),
            'mean_noise_impact': np.mean(action_differences),
            'noise_impact_std': np.std(action_differences)
        }
        
        results['deterministic_vs_noisy'] = action_differences
        
        # 3. 다양한 노이즈 강도 테스트
        print("  - 노이즈 강도 영향 분석")
        original_noise_std = agent.noise.sigma if hasattr(agent.noise, 'sigma') else 0.1
        
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
        for noise_std in noise_levels:
            if hasattr(agent.noise, 'sigma'):
                agent.noise.sigma = noise_std
            
            actions = []
            for _ in range(50):
                if noise_std == 0.0:
                    action = agent.select_action(test_state, add_noise=False)
                else:
                    action = agent.select_action(test_state, add_noise=True)
                actions.append(action)
            
            actions_array = np.array(actions)
            action_diversity = np.std(actions_array, axis=0).mean()
            
            results['noise_effect'][f'noise_{noise_std}'] = {
                'action_diversity': action_diversity,
                'mean_action': np.mean(actions_array, axis=0).tolist(),
                'std_action': np.std(actions_array, axis=0).tolist()
            }
        
        # 원래 노이즈 강도 복원
        if hasattr(agent.noise, 'sigma'):
            agent.noise.sigma = original_noise_std
        
        # 통계 계산
        results['statistics'] = {
            'action_consistency_rate': np.mean(results['action_consistency']),
            'mean_action_variance': np.mean(results['action_variance']),
            'mean_noise_impact': results['noise_effect']['mean_noise_impact']
        }
        
        return results
    
    def compare_determinism(self, dqn_results: Dict, ddpg_results: Dict) -> Dict:
        """DQN과 DDPG의 결정성 비교"""
        print("결정적 정책 특성 비교 분석 중...")
        
        comparison = {
            'determinism_scores': {},
            'consistency_comparison': {},
            'implementation_differences': {}
        }
        
        # 1. 결정성 점수 계산 (0-1 스케일)
        dqn_determinism = dqn_results['statistics']['action_consistency_rate']
        ddpg_determinism = ddpg_results['statistics']['action_consistency_rate']
        
        comparison['determinism_scores'] = {
            'dqn_score': dqn_determinism,
            'ddpg_score': ddpg_determinism,
            'difference': abs(dqn_determinism - ddpg_determinism)
        }
        
        # 2. 일관성 메커니즘 비교
        comparison['consistency_comparison'] = {
            'dqn': {
                'mechanism': 'argmax over Q-values',
                'consistency_rate': dqn_determinism,
                'q_value_stability': dqn_results['statistics']['mean_q_consistency'],
                'decision_confidence': dqn_results['statistics']['mean_argmax_confidence']
            },
            'ddpg': {
                'mechanism': 'direct actor output',
                'consistency_rate': ddpg_determinism,
                'output_variance': ddpg_results['statistics']['mean_action_variance'],
                'noise_sensitivity': ddpg_results['statistics']['mean_noise_impact']
            }
        }
        
        # 3. 구현 방식 차이점
        comparison['implementation_differences'] = {
            'policy_representation': {
                'dqn': 'implicit (Q-values → argmax)',
                'ddpg': 'explicit (actor network direct output)'
            },
            'action_space': {
                'dqn': 'discrete (finite set)',
                'ddpg': 'continuous (infinite set)'
            },
            'exploration_strategy': {
                'dqn': 'epsilon-greedy (probabilistic)',
                'ddpg': 'additive noise (deterministic + noise)'
            }
        }
        
        return comparison
    
    def generate_visualizations(self, dqn_results: Dict, ddpg_results: Dict, 
                              comparison: Dict, save_dir: str):
        """결정적 정책 분석 시각화 생성"""
        print("분석 결과 시각화 생성 중...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 결정성 비교 차트
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DQN vs DDPG Deterministic Policy Analysis', fontsize=16, fontweight='bold')
        
        # DQN Q-값 일관성
        ax = axes[0, 0]
        ax.hist(dqn_results['q_value_consistency'], bins=20, alpha=0.7, 
                color='blue', edgecolor='black')
        ax.set_xlabel('Q-value Standard Deviation')
        ax.set_ylabel('Frequency')
        ax.set_title('DQN: Q-value Consistency')
        ax.grid(True, alpha=0.3)
        
        # DQN argmax 신뢰도
        ax = axes[0, 1]
        ax.hist(dqn_results['argmax_confidence'], bins=20, alpha=0.7, 
                color='blue', edgecolor='black')
        ax.set_xlabel('Argmax Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('DQN: Argmax Confidence')
        ax.grid(True, alpha=0.3)
        
        # DQN Q-값 차이
        ax = axes[0, 2]
        ax.hist(dqn_results['q_value_differences'], bins=20, alpha=0.7, 
                color='blue', edgecolor='black')
        ax.set_xlabel('Q-value Difference (Best - 2nd Best)')
        ax.set_ylabel('Frequency')
        ax.set_title('DQN: Q-value Differences')
        ax.grid(True, alpha=0.3)
        
        # DDPG 액션 분산
        ax = axes[1, 0]
        ax.hist(ddpg_results['action_variance'], bins=20, alpha=0.7, 
                color='red', edgecolor='black')
        ax.set_xlabel('Action Variance')
        ax.set_ylabel('Frequency')
        ax.set_title('DDPG: Action Output Variance')
        ax.grid(True, alpha=0.3)
        
        # DDPG 노이즈 영향
        ax = axes[1, 1]
        ax.hist(ddpg_results['deterministic_vs_noisy'], bins=20, alpha=0.7, 
                color='red', edgecolor='black')
        ax.set_xlabel('Action Change due to Noise')
        ax.set_ylabel('Frequency')
        ax.set_title('DDPG: Action Change due to Noise')
        ax.grid(True, alpha=0.3)
        
        # 결정성 점수 비교
        ax = axes[1, 2]
        algorithms = ['DQN', 'DDPG']
        scores = [comparison['determinism_scores']['dqn_score'], 
                 comparison['determinism_scores']['ddpg_score']]
        bars = ax.bar(algorithms, scores, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Determinism Score')
        ax.set_title('Deterministic Policy Consistency')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'deterministic_policy_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 노이즈 효과 분석 (DDPG 전용)
        if 'noise_effect' in ddpg_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            noise_levels = []
            diversities = []
            
            for key, value in ddpg_results['noise_effect'].items():
                if key.startswith('noise_') and key not in ['mean_noise_impact', 'noise_impact_std']:
                    try:
                        noise_level = float(key.split('_')[1])
                        diversity = value['action_diversity']
                        noise_levels.append(noise_level)
                        diversities.append(diversity)
                    except (ValueError, KeyError):
                        continue
            
            ax.plot(noise_levels, diversities, 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Noise Standard Deviation')
            ax.set_ylabel('Action Diversity (Std)')
            ax.set_title('DDPG: Action Diversity vs Noise Level')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'ddpg_noise_effect.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"시각화 결과 저장: {save_dir}")
    
    def run_full_analysis(self, dqn_model_path: str = None, ddpg_model_path: str = None):
        """전체 분석 실행"""
        print("="*60)
        print("결정적 정책 심층 분석 시작")
        print("="*60)
        
        # 결과 저장 디렉토리 생성
        analysis_dir = os.path.join(self.results_dir, "deterministic_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        dqn_results = None
        ddpg_results = None
        
        # DQN 분석
        if dqn_model_path and os.path.exists(dqn_model_path):
            print("\nDQN 모델 로드 및 분석...")
            env = create_dqn_env("CartPole-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            agent = DQNAgent(state_dim, action_dim)
            agent.load(dqn_model_path)
            agent.q_network.eval()
            
            dqn_results = self.analyze_dqn_determinism(agent, env)
            env.close()
        else:
            print("\nDQN 모델을 찾을 수 없어 샘플 에이전트로 분석...")
            env = create_dqn_env("CartPole-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            agent = DQNAgent(state_dim, action_dim)
            dqn_results = self.analyze_dqn_determinism(agent, env)
            env.close()
        
        # DDPG 분석
        if ddpg_model_path and os.path.exists(ddpg_model_path):
            print("\nDDPG 모델 로드 및 분석...")
            env = create_ddpg_env("Pendulum-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            agent = DDPGAgent(state_dim, action_dim)
            agent.load(ddpg_model_path)
            agent.actor.eval()
            agent.critic.eval()
            
            ddpg_results = self.analyze_ddpg_determinism(agent, env)
            env.close()
        else:
            print("\nDDPG 모델을 찾을 수 없어 샘플 에이전트로 분석...")
            env = create_ddpg_env("Pendulum-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            agent = DDPGAgent(state_dim, action_dim)
            ddpg_results = self.analyze_ddpg_determinism(agent, env)
            env.close()
        
        # 비교 분석
        comparison = self.compare_determinism(dqn_results, ddpg_results)
        
        # 결과 저장
        self.analysis_results = {
            'dqn_analysis': dqn_results,
            'ddpg_analysis': ddpg_results,
            'comparison': comparison
        }
        
        results_file = os.path.join(analysis_dir, "deterministic_policy_analysis.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # numpy 타입을 Python 기본 타입으로 변환
            json_serializable_results = self._convert_to_json_serializable(self.analysis_results)
            json.dump(json_serializable_results, f, indent=2, ensure_ascii=False)
        
        # 시각화 생성
        self.generate_visualizations(dqn_results, ddpg_results, comparison, analysis_dir)
        
        # 결과 요약 출력
        self.print_summary(comparison)
        
        print(f"\n분석 완료! 결과 저장 위치: {analysis_dir}")
        return self.analysis_results
    
    def print_summary(self, comparison: Dict):
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("결정적 정책 분석 결과 요약")
        print("="*60)
        
        print(f"\n📊 결정성 점수:")
        print(f"  • DQN:  {comparison['determinism_scores']['dqn_score']:.3f}")
        print(f"  • DDPG: {comparison['determinism_scores']['ddpg_score']:.3f}")
        print(f"  • 차이:  {comparison['determinism_scores']['difference']:.3f}")
        
        print(f"\n🔍 구현 메커니즘:")
        dqn_info = comparison['consistency_comparison']['dqn']
        ddpg_info = comparison['consistency_comparison']['ddpg']
        
        print(f"  • DQN:")
        print(f"    - 메커니즘: {dqn_info['mechanism']}")
        print(f"    - 일관성: {dqn_info['consistency_rate']:.3f}")
        print(f"    - Q-값 안정성: {dqn_info['q_value_stability']:.6f}")
        
        print(f"  • DDPG:")
        print(f"    - 메커니즘: {ddpg_info['mechanism']}")
        print(f"    - 일관성: {ddpg_info['consistency_rate']:.3f}")
        print(f"    - 출력 분산: {ddpg_info['output_variance']:.6f}")
        
        print(f"\n💡 핵심 차이점:")
        impl_diff = comparison['implementation_differences']
        print(f"  • 정책 표현: DQN({impl_diff['policy_representation']['dqn']}) vs DDPG({impl_diff['policy_representation']['ddpg']})")
        print(f"  • 행동 공간: DQN({impl_diff['action_space']['dqn']}) vs DDPG({impl_diff['action_space']['ddpg']})")
        print(f"  • 탐험 전략: DQN({impl_diff['exploration_strategy']['dqn']}) vs DDPG({impl_diff['exploration_strategy']['ddpg']})")
    
    def _convert_to_json_serializable(self, obj):
        """numpy 타입을 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description='결정적 정책 심층 분석')
    parser.add_argument('--dqn-model', type=str, 
                       help='DQN 모델 경로 (선택사항)')
    parser.add_argument('--ddpg-model', type=str, 
                       help='DDPG 모델 경로 (선택사항)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 분석 실행
    analyzer = DeterministicPolicyAnalyzer(args.results_dir)
    analyzer.run_full_analysis(args.dqn_model, args.ddpg_model)


if __name__ == "__main__":
    main()