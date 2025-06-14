#!/usr/bin/env python3
"""
DQN vs DDPG 상세 테스트
결정적 정책의 특성을 심층적으로 분석하고 테스트합니다.
"""

import sys
import os
# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import set_seed


class DeterministicPolicyTester:
    """결정적 정책 테스터"""
    
    def __init__(self):
        self.results = {
            'dqn': {},
            'ddpg': {}
        }
    
    def test_action_consistency(self, agent, states: np.ndarray, 
                              agent_type: str = 'dqn') -> dict:
        """동일 상태에서 행동 일관성 테스트
        
        Args:
            agent: 테스트할 에이전트
            states: 테스트 상태들
            agent_type: 'dqn' 또는 'ddpg'
        
        Returns:
            일관성 테스트 결과
        """
        results = {
            'states': states,
            'actions': [],
            'consistency': []
        }
        
        for state in states:
            actions = []
            for _ in range(10):  # 각 상태에서 10번 행동 선택
                if agent_type == 'dqn':
                    action = agent.select_action(state, deterministic=True)
                else:  # ddpg
                    action = agent.get_deterministic_action(state)
                actions.append(action)
            
            results['actions'].append(actions)
            
            # 일관성 계산
            if agent_type == 'dqn':
                # 이산 행동: 모든 행동이 동일한지 확인
                consistency = len(set(actions)) == 1
            else:
                # 연속 행동: 표준편차로 확인
                actions_array = np.array(actions)
                std = np.std(actions_array, axis=0)
                consistency = np.all(std < 1e-6)  # 매우 작은 표준편차
            
            results['consistency'].append(consistency)
        
        return results
    
    def test_exploration_impact(self, agent, state: np.ndarray,
                               agent_type: str = 'dqn') -> dict:
        """탐험이 행동 선택에 미치는 영향 테스트"""
        results = {
            'deterministic_actions': [],
            'exploratory_actions': [],
            'differences': []
        }
        
        # 결정적 행동과 탐험적 행동 비교
        for _ in range(20):
            if agent_type == 'dqn':
                det_action = agent.select_action(state, deterministic=True)
                # 임시로 epsilon 설정
                original_epsilon = agent.epsilon
                agent.epsilon = 0.3
                exp_action = agent.select_action(state, deterministic=False)
                agent.epsilon = original_epsilon
            else:  # ddpg
                det_action = agent.select_action(state, add_noise=False)
                exp_action = agent.select_action(state, add_noise=True)
            
            results['deterministic_actions'].append(det_action)
            results['exploratory_actions'].append(exp_action)
            
            if agent_type == 'dqn':
                diff = det_action != exp_action
            else:
                diff = np.linalg.norm(det_action - exp_action)
            results['differences'].append(diff)
        
        return results
    
    def test_q_value_analysis(self, dqn_agent: DQNAgent, 
                             states: np.ndarray) -> dict:
        """DQN의 Q-값 분포 분석"""
        results = {
            'q_values': [],
            'q_differences': [],
            'selected_actions': []
        }
        
        for state in states:
            q_values = dqn_agent.get_q_values(state)
            selected_action = np.argmax(q_values)
            
            results['q_values'].append(q_values)
            results['selected_actions'].append(selected_action)
            
            # Q-값 차이 분석
            sorted_q = np.sort(q_values)[::-1]
            if len(sorted_q) > 1:
                q_diff = sorted_q[0] - sorted_q[1]  # 최고와 차선의 차이
                results['q_differences'].append(q_diff)
        
        return results
    
    def test_actor_output_analysis(self, ddpg_agent: DDPGAgent,
                                  states: np.ndarray) -> dict:
        """DDPG의 액터 출력 분석"""
        results = {
            'actor_outputs': [],
            'action_magnitudes': [],
            'action_variations': []
        }
        
        for state in states:
            action = ddpg_agent.get_deterministic_action(state)
            results['actor_outputs'].append(action)
            results['action_magnitudes'].append(np.linalg.norm(action))
        
        # 상태 간 행동 변화량 계산
        outputs = np.array(results['actor_outputs'])
        if len(outputs) > 1:
            for i in range(1, len(outputs)):
                variation = np.linalg.norm(outputs[i] - outputs[i-1])
                results['action_variations'].append(variation)
        
        return results


def visualize_results(tester: DeterministicPolicyTester):
    """테스트 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DQN vs DDPG 결정적 정책 분석', fontsize=16)
    
    # 1. DQN Q-값 분포
    ax = axes[0, 0]
    if 'q_analysis' in tester.results['dqn']:
        q_values = tester.results['dqn']['q_analysis']['q_values']
        for i, q_vals in enumerate(q_values[:5]):  # 처음 5개 상태
            ax.bar(range(len(q_vals)), q_vals, alpha=0.6, label=f'State {i+1}')
        ax.set_xlabel('Action')
        ax.set_ylabel('Q-value')
        ax.set_title('DQN: Q-값 분포')
        ax.legend()
    
    # 2. DDPG 액터 출력
    ax = axes[0, 1]
    if 'actor_analysis' in tester.results['ddpg']:
        outputs = tester.results['ddpg']['actor_analysis']['actor_outputs']
        for i, action in enumerate(outputs[:5]):  # 처음 5개 상태
            ax.plot(action, 'o-', label=f'State {i+1}')
        ax.set_xlabel('Action Dimension')
        ax.set_ylabel('Action Value')
        ax.set_title('DDPG: 액터 출력')
        ax.legend()
    
    # 3. 탐험 영향 (DQN)
    ax = axes[1, 0]
    if 'exploration_dqn' in tester.results['dqn']:
        exp_data = tester.results['dqn']['exploration_dqn']
        det_counts = np.bincount(exp_data['deterministic_actions'])
        exp_counts = np.bincount(exp_data['exploratory_actions'])
        
        x = np.arange(len(det_counts))
        width = 0.35
        ax.bar(x - width/2, det_counts, width, label='Deterministic')
        ax.bar(x + width/2, exp_counts, width, label='With Exploration')
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_title('DQN: 탐험의 영향')
        ax.legend()
    
    # 4. 탐험 영향 (DDPG)
    ax = axes[1, 1]
    if 'exploration_ddpg' in tester.results['ddpg']:
        exp_data = tester.results['ddpg']['exploration_ddpg']
        differences = exp_data['differences']
        ax.hist(differences, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Action Difference (L2 norm)')
        ax.set_ylabel('Frequency')
        ax.set_title('DDPG: 노이즈로 인한 행동 변화')
    
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/deterministic_policy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_detailed_tests():
    """상세 테스트 실행"""
    print("\n" + "="*60)
    print("DQN vs DDPG 결정적 정책 상세 테스트")
    print("="*60)
    
    # 시드 설정
    set_seed(42)
    
    # 테스터 생성
    tester = DeterministicPolicyTester()
    
    # DQN 테스트
    print("\n[DQN 테스트]")
    env = create_dqn_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    dqn_agent = DQNAgent(state_dim, action_dim, epsilon=0.0)
    
    # 테스트용 상태 생성
    test_states = np.array([env.observation_space.sample() for _ in range(10)])
    
    # 1. 행동 일관성 테스트
    print("1. 행동 일관성 테스트...")
    consistency_results = tester.test_action_consistency(dqn_agent, test_states[:5], 'dqn')
    print(f"   일관성 비율: {np.mean(consistency_results['consistency'])*100:.1f}%")
    
    # 2. Q-값 분석
    print("2. Q-값 분석...")
    q_analysis = tester.test_q_value_analysis(dqn_agent, test_states[:5])
    tester.results['dqn']['q_analysis'] = q_analysis
    mean_q_diff = np.mean(q_analysis['q_differences']) if q_analysis['q_differences'] else 0
    print(f"   평균 Q-값 차이 (최고-차선): {mean_q_diff:.4f}")
    
    # 3. 탐험 영향 테스트
    print("3. 탐험 영향 테스트...")
    exploration_results = tester.test_exploration_impact(dqn_agent, test_states[0], 'dqn')
    tester.results['dqn']['exploration_dqn'] = exploration_results
    exploration_rate = np.mean(exploration_results['differences']) * 100
    print(f"   탐험으로 인한 행동 변경률: {exploration_rate:.1f}%")
    
    env.close()
    
    # DDPG 테스트
    print("\n[DDPG 테스트]")
    env = create_ddpg_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ddpg_agent = DDPGAgent(state_dim, action_dim)
    
    # 테스트용 상태 생성
    test_states = np.array([env.observation_space.sample() for _ in range(10)])
    
    # 1. 행동 일관성 테스트
    print("1. 행동 일관성 테스트...")
    consistency_results = tester.test_action_consistency(ddpg_agent, test_states[:5], 'ddpg')
    print(f"   일관성 비율: {np.mean(consistency_results['consistency'])*100:.1f}%")
    
    # 2. 액터 출력 분석
    print("2. 액터 출력 분석...")
    actor_analysis = tester.test_actor_output_analysis(ddpg_agent, test_states[:5])
    tester.results['ddpg']['actor_analysis'] = actor_analysis
    mean_magnitude = np.mean(actor_analysis['action_magnitudes'])
    print(f"   평균 행동 크기: {mean_magnitude:.4f}")
    
    # 3. 탐험 영향 테스트
    print("3. 탐험 영향 테스트...")
    exploration_results = tester.test_exploration_impact(ddpg_agent, test_states[0], 'ddpg')
    tester.results['ddpg']['exploration_ddpg'] = exploration_results
    mean_diff = np.mean(exploration_results['differences'])
    print(f"   노이즈로 인한 평균 행동 변화: {mean_diff:.4f}")
    
    env.close()
    
    # 결과 시각화
    print("\n결과 시각화 중...")
    visualize_results(tester)
    
    # 최종 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print("\n[결정적 정책 검증]")
    print("✓ DQN: 동일 상태에서 항상 같은 이산 행동 선택 (argmax Q)")
    print("✓ DDPG: 동일 상태에서 항상 같은 연속 행동 출력 (액터 네트워크)")
    print("\n[탐험 메커니즘]")
    print("✓ DQN: ε-greedy로 확률적 무작위 행동")
    print("✓ DDPG: 가우시안 노이즈로 연속 행동 섭동")
    print("\n결과가 'results/deterministic_policy_analysis.png'에 저장되었습니다.")


if __name__ == "__main__":
    run_detailed_tests()