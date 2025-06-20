#!/usr/bin/env python3
"""
DQN vs DDPG 간단한 시연
두 알고리즘의 결정적 정책 특성을 간단히 보여줍니다.
"""

import sys
import os
# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import set_seed


def demonstrate_dqn_deterministic_policy():
    """DQN의 암묵적 결정적 정책 시연"""
    print("\n" + "="*60)
    print("DQN 결정적 정책 시연")
    print("="*60)
    
    # 환경 및 에이전트 생성
    env = create_dqn_env("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, epsilon=0.0)  # 탐험 없음
    
    # 무작위 상태에서 Q-값 확인
    print("\n1. 무작위 상태에서 Q-값 분석:")
    for i in range(3):
        random_state = env.observation_space.sample()
        q_values = agent.get_q_values(random_state)
        selected_action = np.argmax(q_values)
        
        print(f"\n상태 {i+1}: {random_state[:2].round(2)}...")
        print(f"Q-값: {q_values.round(3)}")
        print(f"선택된 행동: {selected_action} (Q-값이 가장 높은 행동)")
    
    # 동일 상태에서 결정적 행동 확인
    print("\n2. 동일 상태에서 행동 일관성 테스트:")
    test_state = env.observation_space.sample()
    actions = []
    for i in range(5):
        action = agent.select_action(test_state, deterministic=True)
        actions.append(action)
    
    print(f"테스트 상태: {test_state[:2].round(2)}...")
    print(f"5번 선택된 행동: {actions}")
    print(f"모든 행동이 동일한가? {len(set(actions)) == 1}")
    
    env.close()


def demonstrate_ddpg_deterministic_policy():
    """DDPG의 명시적 결정적 정책 시연"""
    print("\n" + "="*60)
    print("DDPG 결정적 정책 시연")
    print("="*60)
    
    # 환경 및 에이전트 생성
    env = create_ddpg_env("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = DDPGAgent(state_dim, action_dim)
    
    # 무작위 상태에서 행동 출력 확인
    print("\n1. 무작위 상태에서 액터의 직접 행동 출력:")
    for i in range(3):
        random_state = env.observation_space.sample()
        action = agent.get_deterministic_action(random_state)
        
        print(f"\n상태 {i+1}: {random_state.round(2)}")
        print(f"액터 출력 (결정적 행동): {action.round(3)}")
    
    # 동일 상태에서 결정적 행동 확인
    print("\n2. 동일 상태에서 행동 일관성 테스트:")
    test_state = env.observation_space.sample()
    actions = []
    for i in range(5):
        action = agent.get_deterministic_action(test_state)
        actions.append(action)
    
    print(f"테스트 상태: {test_state.round(2)}")
    print("5번 출력된 행동:")
    for i, action in enumerate(actions):
        print(f"  {i+1}: {action.round(3)}")
    
    # 표준편차 계산
    actions_array = np.array(actions)
    std = np.std(actions_array, axis=0)
    print(f"행동 표준편차: {std} (0에 가까울수록 결정적)")
    
    # 노이즈 있는 행동과 비교
    print("\n3. 탐험 노이즈 유무 비교:")
    deterministic_action = agent.select_action(test_state, add_noise=False)
    noisy_actions = [agent.select_action(test_state, add_noise=True) for _ in range(3)]
    
    print(f"결정적 행동: {deterministic_action.round(3)}")
    print("노이즈 포함 행동:")
    for i, action in enumerate(noisy_actions):
        print(f"  {i+1}: {action.round(3)}")
    
    env.close()


def compare_policies():
    """DQN과 DDPG의 정책 특성 비교"""
    print("\n" + "="*60)
    print("DQN vs DDPG 정책 비교 요약")
    print("="*60)
    
    print("\n[DQN - 암묵적 결정적 정책]")
    print("- Q-네트워크가 각 이산 행동의 가치를 추정")
    print("- argmax Q(s,a)를 통해 간접적으로 최적 행동 선택")
    print("- 탐험: ε-greedy (확률적으로 무작위 행동)")
    
    print("\n[DDPG - 명시적 결정적 정책]")
    print("- 액터 네트워크가 직접 연속 행동을 출력")
    print("- μ(s)를 통해 직접적으로 결정적 행동 생성")
    print("- 탐험: 행동에 가우시안 노이즈 추가")
    
    print("\n[공통점]")
    print("- 둘 다 학습 완료 후 결정적 정책 보유")
    print("- 동일 상태 → 항상 동일 행동 (탐험 제외)")
    
    print("\n[차이점]")
    print("- 행동 공간: DQN(이산) vs DDPG(연속)")
    print("- 정책 표현: DQN(암묵적) vs DDPG(명시적)")
    print("- 네트워크 구조: DQN(단일) vs DDPG(액터-크리틱)")


def main():
    """메인 실행 함수"""
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    print("\n" + "#"*60)
    print("# DQN vs DDPG 결정적 정책 시연")
    print("#"*60)
    
    # 각 알고리즘 시연
    demonstrate_dqn_deterministic_policy()
    demonstrate_ddpg_deterministic_policy()
    
    # 비교 요약
    compare_policies()
    
    print("\n시연 완료!")


if __name__ == "__main__":
    main()