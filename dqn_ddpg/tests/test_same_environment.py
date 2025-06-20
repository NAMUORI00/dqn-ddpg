"""
동일 환경 비교 시스템 테스트

ContinuousCartPole 환경과 DiscretizedDQNAgent가 올바르게 작동하는지 검증합니다.
"""

import os
import sys
import numpy as np

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.continuous_cartpole import create_continuous_cartpole_env, compare_action_spaces
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.ddpg_agent import DDPGAgent


def test_continuous_cartpole():
    """ContinuousCartPole 환경 테스트"""
    print("=== ContinuousCartPole 환경 테스트 ===")
    
    env = create_continuous_cartpole_env()
    
    print(f"관찰 공간: {env.observation_space}")
    print(f"행동 공간: {env.action_space}")
    
    # 몇 스텝 실행
    state, info = env.reset(seed=42)
    print(f"초기 상태: {state}")
    
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"스텝 {i+1}:")
        print(f"  행동: {action}")
        print(f"  보상: {reward}")
        print(f"  종료: {terminated}")
        print(f"  force_applied: {info['force_applied']:.3f}")
        
        if terminated:
            break
    
    env.close()
    print("ContinuousCartPole 테스트 완료\\n")


def test_discretized_dqn():
    """DiscretizedDQNAgent 테스트"""
    print("=== DiscretizedDQNAgent 테스트 ===")
    
    env = create_continuous_cartpole_env()
    state_dim = env.observation_space.shape[0]
    
    # 에이전트 생성
    agent = DiscretizedDQNAgent(
        state_dim=state_dim,
        action_bound=1.0,
        num_actions=21,
        learning_rate=1e-3,
        epsilon=0.5  # 탐험을 위해 낮춤
    )
    
    print(f"에이전트 정보: {agent.get_info()}")
    
    # 몇 스텝 실행
    state, _ = env.reset(seed=42)
    
    for i in range(5):
        # 행동 선택
        action = agent.select_action(state)
        print(f"\\n스텝 {i+1}:")
        print(f"  연속 행동: {action}")
        print(f"  이산 매핑: {agent.continuous_to_discrete(action[0])}")
        
        # 환경 스텝
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 경험 저장
        agent.store_transition(state, action, reward, next_state, terminated)
        
        print(f"  보상: {reward}")
        print(f"  버퍼 크기: {len(agent.buffer.buffer)}")
        
        state = next_state
        if terminated:
            break
    
    # 학습 테스트 (버퍼가 충분할 때)
    if agent.buffer.is_ready(agent.batch_size):
        metrics = agent.update()
        print(f"\\n학습 메트릭: {metrics}")
    
    env.close()
    print("DiscretizedDQNAgent 테스트 완료\\n")


def test_determinism_analysis():
    """결정성 분석 테스트"""
    print("=== 결정성 분석 테스트 ===")
    
    env = create_continuous_cartpole_env()
    state_dim = env.observation_space.shape[0]
    
    # DQN 에이전트 (결정적 모드)
    dqn_agent = DiscretizedDQNAgent(
        state_dim=state_dim,
        action_bound=1.0,
        num_actions=21,
        epsilon=0.0  # 완전 결정적
    )
    
    # DDPG 에이전트
    ddpg_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        action_bound=1.0,
        noise_sigma=0.0  # 노이즈 없음
    )
    
    # 테스트 상태
    test_states = []
    for _ in range(3):
        state, _ = env.reset()
        test_states.append(state)
    test_states = np.array(test_states)
    
    # DQN 결정성 분석
    dqn_analysis = dqn_agent.analyze_policy_determinism(test_states, num_samples=10)
    print("DQN 결정성 분석:")
    for key, value in dqn_analysis.items():
        print(f"  {key}: {value:.6f}")
    
    # 행동 비교
    print("\\n행동 비교:")
    for i, state in enumerate(test_states):
        dqn_action = dqn_agent.get_deterministic_action(state)
        ddpg_action = ddpg_agent.get_deterministic_action(state)
        
        # 배열로 변환하여 일관성 유지
        dqn_val = dqn_action[0] if isinstance(dqn_action, np.ndarray) and dqn_action.ndim > 0 else float(dqn_action)
        ddpg_val = ddpg_action[0] if isinstance(ddpg_action, np.ndarray) and ddpg_action.ndim > 0 else float(ddpg_action)
        
        print(f"상태 {i+1}:")
        print(f"  DQN 행동: {dqn_val:.3f}")
        print(f"  DDPG 행동: {ddpg_val:.3f}")
        print(f"  차이: {abs(dqn_val - ddpg_val):.3f}")
    
    env.close()
    print("결정성 분석 테스트 완료\\n")


def test_action_discretization():
    """행동 이산화 테스트"""
    print("=== 행동 이산화 테스트 ===")
    
    agent = DiscretizedDQNAgent(
        state_dim=4,
        action_bound=1.0,
        num_actions=11  # 작은 수로 테스트
    )
    
    print(f"행동 매핑 테이블: {agent.action_values}")
    
    # 연속 값을 이산화하고 다시 연속으로 변환
    test_values = [-0.8, -0.3, 0.0, 0.3, 0.8]
    
    print("\\n연속 → 이산 → 연속 변환 테스트:")
    for value in test_values:
        discrete_idx = agent.continuous_to_discrete(value)
        continuous_back = agent.discrete_to_continuous(discrete_idx)
        
        comparison = agent.compare_with_continuous_action(value)
        
        print(f"원본: {value:.3f} → 이산: {discrete_idx} → 변환: {continuous_back[0]:.3f}")
        print(f"  오차: {comparison['discretization_error']:.3f}")
        print(f"  해상도: {comparison['action_resolution']:.3f}")
    
    print("행동 이산화 테스트 완료\\n")


def main():
    """모든 테스트 실행"""
    print("동일 환경 비교 시스템 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 환경 테스트
        test_continuous_cartpole()
        
        # 2. 이산화 DQN 테스트
        test_discretized_dqn()
        
        # 3. 행동 이산화 테스트
        test_action_discretization()
        
        # 4. 결정성 분석 테스트
        test_determinism_analysis()
        
        print("=" * 50)
        print("모든 테스트 통과! ✅")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()