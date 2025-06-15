"""
빠른 동일 환경 비교 데모

짧은 훈련으로 DQN과 DDPG의 기본 동작을 확인합니다.
"""

import os
import sys
import numpy as np

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.continuous_cartpole import create_continuous_cartpole_env
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.core.utils import set_seed


def quick_train(agent, env, agent_name, episodes=20):
    """짧은 훈련"""
    print(f"\\n=== {agent_name} 짧은 훈련 ({episodes} 에피소드) ===")
    
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_score = 0
        
        for step in range(200):  # 최대 200 스텝
            # 행동 선택
            if agent_name == 'DQN':
                action = agent.select_action(state)
            else:  # DDPG
                action = agent.select_action(state)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 학습 (버퍼가 충분할 때)
            if agent.buffer.is_ready(min(32, agent.batch_size)):
                agent.update()
            
            episode_score += reward
            state = next_state
            
            if done:
                break
        
        scores.append(episode_score)
        
        if (episode + 1) % 5 == 0:
            recent_avg = np.mean(scores[-5:])
            print(f"{agent_name} Episode {episode+1}: 최근 평균 = {recent_avg:.1f}")
    
    final_avg = np.mean(scores[-5:])
    print(f"{agent_name} 최종 평균 점수: {final_avg:.1f}")
    
    return scores


def evaluate_determinism(agent, env, agent_name, num_tests=5):
    """결정성 평가"""
    print(f"\\n=== {agent_name} 결정성 평가 ===")
    
    test_states = []
    for _ in range(num_tests):
        state, _ = env.reset()
        test_states.append(state)
    
    # 각 상태에서 여러 번 행동 선택해서 일관성 확인
    for i, state in enumerate(test_states):
        actions = []
        for _ in range(10):
            if agent_name == 'DQN':
                action = agent.get_deterministic_action(state)
            else:  # DDPG
                action = agent.get_deterministic_action(state)
            
            # 값 추출
            action_val = action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else float(action)
            actions.append(action_val)
        
        # 일관성 확인
        unique_actions = len(set(actions))
        variance = np.var(actions)
        
        print(f"  상태 {i+1}: 행동={actions[0]:.3f}, 고유값={unique_actions}, 분산={variance:.6f}")
    
    return True


def main():
    """메인 데모 실행"""
    print("빠른 동일 환경 비교 데모")
    print("=" * 50)
    
    set_seed(42)
    
    # 환경 생성
    env = create_continuous_cartpole_env()
    state_dim = env.observation_space.shape[0]
    
    print(f"환경: ContinuousCartPole-v0")
    print(f"상태 차원: {state_dim}")
    print(f"행동 공간: {env.action_space}")
    
    # 에이전트 생성
    print("\\n에이전트 생성...")
    
    dqn_agent = DiscretizedDQNAgent(
        state_dim=state_dim,
        action_bound=1.0,
        num_actions=11,  # 적은 수로 빠른 테스트
        learning_rate=1e-3,
        epsilon=0.3,  # 약간의 탐험
        batch_size=32
    )
    
    ddpg_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        action_bound=1.0,
        actor_lr=1e-3,
        critic_lr=1e-3,
        noise_sigma=0.1,
        batch_size=32
    )
    
    print("DQN 에이전트 생성 완료")
    print("DDPG 에이전트 생성 완료")
    
    # 훈련
    dqn_scores = quick_train(dqn_agent, env, 'DQN', episodes=20)
    ddpg_scores = quick_train(ddpg_agent, env, 'DDPG', episodes=20)
    
    # 결정성 평가
    evaluate_determinism(dqn_agent, env, 'DQN')
    evaluate_determinism(ddpg_agent, env, 'DDPG')
    
    # 결과 요약
    print("\\n" + "=" * 50)
    print("결과 요약")
    print("=" * 50)
    print(f"DQN 최종 평균: {np.mean(dqn_scores[-5:]):.1f}")
    print(f"DDPG 최종 평균: {np.mean(ddpg_scores[-5:]):.1f}")
    
    # 동일 상태에서 행동 비교
    print("\\n동일 상태에서 행동 비교:")
    state, _ = env.reset(seed=123)
    
    dqn_action = dqn_agent.get_deterministic_action(state)
    ddpg_action = ddpg_agent.get_deterministic_action(state)
    
    dqn_val = dqn_action[0] if isinstance(dqn_action, np.ndarray) and dqn_action.ndim > 0 else float(dqn_action)
    ddpg_val = ddpg_action[0] if isinstance(ddpg_action, np.ndarray) and ddpg_action.ndim > 0 else float(ddpg_action)
    
    print(f"DQN 행동: {dqn_val:.3f}")
    print(f"DDPG 행동: {ddpg_val:.3f}")
    print(f"차이: {abs(dqn_val - ddpg_val):.3f}")
    
    env.close()
    print("\\n데모 완료! ✅")


if __name__ == "__main__":
    main()