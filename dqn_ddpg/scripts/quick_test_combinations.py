#!/usr/bin/env python3
"""
빠른 조합 테스트 스크립트 - 모든 알고리즘-환경 조합이 작동하는지 확인

각 조합을 10 에피소드만 실행하여 오류 없이 작동하는지 빠르게 확인합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import gymnasium as gym

from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.discretized_dqn_agent import DiscretizedDQNAgent
from src.agents.discrete_ddpg_agent import DiscreteDDPGAgent
from src.environments.env_factory import SimpleEnvironmentFactory
from src.core.config_manager import ConfigManager
from src.core.utils import get_device, set_seed


def test_combination(algorithm: str, environment: str, episodes: int = 10):
    """특정 조합 빠른 테스트"""
    
    print(f"🧪 테스트: {algorithm} + {environment}")
    
    try:
        # 환경 설정
        device = get_device()
        config_manager = ConfigManager()
        env_factory = SimpleEnvironmentFactory(config_manager)
        
        # 환경 생성
        env = env_factory.create_env(
            env_name=environment,
            agent_type='auto',
            training=True,
            seed=42
        )
        
        state_dim = env.observation_space.shape[0]
        
        # 에이전트 생성
        if environment == "CartPole-v1":
            if algorithm == "DQN":
                agent = DQNAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.n,
                    learning_rate=0.001,
                    device=device
                )
            elif algorithm == "DDPG":
                agent = DiscreteDDPGAgent(
                    state_dim=state_dim,
                    num_actions=env.action_space.n,
                    actor_lr=0.0001,  # learning_rate -> actor_lr
                    critic_lr=0.0001,
                    device=device
                )
                
        elif environment == "Pendulum-v1":
            if algorithm == "DQN":
                agent = DiscretizedDQNAgent(
                    state_dim=state_dim,
                    action_bound=2.0,
                    num_actions=21,
                    learning_rate=0.001,
                    device=device
                )
            elif algorithm == "DDPG":
                agent = DDPGAgent(
                    state_dim=state_dim,
                    action_dim=env.action_space.shape[0],
                    actor_lr=0.0001,  # learning_rate -> actor_lr
                    critic_lr=0.0001,
                    device=device
                )
        
        # 빠른 학습 테스트
        total_reward = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            max_steps = 50  # 빠른 테스트를 위해 단축
            
            for step in range(max_steps):
                # 행동 선택 (인터페이스 통일)
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state)
                else:
                    action = agent.act(state)
                
                # 환경 스텝
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 경험 저장 (DiscretizedDQN은 특별한 처리 필요)
                if isinstance(agent, DiscretizedDQNAgent):
                    # DiscretizedDQN의 경우 store_transition 사용 (자동으로 연속->이산 변환)
                    agent.store_transition(state, action, reward, next_state, done)
                else:
                    # 다른 에이전트는 직접 버퍼에 저장
                    agent.buffer.push(state, action, reward, next_state, done)
                
                # 학습 (버퍼가 충분하면)
                if len(agent.buffer) >= 32:  # 작은 배치로 테스트
                    loss_dict = agent.update()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / episodes
        env.close()
        
        print(f"   ✅ 성공! 평균 보상: {avg_reward:.2f}")
        return True
        
    except Exception as e:
        import traceback
        print(f"   ❌ 실패: {e}")
        print(f"   스택 트레이스: {traceback.format_exc()}")
        return False


def main():
    """모든 조합 빠른 테스트"""
    
    print("🧪 알고리즘-환경 조합 빠른 테스트\\n")
    
    set_seed(42)
    
    combinations = [
        ("DQN", "CartPole-v1"),     # 자연스러운 조합
        ("DDPG", "CartPole-v1"),    # 적응된 조합
        ("DQN", "Pendulum-v1"),     # 적응된 조합  
        ("DDPG", "Pendulum-v1")     # 자연스러운 조합
    ]
    
    results = {}
    
    for algorithm, environment in combinations:
        success = test_combination(algorithm, environment, episodes=10)
        results[f"{algorithm}_{environment}"] = success
    
    # 결과 요약
    print(f"\\n{'='*50}")
    print("📊 테스트 결과 요약")
    print(f"{'='*50}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for combination, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"   {combination:20s} | {status}")
    
    print(f"\\n🎯 전체 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 조합이 정상 작동합니다!")
        print("📝 이제 전체 학습을 실행할 수 있습니다:")
        print("   python scripts/train_cross_environment.py")
    else:
        print("⚠️ 일부 조합에 문제가 있습니다. 로그를 확인해주세요.")


if __name__ == "__main__":
    main()