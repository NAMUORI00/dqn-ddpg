"""
실시간 그래프 스크린샷 생성 테스트
"""

import numpy as np
import gymnasium as gym
import cv2
from pathlib import Path

from src.environments.realtime_graph_wrapper import RealtimeGraphWrapper, AdvancedMetricsWrapper


def generate_screenshots():
    """다양한 상태의 실시간 그래프 스크린샷 생성"""
    print("=== 실시간 그래프 스크린샷 생성 ===")
    
    output_dir = Path("videos/realtime_graph_test/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 기본 그래프 래퍼 테스트
    print("\n1. DQN 기본 그래프 테스트")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RealtimeGraphWrapper(
        env,
        graph_width=400,
        graph_height=300,
        history_length=50,
        ma_window=10,
        algorithm_name="DQN"
    )
    
    # 몇 에피소드 실행하여 데이터 쌓기
    for episode in range(15):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        print(f"  에피소드 {episode + 1}: {steps} 스텝, 보상 {env.total_reward:.1f}")
    
    # 마지막 프레임 저장
    frame = env.render()
    if frame is not None:
        cv2.imwrite(str(output_dir / "dqn_basic_graph.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  저장: {output_dir / 'dqn_basic_graph.png'}")
    
    env.close()
    
    # 2. DDPG 고급 메트릭 테스트
    print("\n2. DDPG 고급 메트릭 그래프 테스트")
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = AdvancedMetricsWrapper(
        env,
        graph_width=500,
        graph_height=400,
        history_length=100,
        ma_window=20,
        show_q_values=True,
        show_loss=True,
        algorithm_name="DDPG"
    )
    
    # 몇 에피소드 실행하며 가짜 메트릭 추가
    for episode in range(20):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 시뮬레이션된 메트릭 업데이트
            if steps % 10 == 0:
                fake_q_value = -100 + episode * 5 + np.random.randn() * 10
                fake_loss = np.exp(-episode * 0.05) * (0.1 + np.random.rand() * 0.05)
                
                env.update_metrics(
                    q_value=fake_q_value,
                    loss=fake_loss
                )
            
            steps += 1
        
        print(f"  에피소드 {episode + 1}: {steps} 스텝, 보상 {env.total_reward:.1f}")
    
    # 마지막 프레임 저장
    frame = env.render()
    if frame is not None:
        cv2.imwrite(str(output_dir / "ddpg_advanced_graph.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  저장: {output_dir / 'ddpg_advanced_graph.png'}")
    
    env.close()
    
    # 3. 나란히 비교 스크린샷
    print("\n3. DQN vs DDPG 비교 스크린샷")
    env1 = gym.make('CartPole-v1', render_mode='rgb_array')
    env1 = RealtimeGraphWrapper(env1, algorithm_name="DQN", graph_width=350)
    
    env2 = gym.make('Pendulum-v1', render_mode='rgb_array')
    env2 = RealtimeGraphWrapper(env2, algorithm_name="DDPG", graph_width=350)
    
    # 각각 10 에피소드 실행
    for episode in range(10):
        # DQN
        state1, _ = env1.reset()
        done1 = False
        while not done1:
            action1 = env1.action_space.sample()
            _, _, terminated1, truncated1, _ = env1.step(action1)
            done1 = terminated1 or truncated1
        
        # DDPG
        state2, _ = env2.reset()
        done2 = False
        step_count = 0
        while not done2 and step_count < 200:
            action2 = env2.action_space.sample()
            _, _, terminated2, truncated2, _ = env2.step(action2)
            done2 = terminated2 or truncated2
            step_count += 1
    
    # 프레임 렌더링 및 결합
    frame1 = env1.render()
    frame2 = env2.render()
    
    if frame1 is not None and frame2 is not None:
        # 높이 맞추기
        height = max(frame1.shape[0], frame2.shape[0])
        if frame1.shape[0] != height:
            frame1 = cv2.resize(frame1, (frame1.shape[1], height))
        if frame2.shape[0] != height:
            frame2 = cv2.resize(frame2, (frame2.shape[1], height))
        
        # 중간 구분선
        separator = np.ones((height, 20, 3), dtype=np.uint8) * 200
        
        # 프레임 결합
        combined_frame = np.hstack([frame1, separator, frame2])
        cv2.imwrite(str(output_dir / "dqn_vs_ddpg_comparison.png"), 
                   cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        print(f"  저장: {output_dir / 'dqn_vs_ddpg_comparison.png'}")
    
    env1.close()
    env2.close()
    
    print(f"\n모든 스크린샷이 {output_dir}에 저장되었습니다!")


if __name__ == "__main__":
    generate_screenshots()