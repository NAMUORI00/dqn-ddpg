"""
실시간 그래프 래퍼 테스트 및 데모
게임 화면 옆에 실시간으로 업데이트되는 그래프를 보여줍니다.
"""

import numpy as np
import gymnasium as gym
import cv2
import argparse
from pathlib import Path

from src.environments.realtime_graph_wrapper import RealtimeGraphWrapper, AdvancedMetricsWrapper
from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent


def test_basic_graph_wrapper():
    """기본 그래프 래퍼 테스트"""
    print("=== 기본 실시간 그래프 테스트 ===")
    
    # CartPole 환경에 그래프 래퍼 적용
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RealtimeGraphWrapper(
        env,
        graph_width=400,
        graph_height=300,
        history_length=50,
        ma_window=10,
        show_current_episode=True,
        algorithm_name="DQN"
    )
    
    # 간단한 랜덤 정책으로 테스트
    for episode in range(10):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 렌더링된 프레임 표시
            frame = env.render()
            if frame is not None:
                cv2.imshow('DQN with Realtime Graph', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return
    
    env.close()
    cv2.destroyAllWindows()


def test_advanced_metrics_wrapper():
    """고급 메트릭 래퍼 테스트"""
    print("\n=== 고급 메트릭 실시간 그래프 테스트 ===")
    
    # Pendulum 환경에 고급 메트릭 래퍼 적용
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = AdvancedMetricsWrapper(
        env,
        graph_width=500,
        graph_height=400,
        history_length=100,
        ma_window=20,
        show_current_episode=True,
        show_q_values=True,
        show_loss=True,
        show_exploration=True,
        algorithm_name="DDPG"
    )
    
    # 시뮬레이션된 메트릭으로 테스트
    for episode in range(10):
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 시뮬레이션된 메트릭 업데이트
            if step % 5 == 0:  # 5스텝마다 업데이트
                fake_q_value = np.random.randn() * 10 + episode * 5
                fake_loss = np.exp(-episode * 0.1) * (1 + np.random.rand())
                fake_exploration = max(0.01, 1.0 - episode * 0.1)
                
                env.update_metrics(
                    q_value=fake_q_value,
                    loss=fake_loss,
                    exploration_rate=fake_exploration
                )
            
            # 렌더링된 프레임 표시
            frame = env.render()
            if frame is not None:
                cv2.imshow('DDPG with Advanced Metrics', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return
            
            step += 1
    
    env.close()
    cv2.destroyAllWindows()


def test_with_real_agents():
    """실제 에이전트와 함께 테스트"""
    print("\n=== 실제 에이전트와 함께 실시간 그래프 테스트 ===")
    
    # DQN 테스트
    print("\n1. DQN 에이전트 테스트")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = AdvancedMetricsWrapper(
        env,
        graph_width=500,
        graph_height=400,
        history_length=100,
        ma_window=10,
        show_current_episode=True,
        show_q_values=True,
        show_loss=True,
        algorithm_name="DQN"
    )
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32
    )
    
    # 몇 에피소드 실행
    for episode in range(5):
        state, _ = env.reset()
        done = False
        
        while not done:
            # 에이전트 행동 선택
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장 및 학습
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                
                # 메트릭 업데이트
                q_values = agent.q_network(agent._to_tensor(state))
                max_q_value = q_values.max().item()
                
                env.update_metrics(
                    q_value=max_q_value,
                    loss=loss,
                    exploration_rate=agent.epsilon
                )
            
            state = next_state
            
            # 렌더링
            frame = env.render()
            if frame is not None:
                cv2.imshow('DQN Agent with Realtime Metrics', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return
        
        # 타겟 네트워크 업데이트
        if episode % 10 == 0:
            agent.update_target_network()
    
    env.close()
    cv2.destroyAllWindows()
    
    print("\n테스트 완료!")


def create_comparison_video_with_graphs():
    """두 알고리즘의 실시간 그래프를 포함한 비교 영상 생성"""
    print("\n=== DQN vs DDPG 실시간 그래프 비교 영상 생성 ===")
    
    # 출력 디렉토리 생성
    output_dir = Path("videos/realtime_graph_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output_dir / "realtime_graph_comparison.mp4"
    
    # 첫 프레임으로 크기 확인
    env1 = gym.make('CartPole-v1', render_mode='rgb_array')
    env1 = RealtimeGraphWrapper(env1, algorithm_name="DQN", graph_width=400)
    
    env2 = gym.make('Pendulum-v1', render_mode='rgb_array')
    env2 = RealtimeGraphWrapper(env2, algorithm_name="DDPG", graph_width=400)
    
    # 프레임 크기 확인
    env1.reset()
    env2.reset()
    frame1 = env1.render()
    frame2 = env2.render()
    
    # 높이 맞추기
    height = max(frame1.shape[0], frame2.shape[0])
    if frame1.shape[0] != height:
        frame1 = cv2.resize(frame1, (frame1.shape[1], height))
    if frame2.shape[0] != height:
        frame2 = cv2.resize(frame2, (frame2.shape[1], height))
    
    # 비디오 라이터 생성
    width = frame1.shape[1] + frame2.shape[1] + 20  # 중간 여백
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
    
    print(f"비디오 저장 위치: {output_path}")
    print("녹화 중... (20 에피소드)")
    
    # 동시에 실행
    for episode in range(20):
        print(f"에피소드 {episode + 1}/20")
        
        # 두 환경 리셋
        state1, _ = env1.reset()
        state2, _ = env2.reset()
        done1 = done2 = False
        
        # 최대 스텝 수 설정
        max_steps = 500
        step = 0
        
        while step < max_steps and (not done1 or not done2):
            # 각 환경에서 행동
            if not done1:
                action1 = env1.action_space.sample()
                state1, _, terminated1, truncated1, _ = env1.step(action1)
                done1 = terminated1 or truncated1
            
            if not done2:
                action2 = env2.action_space.sample()
                state2, _, terminated2, truncated2, _ = env2.step(action2)
                done2 = terminated2 or truncated2
            
            # 프레임 렌더링
            frame1 = env1.render()
            frame2 = env2.render()
            
            # 높이 맞추기
            if frame1.shape[0] != height:
                frame1 = cv2.resize(frame1, (frame1.shape[1], height))
            if frame2.shape[0] != height:
                frame2 = cv2.resize(frame2, (frame2.shape[1], height))
            
            # 중간 구분선 추가
            separator = np.ones((height, 20, 3), dtype=np.uint8) * 200
            
            # 프레임 결합
            combined_frame = np.hstack([frame1, separator, frame2])
            
            # BGR로 변환하여 저장
            video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            step += 1
    
    # 정리
    video_writer.release()
    env1.close()
    env2.close()
    
    print(f"\n비교 영상 생성 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실시간 그래프 래퍼 테스트")
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'advanced', 'agent', 'compare'],
                       help='테스트 모드 선택')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        test_basic_graph_wrapper()
    elif args.mode == 'advanced':
        test_advanced_metrics_wrapper()
    elif args.mode == 'agent':
        test_with_real_agents()
    elif args.mode == 'compare':
        create_comparison_video_with_graphs()
    
    print("\n모든 테스트 완료!")