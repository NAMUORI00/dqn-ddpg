"""
간단한 실시간 그래프 테스트 - 비디오 파일로 저장
"""

import numpy as np
import gymnasium as gym
import cv2
from pathlib import Path

from src.environments.realtime_graph_wrapper import RealtimeGraphWrapper


def test_and_save_video():
    """실시간 그래프를 비디오 파일로 저장하는 테스트"""
    print("=== 실시간 그래프 비디오 저장 테스트 ===")
    
    # 출력 디렉토리 생성
    output_dir = Path("videos/realtime_graph_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output_dir / "realtime_graph_demo.mp4"
    
    # 첫 프레임으로 크기 확인
    state, _ = env.reset()
    frame = env.render()
    if frame is None:
        print("ERROR: 프레임 렌더링 실패")
        return
    
    height, width = frame.shape[:2]
    fps = 30
    
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    print(f"비디오 저장 위치: {output_path}")
    print(f"프레임 크기: {width}x{height}")
    
    # 10개 에피소드 실행
    for episode in range(10):
        print(f"\n에피소드 {episode + 1}/10 실행 중...")
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 500:  # 최대 500 스텝
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 렌더링된 프레임을 비디오에 저장
            frame = env.render()
            if frame is not None:
                # RGB를 BGR로 변환하여 저장
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            step_count += 1
        
        print(f"  - 에피소드 종료: {step_count} 스텝, 총 보상: {env.total_reward:.1f}")
    
    # 정리
    video_writer.release()
    env.close()
    
    print(f"\n비디오 저장 완료: {output_path}")
    print("\n프레임 샘플 이미지도 저장합니다...")
    
    # 마지막 프레임을 이미지로도 저장
    if frame is not None:
        sample_path = output_dir / "sample_frame.png"
        cv2.imwrite(str(sample_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"샘플 이미지 저장: {sample_path}")


def test_algorithm_comparison():
    """DQN과 DDPG의 그래프를 나란히 비교하는 비디오 생성"""
    print("\n=== DQN vs DDPG 실시간 그래프 비교 ===")
    
    output_dir = Path("videos/realtime_graph_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 두 환경 생성
    env1 = gym.make('CartPole-v1', render_mode='rgb_array')
    env1 = RealtimeGraphWrapper(env1, algorithm_name="DQN", graph_width=350)
    
    env2 = gym.make('Pendulum-v1', render_mode='rgb_array')
    env2 = RealtimeGraphWrapper(env2, algorithm_name="DDPG", graph_width=350)
    
    # 비디오 설정
    output_path = output_dir / "dqn_vs_ddpg_graphs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 첫 프레임으로 크기 확인
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
    
    width = frame1.shape[1] + frame2.shape[1] + 20
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
    
    print(f"비교 비디오 저장 위치: {output_path}")
    
    # 5개 에피소드 실행
    for episode in range(5):
        print(f"\n에피소드 {episode + 1}/5")
        
        state1, _ = env1.reset()
        state2, _ = env2.reset()
        done1 = done2 = False
        step = 0
        max_steps = 300
        
        while step < max_steps and (not done1 or not done2):
            if not done1:
                action1 = env1.action_space.sample()
                state1, _, terminated1, truncated1, _ = env1.step(action1)
                done1 = terminated1 or truncated1
            
            if not done2:
                action2 = env2.action_space.sample()
                state2, _, terminated2, truncated2, _ = env2.step(action2)
                done2 = terminated2 or truncated2
            
            # 프레임 렌더링 및 결합
            frame1 = env1.render()
            frame2 = env2.render()
            
            if frame1.shape[0] != height:
                frame1 = cv2.resize(frame1, (frame1.shape[1], height))
            if frame2.shape[0] != height:
                frame2 = cv2.resize(frame2, (frame2.shape[1], height))
            
            # 중간 구분선
            separator = np.ones((height, 20, 3), dtype=np.uint8) * 200
            
            # 프레임 결합
            combined_frame = np.hstack([frame1, separator, frame2])
            video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            step += 1
        
        print(f"  DQN 보상: {env1.total_reward:.1f}, DDPG 보상: {env2.total_reward:.1f}")
    
    video_writer.release()
    env1.close()
    env2.close()
    
    print(f"\n비교 비디오 저장 완료: {output_path}")


if __name__ == "__main__":
    # 단일 알고리즘 테스트
    test_and_save_video()
    
    # 알고리즘 비교 테스트
    test_algorithm_comparison()
    
    print("\n모든 테스트 완료! videos/realtime_graph_test/ 폴더를 확인하세요.")