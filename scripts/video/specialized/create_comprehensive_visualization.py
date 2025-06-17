"""
종합적인 시각화 통합영상 생성 스크립트
게임플레이 + 실시간 그래프 + 비교분석을 모두 포함한 완전한 시각화를 생성합니다.
"""

import numpy as np
import sys
import os
import gymnasium as gym
import cv2
import argparse
from pathlib import Path
import time

# 프로젝트 루트 추가 (scripts/video/specialized에서 루트로)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.comprehensive_visualization_wrapper import (
    ComprehensiveVisualizationWrapper, 
    create_side_by_side_comparison
)
from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent


def create_comprehensive_demo_video():
    """종합 시각화 데모 비디오 생성"""
    print("=== 종합 시각화 통합영상 생성 ===")
    
    # 작업 디렉토리를 루트로 변경
    os.chdir(project_root)
    
    # 출력 디렉토리 생성
    output_dir = Path("videos/comprehensive_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. DQN 환경 설정
    dqn_env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn_wrapper = ComprehensiveVisualizationWrapper(
        dqn_env,
        algorithm_name="DQN",
        frame_width=600,
        frame_height=400,
        graph_height=300,
        stats_height=150
    )
    
    # 2. DDPG 환경 설정  
    ddpg_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    ddpg_wrapper = ComprehensiveVisualizationWrapper(
        ddpg_env,
        algorithm_name="DDPG",
        frame_width=600,
        frame_height=400,
        graph_height=300,
        stats_height=150
    )
    
    # 상호 참조 설정 (비교용)
    dqn_wrapper.partner_wrapper = ddpg_wrapper
    ddpg_wrapper.partner_wrapper = dqn_wrapper
    
    # 3. 비디오 라이터 설정
    # 첫 프레임으로 크기 확인
    dqn_wrapper.reset()
    ddpg_wrapper.reset()
    
    combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
    if combined_frame is None:
        print("ERROR: 프레임 생성 실패")
        return
    
    height, width = combined_frame.shape[:2]
    fps = 30
    
    # 메인 비디오 (나란히 비교)
    main_output = output_dir / "comprehensive_dqn_vs_ddpg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    main_writer = cv2.VideoWriter(str(main_output), fourcc, fps, (width, height))
    
    # 개별 비디오들도 생성
    dqn_output = output_dir / "dqn_comprehensive.mp4"
    ddpg_output = output_dir / "ddpg_comprehensive.mp4"
    
    dqn_single_frame = dqn_wrapper.render()
    ddpg_single_frame = ddpg_wrapper.render()
    
    dqn_writer = cv2.VideoWriter(str(dqn_output), fourcc, fps, 
                                (dqn_single_frame.shape[1], dqn_single_frame.shape[0]))
    ddpg_writer = cv2.VideoWriter(str(ddpg_output), fourcc, fps,
                                 (ddpg_single_frame.shape[1], ddpg_single_frame.shape[0]))
    
    print(f"메인 통합 비디오: {main_output}")
    print(f"DQN 개별 비디오: {dqn_output}")
    print(f"DDPG 개별 비디오: {ddpg_output}")
    print(f"프레임 크기: {width}x{height}")
    
    # 4. 에피소드 실행 및 녹화
    total_episodes = 25
    print(f"\n{total_episodes}개 에피소드 실행 시작...")
    
    for episode in range(total_episodes):
        print(f"\n에피소드 {episode + 1}/{total_episodes}")
        
        # 환경 리셋
        dqn_wrapper.reset()
        ddpg_wrapper.reset()
        
        dqn_done = False
        ddpg_done = False
        step = 0
        max_steps = 500
        
        while step < max_steps and (not dqn_done or not ddpg_done):
            # DQN 스텝
            if not dqn_done:
                dqn_action = dqn_wrapper.action_space.sample()
                _, dqn_reward, dqn_terminated, dqn_truncated, _ = dqn_wrapper.step(dqn_action)
                dqn_done = dqn_terminated or dqn_truncated
                
                # 시뮬레이션된 메트릭 업데이트
                if step % 5 == 0:
                    fake_q_value = np.random.randn() * 5 + episode * 2
                    fake_loss = np.exp(-episode * 0.05) * (0.1 + np.random.rand() * 0.1)
                    fake_exploration = max(0.01, 1.0 - episode * 0.04)
                    
                    dqn_wrapper.update_metrics(
                        q_value=fake_q_value,
                        loss=fake_loss,
                        exploration_rate=fake_exploration
                    )
            
            # DDPG 스텝
            if not ddpg_done:
                ddpg_action = ddpg_wrapper.action_space.sample()
                _, ddpg_reward, ddpg_terminated, ddpg_truncated, _ = ddpg_wrapper.step(ddpg_action)
                ddpg_done = ddpg_terminated or ddpg_truncated
                
                # 시뮬레이션된 메트릭 업데이트
                if step % 5 == 0:
                    fake_q_value = -200 + episode * 8 + np.random.randn() * 15
                    fake_loss = np.exp(-episode * 0.03) * (0.05 + np.random.rand() * 0.08)
                    
                    ddpg_wrapper.update_metrics(
                        q_value=fake_q_value,
                        loss=fake_loss
                    )
            
            # 프레임 렌더링 및 저장
            # 1. 메인 통합 비디오
            combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
            if combined_frame is not None:
                main_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            # 2. 개별 비디오들
            dqn_frame = dqn_wrapper.render()
            ddpg_frame = ddpg_wrapper.render()
            
            if dqn_frame is not None:
                dqn_writer.write(cv2.cvtColor(dqn_frame, cv2.COLOR_RGB2BGR))
            
            if ddpg_frame is not None:
                ddpg_writer.write(cv2.cvtColor(ddpg_frame, cv2.COLOR_RGB2BGR))
            
            step += 1
        
        # 에피소드 결과 출력
        print(f"  DQN: {dqn_wrapper.current_step} 스텝, 보상 {dqn_wrapper.total_reward:.1f}")
        print(f"  DDPG: {ddpg_wrapper.current_step} 스텝, 보상 {ddpg_wrapper.total_reward:.1f}")
    
    # 5. 정리
    main_writer.release()
    dqn_writer.release() 
    ddpg_writer.release()
    dqn_wrapper.close()
    ddpg_wrapper.close()
    
    print(f"\n✅ 종합 시각화 영상 생성 완료!")
    print(f"📁 저장 위치: {output_dir}")
    print(f"🎬 메인 통합 영상: comprehensive_dqn_vs_ddpg.mp4")
    print(f"🎬 DQN 개별 영상: dqn_comprehensive.mp4")
    print(f"🎬 DDPG 개별 영상: ddpg_comprehensive.mp4")
    
    # 최종 스크린샷도 저장
    save_final_screenshots(output_dir)


def save_final_screenshots(output_dir: Path):
    """최종 스크린샷 저장"""
    print("\n📸 최종 스크린샷 생성 중...")
    
    # 새로운 환경으로 스크린샷 생성
    dqn_env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn_wrapper = ComprehensiveVisualizationWrapper(dqn_env, algorithm_name="DQN")
    
    ddpg_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    ddpg_wrapper = ComprehensiveVisualizationWrapper(ddpg_env, algorithm_name="DDPG")
    
    # 몇 에피소드 실행하여 데이터 축적
    for episode in range(10):
        for wrapper in [dqn_wrapper, ddpg_wrapper]:
            wrapper.reset()
            done = False
            step = 0
            
            while not done and step < 200:
                action = wrapper.action_space.sample()
                _, _, terminated, truncated, _ = wrapper.step(action)
                done = terminated or truncated
                step += 1
                
                # 메트릭 업데이트
                if step % 10 == 0:
                    if wrapper.algorithm_name == "DQN":
                        wrapper.update_metrics(
                            q_value=np.random.randn() * 5 + episode,
                            loss=np.exp(-episode * 0.1) * 0.1
                        )
                    else:
                        wrapper.update_metrics(
                            q_value=-150 + episode * 10 + np.random.randn() * 20,
                            loss=np.exp(-episode * 0.05) * 0.05
                        )
    
    # 스크린샷 저장
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    # 개별 스크린샷
    dqn_frame = dqn_wrapper.render()
    ddpg_frame = ddpg_wrapper.render()
    
    if dqn_frame is not None:
        cv2.imwrite(str(screenshots_dir / "dqn_comprehensive_final.png"),
                   cv2.cvtColor(dqn_frame, cv2.COLOR_RGB2BGR))
    
    if ddpg_frame is not None:
        cv2.imwrite(str(screenshots_dir / "ddpg_comprehensive_final.png"),
                   cv2.cvtColor(ddpg_frame, cv2.COLOR_RGB2BGR))
    
    # 통합 비교 스크린샷
    combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
    if combined_frame is not None:
        cv2.imwrite(str(screenshots_dir / "comprehensive_comparison_final.png"),
                   cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
    
    dqn_wrapper.close()
    ddpg_wrapper.close()
    
    print(f"📸 스크린샷 저장 완료: {screenshots_dir}")


def create_with_real_agents():
    """실제 에이전트와 함께 종합 시각화 생성"""
    print("\n=== 실제 에이전트와 함께 종합 시각화 ===")
    
    output_dir = Path("videos/comprehensive_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DQN 환경 및 에이전트
    dqn_env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn_wrapper = ComprehensiveVisualizationWrapper(dqn_env, algorithm_name="DQN")
    
    dqn_agent = DQNAgent(
        state_dim=dqn_env.observation_space.shape[0],
        action_dim=dqn_env.action_space.n,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # DDPG 환경 및 에이전트  
    ddpg_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    ddpg_wrapper = ComprehensiveVisualizationWrapper(ddpg_env, algorithm_name="DDPG")
    
    ddpg_agent = DDPGAgent(
        state_dim=ddpg_env.observation_space.shape[0],
        action_dim=ddpg_env.action_space.shape[0],
        max_action=float(ddpg_env.action_space.high[0]),
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005
    )
    
    # 비디오 설정
    output_path = output_dir / "comprehensive_with_real_agents.mp4"
    
    combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, 
                                  (combined_frame.shape[1], combined_frame.shape[0]))
    
    print(f"실제 에이전트 비디오: {output_path}")
    
    # 학습 및 녹화
    for episode in range(15):
        print(f"\n에피소드 {episode + 1}/15")
        
        # DQN 에피소드
        dqn_state, _ = dqn_wrapper.reset()
        dqn_done = False
        
        while not dqn_done:
            dqn_action = dqn_agent.act(dqn_state)
            dqn_next_state, dqn_reward, dqn_terminated, dqn_truncated, _ = dqn_wrapper.step(dqn_action)
            dqn_done = dqn_terminated or dqn_truncated
            
            # 학습
            dqn_agent.remember(dqn_state, dqn_action, dqn_reward, dqn_next_state, dqn_done)
            if len(dqn_agent.memory) > 32:
                loss = dqn_agent.replay()
                q_values = dqn_agent.q_network(dqn_agent._to_tensor(dqn_state))
                max_q = q_values.max().item()
                
                dqn_wrapper.update_metrics(
                    q_value=max_q,
                    loss=loss,
                    exploration_rate=dqn_agent.epsilon
                )
            
            dqn_state = dqn_next_state
        
        # DDPG 에피소드
        ddpg_state, _ = ddpg_wrapper.reset()
        ddpg_done = False
        step = 0
        
        while not ddpg_done and step < 200:
            ddpg_action = ddpg_agent.act(ddpg_state)
            ddpg_next_state, ddpg_reward, ddpg_terminated, ddpg_truncated, _ = ddpg_wrapper.step(ddpg_action)
            ddpg_done = ddpg_terminated or ddpg_truncated
            
            # 학습
            ddpg_agent.remember(ddpg_state, ddpg_action, ddpg_reward, ddpg_next_state, ddpg_done)
            if len(ddpg_agent.replay_buffer) > 100:
                actor_loss, critic_loss = ddpg_agent.train()
                q_value = ddpg_agent.critic(ddpg_agent._to_tensor(ddpg_state), 
                                          ddpg_agent._to_tensor(ddpg_action)).item()
                
                ddpg_wrapper.update_metrics(
                    q_value=q_value,
                    loss=critic_loss
                )
            
            ddpg_state = ddpg_next_state
            step += 1
        
        # 프레임 저장 (매 에피소드마다 몇 프레임)
        for _ in range(30):  # 1초분의 프레임
            combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
            if combined_frame is not None:
                video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        
        # 타겟 네트워크 업데이트
        if episode % 10 == 0:
            dqn_agent.update_target_network()
        
        print(f"  DQN: {dqn_wrapper.total_reward:.1f}, DDPG: {ddpg_wrapper.total_reward:.1f}")
    
    video_writer.release()
    dqn_wrapper.close()
    ddpg_wrapper.close()
    
    print(f"\n✅ 실제 에이전트 종합 시각화 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="종합 시각화 통합영상 생성")
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'real_agents', 'both'],
                       help='생성 모드 선택')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        create_comprehensive_demo_video()
    elif args.mode == 'real_agents':
        create_with_real_agents()
    elif args.mode == 'both':
        create_comprehensive_demo_video()
        create_with_real_agents()
    
    print(f"\n🎉 모든 종합 시각화 영상 생성 완료!")
    print(f"📁 videos/comprehensive_visualization/ 폴더를 확인하세요.")