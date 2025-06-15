"""
종합 시각화 스크린샷 생성 테스트
완성된 종합 시각화의 최종 결과물을 스크린샷으로 저장합니다.
"""

import numpy as np
import gymnasium as gym
import cv2
from pathlib import Path

from src.environments.comprehensive_visualization_wrapper import (
    ComprehensiveVisualizationWrapper,
    create_side_by_side_comparison
)


def generate_comprehensive_screenshots():
    """종합 시각화 스크린샷 생성"""
    print("=== 종합 시각화 최종 스크린샷 생성 ===")
    
    output_dir = Path("videos/comprehensive_visualization/final_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DQN 환경 설정
    dqn_env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn_wrapper = ComprehensiveVisualizationWrapper(
        dqn_env,
        algorithm_name="DQN",
        frame_width=600,
        frame_height=400,
        graph_height=300,
        stats_height=150
    )
    
    # DDPG 환경 설정
    ddpg_env = gym.make('Pendulum-v1', render_mode='rgb_array')
    ddpg_wrapper = ComprehensiveVisualizationWrapper(
        ddpg_env,
        algorithm_name="DDPG", 
        frame_width=600,
        frame_height=400,
        graph_height=300,
        stats_height=150
    )
    
    # 상호 참조 설정
    dqn_wrapper.partner_wrapper = ddpg_wrapper
    ddpg_wrapper.partner_wrapper = dqn_wrapper
    
    print("데이터 축적을 위한 에피소드 실행 중...")
    
    # 20개 에피소드 실행하여 충분한 데이터 축적
    for episode in range(20):
        print(f"  에피소드 {episode + 1}/20", end="")
        
        # DQN 에피소드
        dqn_wrapper.reset()
        dqn_done = False
        dqn_step = 0
        
        while not dqn_done and dqn_step < 300:
            dqn_action = dqn_wrapper.action_space.sample()
            _, _, dqn_terminated, dqn_truncated, _ = dqn_wrapper.step(dqn_action)
            dqn_done = dqn_terminated or dqn_truncated
            
            # 실제적인 메트릭 시뮬레이션
            if dqn_step % 10 == 0:
                # DQN Q-값: 에피소드가 진행될수록 개선
                base_q = 20 + episode * 2
                q_noise = np.random.randn() * 5
                q_value = base_q + q_noise
                
                # DQN 손실: 에피소드가 진행될수록 감소
                base_loss = np.exp(-episode * 0.08) * 0.5
                loss_noise = np.random.rand() * 0.1
                loss = base_loss + loss_noise
                
                # 탐색률: 선형 감소
                exploration = max(0.01, 1.0 - episode * 0.045)
                
                dqn_wrapper.update_metrics(
                    q_value=q_value,
                    loss=loss,
                    exploration_rate=exploration
                )
            
            dqn_step += 1
        
        # DDPG 에피소드
        ddpg_wrapper.reset()
        ddpg_done = False
        ddpg_step = 0
        
        while not ddpg_done and ddpg_step < 200:
            ddpg_action = ddpg_wrapper.action_space.sample()
            _, _, ddpg_terminated, ddpg_truncated, _ = ddpg_wrapper.step(ddpg_action)
            ddpg_done = ddpg_terminated or ddpg_truncated
            
            # DDPG 메트릭 시뮬레이션
            if ddpg_step % 10 == 0:
                # DDPG Q-값: 음수에서 시작해서 개선
                base_q = -300 + episode * 12
                q_noise = np.random.randn() * 20
                q_value = base_q + q_noise
                
                # DDPG 손실: 더 천천히 감소
                base_loss = np.exp(-episode * 0.05) * 0.2
                loss_noise = np.random.rand() * 0.05
                loss = base_loss + loss_noise
                
                ddpg_wrapper.update_metrics(
                    q_value=q_value,
                    loss=loss
                )
            
            ddpg_step += 1
        
        print(f" - DQN: {dqn_wrapper.total_reward:.1f}, DDPG: {ddpg_wrapper.total_reward:.1f}")
    
    print("\n스크린샷 생성 중...")
    
    # 1. 개별 알고리즘 스크린샷
    dqn_frame = dqn_wrapper.render()
    ddpg_frame = ddpg_wrapper.render()
    
    if dqn_frame is not None:
        cv2.imwrite(str(output_dir / "dqn_comprehensive_final.png"),
                   cv2.cvtColor(dqn_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ DQN 종합 스크린샷: {output_dir / 'dqn_comprehensive_final.png'}")
    
    if ddpg_frame is not None:
        cv2.imwrite(str(output_dir / "ddpg_comprehensive_final.png"),
                   cv2.cvtColor(ddpg_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ DDPG 종합 스크린샷: {output_dir / 'ddpg_comprehensive_final.png'}")
    
    # 2. 나란히 비교 통합 스크린샷
    combined_frame = create_side_by_side_comparison(dqn_wrapper, ddpg_wrapper)
    if combined_frame is not None:
        cv2.imwrite(str(output_dir / "comprehensive_comparison_final.png"),
                   cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        print(f"✅ 통합 비교 스크린샷: {output_dir / 'comprehensive_comparison_final.png'}")
    
    # 3. 각 구성 요소별 개별 스크린샷도 생성
    if dqn_frame is not None:
        # DQN 게임 부분만 추출
        game_part = dqn_frame[:400, :, :]
        cv2.imwrite(str(output_dir / "dqn_game_only.png"),
                   cv2.cvtColor(game_part, cv2.COLOR_RGB2BGR))
        
        # DQN 그래프 부분만 추출
        graph_part = dqn_frame[400:700, :, :]
        cv2.imwrite(str(output_dir / "dqn_graphs_only.png"),
                   cv2.cvtColor(graph_part, cv2.COLOR_RGB2BGR))
        
        # DQN 통계 부분만 추출
        stats_part = dqn_frame[700:, :, :]
        cv2.imwrite(str(output_dir / "dqn_stats_only.png"),
                   cv2.cvtColor(stats_part, cv2.COLOR_RGB2BGR))
    
    if ddpg_frame is not None:
        # DDPG 게임 부분만 추출
        game_part = ddpg_frame[:400, :, :]
        cv2.imwrite(str(output_dir / "ddpg_game_only.png"),
                   cv2.cvtColor(game_part, cv2.COLOR_RGB2BGR))
        
        # DDPG 그래프 부분만 추출  
        graph_part = ddpg_frame[400:700, :, :]
        cv2.imwrite(str(output_dir / "ddpg_graphs_only.png"),
                   cv2.cvtColor(graph_part, cv2.COLOR_RGB2BGR))
        
        # DDPG 통계 부분만 추출
        stats_part = ddpg_frame[700:, :, :]
        cv2.imwrite(str(output_dir / "ddpg_stats_only.png"),
                   cv2.cvtColor(stats_part, cv2.COLOR_RGB2BGR))
    
    # 정리
    dqn_wrapper.close()
    ddpg_wrapper.close()
    
    print(f"\n🎉 모든 종합 시각화 스크린샷 생성 완료!")
    print(f"📁 저장 위치: {output_dir}")
    print("\n📋 생성된 파일들:")
    for file_path in output_dir.glob("*.png"):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  📸 {file_path.name} ({file_size:.1f} KB)")


if __name__ == "__main__":
    generate_comprehensive_screenshots()