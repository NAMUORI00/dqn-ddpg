#!/usr/bin/env python3
"""
빠른 비디오 데모 생성기
간단한 학습 곡선 시각화 비디오를 빠르게 생성합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import json
import os
from pathlib import Path
from datetime import datetime
import sys

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "src" / "core"))
from video_utils import SampleDataGenerator, VideoEncoder, cleanup_temp_files


def create_sample_data():
    """간단한 샘플 데이터 생성"""
    dqn_data = SampleDataGenerator.create_learning_curves("dqn", 200)
    ddpg_data = SampleDataGenerator.create_learning_curves("ddpg", 200)
    
    return dqn_data['episode_rewards'], ddpg_data['episode_rewards']


def create_quick_video(output_file="quick_demo.mp4", duration=30, fps=15):
    """빠른 비디오 생성"""
    print(f"Creating quick demo video: {output_file}")
    
    # 데이터 생성
    dqn_rewards, ddpg_rewards = create_sample_data()
    
    # 영어로 된 간단한 플롯 생성
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a1a')
    
    total_frames = duration * fps
    max_episodes = max(len(dqn_rewards), len(ddpg_rewards))
    
    def animate(frame):
        # 진행률 계산
        progress = frame / total_frames
        current_episode = int(progress * max_episodes)
        
        # 왼쪽 플롯: DQN
        ax1.clear()
        ax1.set_facecolor('#1a1a1a')
        
        if current_episode > 0:
            episodes = range(min(current_episode, len(dqn_rewards)))
            rewards = dqn_rewards[:current_episode]
            
            ax1.plot(episodes, rewards, color='#00ff88', linewidth=2, label='DQN')
            
            # 이동평균
            if len(rewards) > 20:
                window = min(20, len(rewards)//5)
                ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), ma_rewards, 
                        color='#00ff88', linewidth=3, alpha=0.8)
        
        ax1.set_title('DQN (CartPole-v1)', color='white', fontsize=14)
        ax1.set_xlabel('Episode', color='white')
        ax1.set_ylabel('Reward', color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 500])
        
        # 오른쪽 플롯: DDPG
        ax2.clear()
        ax2.set_facecolor('#1a1a1a')
        
        if current_episode > 0:
            episodes = range(min(current_episode, len(ddpg_rewards)))
            rewards = ddpg_rewards[:current_episode]
            
            ax2.plot(episodes, rewards, color='#ff6b6b', linewidth=2, label='DDPG')
            
            # 이동평균
            if len(rewards) > 20:
                window = min(20, len(rewards)//5)
                ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(rewards)), ma_rewards, 
                        color='#ff6b6b', linewidth=3, alpha=0.8)
        
        ax2.set_title('DDPG (Pendulum-v1)', color='white', fontsize=14)
        ax2.set_xlabel('Episode', color='white')
        ax2.set_ylabel('Reward', color='white')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-1000, 0])
        
        # 상단 제목
        fig.suptitle(f'DQN vs DDPG Training Progress - Episode {current_episode}/{max_episodes}', 
                    color='white', fontsize=16, y=0.95)
        
        plt.tight_layout()
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                  interval=1000//fps, blit=False)
    
    # OpenCV로 저장 (ffmpeg 없이)
    print("Saving frames...")
    frames_dir = Path("temp_frames")
    frames_dir.mkdir(exist_ok=True)
    
    frame_paths = []
    for i in range(total_frames):
        animate(i)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        plt.savefig(frame_path, dpi=100, facecolor='#1a1a1a', 
                   bbox_inches='tight', pad_inches=0.1)
        frame_paths.append(str(frame_path))
        
        if i % 30 == 0:
            print(f"Progress: {i}/{total_frames} frames")
    
    plt.close(fig)
    
    # OpenCV로 비디오 생성
    print("Creating video with OpenCV...")
    success = VideoEncoder.save_frames_to_video(
        frames_dir="temp_frames",
        output_path=output_file,
        fps=fps,
        frame_pattern="frame_*.png"
    )
    
    print(f"Video created: {output_file}")
    return output_file


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick video demo generator")
    parser.add_argument("--output", default="quick_demo.mp4", help="Output video file")
    parser.add_argument("--duration", type=int, default=20, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 비디오 생성
    result = create_quick_video(args.output, args.duration, args.fps)
    
    if result and os.path.exists(result):
        file_size = os.path.getsize(result) / (1024 * 1024)
        print(f"\nSuccess! Created {result} ({file_size:.1f} MB)")
    else:
        print("\nFailed to create video")


if __name__ == "__main__":
    main()