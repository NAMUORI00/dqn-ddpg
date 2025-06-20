"""
학습과 동기화된 실시간 게임플레이 영상 생성

학습 그래프의 각 에피소드에 정확히 대응하는 게임플레이 영상을 
실시간으로 생성하여 통합 비디오를 만듭니다.
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import gc

# 프로젝트 루트 추가 (scripts/video/comparison에서 루트로)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.core.utils import set_seed

# 작업 디렉토리를 루트로 변경
os.chdir(project_root)


class SynchronizedTrainingVideoGenerator:
    """학습과 동기화된 비디오 생성기"""
    
    def __init__(self, output_dir: str = "videos/synchronized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 색상 설정
        self.colors = {
            'dqn': '#3498db',
            'ddpg': '#e74c3c',
            'background': '#2c3e50',
            'grid': '#34495e',
            'text': '#ecf0f1',
            'success': '#27ae60',
            'failure': '#c0392b'
        }
        
        # 비디오 설정
        self.fps = 30
        self.resolution = (1920, 1080)
        
        # 녹화할 에피소드 설정
        self.record_episodes = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
        # 프레임 버퍼
        self.episode_frames = {
            'dqn': {},
            'ddpg': {}
        }
        
    def train_and_record_cartpole(self):
        """CartPole 환경에서 학습하며 녹화"""
        print("\n=== CartPole 환경 학습 및 녹화 시작 ===")
        
        # 환경 생성
        dqn_env = gym.make('CartPole-v1', render_mode='rgb_array')
        
        # 연속 행동 공간을 위한 CartPole 래퍼
        class ContinuousCartPole(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                
            def step(self, action):
                # 연속 행동을 이산 행동으로 변환
                discrete_action = 0 if action[0] < 0 else 1
                return self.env.step(discrete_action)
        
        ddpg_env = ContinuousCartPole(gym.make('CartPole-v1'))
        
        # 에이전트 생성
        state_dim = dqn_env.observation_space.shape[0]
        dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=2,
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=64
        )
        
        ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=1,
            action_bound=1.0,
            actor_lr=1e-4,
            critic_lr=1e-3,
            buffer_size=50000,
            batch_size=64
        )
        
        # 학습 데이터 저장
        training_data = {
            'dqn': {'rewards': [], 'lengths': []},
            'ddpg': {'rewards': [], 'lengths': []}
        }
        
        # DQN 학습 및 녹화
        print("\n[DQN] 학습 시작...")
        for episode in range(1, 501):
            state, _ = dqn_env.reset()
            frames = []
            total_reward = 0
            steps = 0
            
            # 이 에피소드를 녹화해야 하는지 확인
            should_record = episode in self.record_episodes
            
            while True:
                if should_record:
                    # 환경 렌더링
                    frame = dqn_env.render()
                    if frame is not None:
                        frames.append(frame)
                
                # 행동 선택
                action = dqn_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = dqn_env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                dqn_agent.store_transition(state, action, reward, next_state, done)
                
                # 학습
                if len(dqn_agent.buffer) > dqn_agent.batch_size:
                    dqn_agent.update()
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # 결과 저장
            training_data['dqn']['rewards'].append(total_reward)
            training_data['dqn']['lengths'].append(steps)
            
            if should_record and frames:
                self.episode_frames['dqn'][episode] = frames
                print(f"[DQN] 에피소드 {episode}: 보상={total_reward:.1f}, 길이={steps}, 프레임={len(frames)}")
            elif episode % 50 == 0:
                print(f"[DQN] 에피소드 {episode}: 보상={total_reward:.1f}")
        
        # DDPG 학습 및 녹화
        print("\n[DDPG] 학습 시작...")
        for episode in range(1, 501):
            state, _ = ddpg_env.reset()
            frames = []
            total_reward = 0
            steps = 0
            
            should_record = episode in self.record_episodes
            
            while True:
                if should_record:
                    # 수동 렌더링 (연속 행동 공간이므로)
                    frame = self._render_cartpole_state(ddpg_env, state)
                    frames.append(frame)
                
                # 행동 선택
                action = ddpg_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = ddpg_env.step(action)
                done = terminated or truncated
                
                # 경험 저장
                ddpg_agent.store_transition(state, action, reward, next_state, done)
                
                # 학습
                if len(ddpg_agent.buffer) > ddpg_agent.batch_size:
                    ddpg_agent.update()
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # 결과 저장
            training_data['ddpg']['rewards'].append(total_reward)
            training_data['ddpg']['lengths'].append(steps)
            
            if should_record and frames:
                self.episode_frames['ddpg'][episode] = frames
                print(f"[DDPG] 에피소드 {episode}: 보상={total_reward:.1f}, 길이={steps}, 프레임={len(frames)}")
            elif episode % 50 == 0:
                print(f"[DDPG] 에피소드 {episode}: 보상={total_reward:.1f}")
        
        dqn_env.close()
        ddpg_env.close()
        
        return training_data
    
    def train_and_record_pendulum(self):
        """Pendulum 환경에서 학습하며 녹화"""
        print("\n=== Pendulum 환경 학습 및 녹화 시작 ===")
        
        # 환경 생성
        ddpg_env = gym.make('Pendulum-v1', render_mode='rgb_array')
        
        # DQN을 위한 이산화된 Pendulum
        class DiscretePendulum(gym.Wrapper):
            def __init__(self, env, num_actions=11):
                super().__init__(env)
                self.num_actions = num_actions
                self.action_space = gym.spaces.Discrete(num_actions)
                self.action_mapping = np.linspace(-2.0, 2.0, num_actions)
                
            def step(self, action):
                continuous_action = np.array([self.action_mapping[action]])
                return self.env.step(continuous_action)
        
        dqn_env = DiscretePendulum(gym.make('Pendulum-v1', render_mode='rgb_array'))
        
        # 에이전트 생성
        state_dim = 3  # Pendulum state dimension
        dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=11,
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=64
        )
        
        ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=1,
            action_bound=2.0,
            actor_lr=1e-4,
            critic_lr=1e-3,
            buffer_size=50000,
            batch_size=64
        )
        
        # 학습 데이터 저장
        training_data = {
            'dqn': {'rewards': [], 'lengths': []},
            'ddpg': {'rewards': [], 'lengths': []}
        }
        
        # 녹화할 에피소드 조정 (Pendulum은 300 에피소드)
        pendulum_record_episodes = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        
        # DQN 학습 및 녹화
        print("\n[DQN] 학습 시작...")
        for episode in range(1, 301):
            state, _ = dqn_env.reset()
            frames = []
            total_reward = 0
            steps = 0
            
            should_record = episode in pendulum_record_episodes
            
            for _ in range(200):  # Pendulum은 최대 200 스텝
                if should_record:
                    frame = dqn_env.render()
                    if frame is not None:
                        frames.append(frame)
                
                action = dqn_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = dqn_env.step(action)
                done = terminated or truncated
                
                dqn_agent.store_transition(state, action, reward, next_state, done)
                
                if len(dqn_agent.buffer) > dqn_agent.batch_size:
                    dqn_agent.update()
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            training_data['dqn']['rewards'].append(total_reward)
            training_data['dqn']['lengths'].append(steps)
            
            if should_record and frames:
                self.episode_frames['dqn'][episode] = frames
                print(f"[DQN] 에피소드 {episode}: 보상={total_reward:.1f}, 프레임={len(frames)}")
            elif episode % 50 == 0:
                print(f"[DQN] 에피소드 {episode}: 보상={total_reward:.1f}")
        
        # DDPG 학습 및 녹화
        print("\n[DDPG] 학습 시작...")
        for episode in range(1, 301):
            state, _ = ddpg_env.reset()
            frames = []
            total_reward = 0
            steps = 0
            
            should_record = episode in pendulum_record_episodes
            
            for _ in range(200):
                if should_record:
                    frame = ddpg_env.render()
                    if frame is not None:
                        frames.append(frame)
                
                action = ddpg_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = ddpg_env.step(action)
                done = terminated or truncated
                
                ddpg_agent.store_transition(state, action, reward, next_state, done)
                
                if len(ddpg_agent.buffer) > ddpg_agent.batch_size:
                    ddpg_agent.update()
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            training_data['ddpg']['rewards'].append(total_reward)
            training_data['ddpg']['lengths'].append(steps)
            
            if should_record and frames:
                self.episode_frames['ddpg'][episode] = frames
                print(f"[DDPG] 에피소드 {episode}: 보상={total_reward:.1f}, 프레임={len(frames)}")
            elif episode % 50 == 0:
                print(f"[DDPG] 에피소드 {episode}: 보상={total_reward:.1f}")
        
        dqn_env.close()
        ddpg_env.close()
        
        return training_data
    
    def _render_cartpole_state(self, env, state):
        """CartPole 상태를 수동으로 렌더링"""
        # 간단한 시각화 (실제 물리 상태 기반)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 2.0)
        
        cart_pos = state[0]
        pole_angle = state[2]
        
        # 카트 그리기
        cart_width = 0.5
        cart_height = 0.3
        cart = plt.Rectangle((cart_pos - cart_width/2, 0), cart_width, cart_height, 
                           fill=True, color='blue')
        ax.add_patch(cart)
        
        # 폴 그리기
        pole_length = 1.0
        pole_end_x = cart_pos + pole_length * np.sin(pole_angle)
        pole_end_y = cart_height + pole_length * np.cos(pole_angle)
        ax.plot([cart_pos, pole_end_x], [cart_height, pole_end_y], 'r-', linewidth=5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # matplotlib figure를 numpy 배열로 변환
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
    
    def create_synchronized_video(self, env_type: str, training_data: Dict, duration: int = 60):
        """동기화된 비디오 생성"""
        print(f"\n=== {env_type.upper()} 동기화 비디오 생성 시작 ===")
        
        # matplotlib 설정
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 9), facecolor=self.colors['background'])
        
        # 2x2 레이아웃
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                             left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        ax_dqn_graph = fig.add_subplot(gs[0, 0])
        ax_ddpg_graph = fig.add_subplot(gs[0, 1])
        ax_dqn_game = fig.add_subplot(gs[1, 0])
        ax_ddpg_game = fig.add_subplot(gs[1, 1])
        
        # 제목
        env_title = "CartPole-v1" if env_type == "cartpole" else "Pendulum-v1"
        fig.suptitle(f"{env_title}: Synchronized Learning & Gameplay", 
                    fontsize=20, color=self.colors['text'])
        
        # 비디오 설정
        output_path = self.output_dir / f"{env_type}_synchronized.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, self.resolution)
        
        # 데이터 준비
        dqn_rewards = training_data['dqn']['rewards']
        ddpg_rewards = training_data['ddpg']['rewards']
        max_episodes = len(dqn_rewards)
        
        total_frames = self.fps * duration
        episodes_per_frame = max_episodes / total_frames
        
        print(f"총 {total_frames} 프레임 생성 중...")
        
        for frame_idx in range(total_frames):
            current_episode = int(frame_idx * episodes_per_frame) + 1
            
            # 학습 그래프 업데이트
            self._update_learning_graphs(
                ax_dqn_graph, ax_ddpg_graph,
                dqn_rewards, ddpg_rewards,
                current_episode, env_type
            )
            
            # 게임플레이 프레임 업데이트
            self._update_synchronized_gameplay(
                ax_dqn_game, ax_ddpg_game,
                current_episode, env_type
            )
            
            # Figure를 이미지로 변환
            fig.canvas.draw()
            try:
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            except AttributeError:
                img = np.array(fig.canvas.buffer_rgba())
                img = img[:, :, :3]
            
            if len(img.shape) == 1:
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # BGR 변환 및 리사이즈
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, self.resolution)
            
            out.write(img_resized)
            
            if frame_idx % 30 == 0:
                print(f"진행률: {frame_idx/total_frames*100:.1f}%")
        
        out.release()
        plt.close(fig)
        
        print(f"\n비디오 저장 완료: {output_path}")
        print(f"파일 크기: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        return str(output_path)
    
    def _update_learning_graphs(self, ax_dqn, ax_ddpg, dqn_rewards, ddpg_rewards, 
                               current_episode, env_type):
        """학습 그래프 업데이트"""
        # DQN 그래프
        ax_dqn.clear()
        if current_episode > 0:
            episodes = range(1, min(current_episode + 1, len(dqn_rewards) + 1))
            rewards_to_show = dqn_rewards[:current_episode]
            
            ax_dqn.plot(episodes, rewards_to_show, 
                       color=self.colors['dqn'], linewidth=2, alpha=0.8)
            
            # 이동평균
            if len(rewards_to_show) > 20:
                window = 20
                ma = np.convolve(rewards_to_show, np.ones(window)/window, mode='valid')
                ax_dqn.plot(range(window, current_episode + 1), ma,
                          color=self.colors['dqn'], linewidth=3)
            
            # 현재 에피소드 표시
            if current_episode <= len(dqn_rewards):
                ax_dqn.scatter([current_episode], [dqn_rewards[current_episode-1]], 
                             color=self.colors['success'], s=100, zorder=5)
        
        ax_dqn.set_title('DQN Learning Progress', color=self.colors['text'], fontsize=14)
        ax_dqn.set_xlabel('Episode', color=self.colors['text'])
        ax_dqn.set_ylabel('Reward', color=self.colors['text'])
        ax_dqn.grid(True, alpha=0.3)
        
        # DDPG 그래프
        ax_ddpg.clear()
        if current_episode > 0:
            episodes = range(1, min(current_episode + 1, len(ddpg_rewards) + 1))
            rewards_to_show = ddpg_rewards[:current_episode]
            
            ax_ddpg.plot(episodes, rewards_to_show,
                        color=self.colors['ddpg'], linewidth=2, alpha=0.8)
            
            if len(rewards_to_show) > 20:
                window = 20
                ma = np.convolve(rewards_to_show, np.ones(window)/window, mode='valid')
                ax_ddpg.plot(range(window, current_episode + 1), ma,
                           color=self.colors['ddpg'], linewidth=3)
            
            if current_episode <= len(ddpg_rewards):
                ax_ddpg.scatter([current_episode], [ddpg_rewards[current_episode-1]],
                              color=self.colors['success'], s=100, zorder=5)
        
        ax_ddpg.set_title('DDPG Learning Progress', color=self.colors['text'], fontsize=14)
        ax_ddpg.set_xlabel('Episode', color=self.colors['text'])
        ax_ddpg.set_ylabel('Reward', color=self.colors['text'])
        ax_ddpg.grid(True, alpha=0.3)
        
        # Y축 범위 설정
        if env_type == "cartpole":
            ax_dqn.set_ylim(0, 550)
            ax_ddpg.set_ylim(0, 550)
        else:
            ax_dqn.set_ylim(-1800, 0)
            ax_ddpg.set_ylim(-1800, 0)
    
    def _update_synchronized_gameplay(self, ax_dqn, ax_ddpg, current_episode, env_type):
        """동기화된 게임플레이 프레임 업데이트"""
        # 가장 가까운 녹화된 에피소드 찾기
        def find_nearest_recorded_episode(episode, recorded_episodes):
            if episode in recorded_episodes:
                return episode
            
            # 가장 가까운 이전 에피소드 찾기
            prev_episodes = [e for e in recorded_episodes if e <= episode]
            if prev_episodes:
                return max(prev_episodes)
            
            # 없으면 첫 번째 에피소드
            return min(recorded_episodes) if recorded_episodes else 1
        
        # DQN 게임플레이
        ax_dqn.clear()
        ax_dqn.axis('off')
        
        dqn_recorded = list(self.episode_frames['dqn'].keys())
        if dqn_recorded:
            nearest_dqn = find_nearest_recorded_episode(current_episode, dqn_recorded)
            frames = self.episode_frames['dqn'][nearest_dqn]
            
            if frames:
                # 프레임 인덱스 계산 (반복 재생)
                frame_idx = int((current_episode - nearest_dqn) * 10) % len(frames)
                frame = frames[min(frame_idx, len(frames) - 1)]
                
                ax_dqn.imshow(frame)
                ax_dqn.set_title(f'DQN Gameplay - Episode {current_episode} (from ep. {nearest_dqn})',
                               color=self.colors['text'], fontsize=12)
        else:
            ax_dqn.text(0.5, 0.5, f'DQN\nEpisode {current_episode}',
                       ha='center', va='center', transform=ax_dqn.transAxes,
                       fontsize=16, color=self.colors['dqn'])
        
        # DDPG 게임플레이
        ax_ddpg.clear()
        ax_ddpg.axis('off')
        
        ddpg_recorded = list(self.episode_frames['ddpg'].keys())
        if ddpg_recorded:
            nearest_ddpg = find_nearest_recorded_episode(current_episode, ddpg_recorded)
            frames = self.episode_frames['ddpg'][nearest_ddpg]
            
            if frames:
                frame_idx = int((current_episode - nearest_ddpg) * 10) % len(frames)
                frame = frames[min(frame_idx, len(frames) - 1)]
                
                ax_ddpg.imshow(frame)
                ax_ddpg.set_title(f'DDPG Gameplay - Episode {current_episode} (from ep. {nearest_ddpg})',
                                color=self.colors['text'], fontsize=12)
        else:
            ax_ddpg.text(0.5, 0.5, f'DDPG\nEpisode {current_episode}',
                        ha='center', va='center', transform=ax_ddpg.transAxes,
                        fontsize=16, color=self.colors['ddpg'])


def main():
    parser = argparse.ArgumentParser(description="학습과 동기화된 게임플레이 비디오 생성")
    
    parser.add_argument("--cartpole", action="store_true", help="CartPole 비디오 생성")
    parser.add_argument("--pendulum", action="store_true", help="Pendulum 비디오 생성")
    parser.add_argument("--all", action="store_true", help="모든 환경 비디오 생성")
    parser.add_argument("--duration", type=int, default=60, help="비디오 길이 (초)")
    parser.add_argument("--output-dir", type=str, default="videos/synchronized",
                       help="출력 디렉토리")
    
    args = parser.parse_args()
    
    if not args.cartpole and not args.pendulum and not args.all:
        args.all = True
    
    # 랜덤 시드 설정
    set_seed(42)
    
    generator = SynchronizedTrainingVideoGenerator(args.output_dir)
    
    print("="*60)
    print("🎮 Synchronized Training & Gameplay Video Generator")
    print("="*60)
    
    videos_created = []
    
    if args.all or args.cartpole:
        # CartPole 학습 및 녹화
        training_data = generator.train_and_record_cartpole()
        
        # 비디오 생성
        video_path = generator.create_synchronized_video("cartpole", training_data, args.duration)
        videos_created.append(video_path)
        
        # 메모리 정리
        generator.episode_frames = {'dqn': {}, 'ddpg': {}}
        gc.collect()
    
    if args.all or args.pendulum:
        # Pendulum 학습 및 녹화
        training_data = generator.train_and_record_pendulum()
        
        # 비디오 생성
        video_path = generator.create_synchronized_video("pendulum", training_data, args.duration)
        videos_created.append(video_path)
    
    print("\n" + "="*60)
    print("✅ 동기화된 비디오 생성 완료!")
    print("="*60)
    print("\n생성된 비디오:")
    for video in videos_created:
        print(f"  - {video}")


if __name__ == "__main__":
    main()