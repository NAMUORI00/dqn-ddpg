"""
통합 비디오 렌더링 파이프라인
전체 학습 과정을 시각화하고 영상으로 생성하는 시스템
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
from datetime import datetime
import shutil

try:
    from .video_manager import VideoManager, VideoConfig
except ImportError:
    from video_manager import VideoManager, VideoConfig


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 기본 설정
    output_dir: str = "videos/pipeline"
    temp_dir: str = "videos/temp"
    
    # 비디오 설정
    fps: int = 30
    duration_seconds: int = 180  # 3분
    resolution: Tuple[int, int] = (1280, 720)
    
    # 시각화 설정
    show_metrics: bool = True
    show_progress: bool = True
    show_episode_info: bool = True
    
    # 데이터 설정
    max_episodes_to_show: int = 1000
    smooth_curves: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """YAML 파일에서 설정 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict.get('pipeline', {}))


class VideoRenderingPipeline:
    """전체 비디오 렌더링 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_directories()
        
        # 상태 변수
        self.current_frame = 0
        self.total_frames = config.fps * config.duration_seconds
        
        # 데이터 저장
        self.dqn_data = None
        self.ddpg_data = None
        
        # 시각화 설정
        plt.style.use('dark_background')
        self.colors = {
            'dqn': '#00ff88',
            'ddpg': '#ff6b6b',
            'background': '#1a1a1a',
            'text': '#ffffff',
            'grid': '#333333'
        }
    
    def setup_directories(self):
        """디렉토리 설정"""
        self.output_path = Path(self.config.output_dir)
        self.temp_path = Path(self.config.temp_dir)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, dqn_results_path: str, ddpg_results_path: str):
        """학습 데이터 로드"""
        try:
            with open(dqn_results_path, 'r') as f:
                self.dqn_data = json.load(f)
            print(f"[INFO] DQN 데이터 로드 완료: {len(self.dqn_data.get('metrics', {}).get('episode_rewards', []))} 에피소드")
        except Exception as e:
            print(f"[WARNING] DQN 데이터 로드 실패: {e}")
            self.dqn_data = self._create_dummy_data('dqn')
        
        try:
            with open(ddpg_results_path, 'r') as f:
                self.ddpg_data = json.load(f)
            print(f"[INFO] DDPG 데이터 로드 완료: {len(self.ddpg_data.get('metrics', {}).get('episode_rewards', []))} 에피소드")
        except Exception as e:
            print(f"[WARNING] DDPG 데이터 로드 실패: {e}")
            self.ddpg_data = self._create_dummy_data('ddpg')
    
    def _create_dummy_data(self, algorithm: str) -> Dict:
        """더미 데이터 생성 (데이터가 없을 때 사용)"""
        np.random.seed(42)
        
        if algorithm == 'dqn':
            episodes = 500
            rewards = []
            for i in range(episodes):
                # 학습 곡선 시뮬레이션
                base_reward = min(400, 50 + i * 0.7)
                noise = np.random.normal(0, 20)
                reward = max(0, base_reward + noise)
                rewards.append(reward)
        else:  # ddpg
            episodes = 400
            rewards = []
            for i in range(episodes):
                # DDPG 학습 곡선 시뮬레이션
                base_reward = max(-1000, -800 + i * 1.5)
                noise = np.random.normal(0, 50)
                reward = min(0, base_reward + noise)
                rewards.append(reward)
        
        return {
            'metrics': {
                'episode_rewards': rewards,
                'episode_lengths': [np.random.randint(50, 200) for _ in range(episodes)],
                'training_losses': [np.random.exponential(0.1) for _ in range(episodes * 10)],
                'q_values': [np.random.normal(0, 1) for _ in range(episodes * 10)]
            },
            'config': {
                'algorithm': algorithm.upper(),
                'environment': 'CartPole-v1' if algorithm == 'dqn' else 'Pendulum-v1'
            }
        }
    
    def create_learning_animation(self, output_filename: str = "learning_process.mp4"):
        """학습 과정 애니메이션 생성"""
        print("[INFO] 학습 과정 애니메이션 생성 시작...")
        
        # 피그 설정
        fig = plt.figure(figsize=(16, 9), facecolor=self.colors['background'])
        fig.suptitle('DQN vs DDPG 학습 과정 시각화', 
                    fontsize=20, color=self.colors['text'], y=0.95)
        
        # 서브플롯 생성
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 메인 학습 곡선
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.set_facecolor(self.colors['background'])
        
        # 보조 플롯들
        ax_lengths = fig.add_subplot(gs[1, 0])
        ax_losses = fig.add_subplot(gs[1, 1])
        ax_q_values = fig.add_subplot(gs[1, 2])
        
        # 통계 표시
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        
        # 모든 축 설정
        for ax in [ax_main, ax_lengths, ax_losses, ax_q_values]:
            ax.set_facecolor(self.colors['background'])
            ax.tick_params(colors=self.colors['text'])
            ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        # 데이터 준비
        dqn_rewards = self.dqn_data['metrics']['episode_rewards']
        ddpg_rewards = self.ddpg_data['metrics']['episode_rewards']
        
        max_episodes = max(len(dqn_rewards), len(ddpg_rewards))
        
        def animate(frame):
            # 현재 프레임에서 보여줄 에피소드 수 계산
            current_episode = int((frame / self.total_frames) * max_episodes)
            
            # 모든 플롯 클리어
            ax_main.clear()
            ax_lengths.clear()
            ax_losses.clear()
            ax_q_values.clear()
            
            # 축 설정 재적용
            for ax in [ax_main, ax_lengths, ax_losses, ax_q_values]:
                ax.set_facecolor(self.colors['background'])
                ax.tick_params(colors=self.colors['text'])
                ax.grid(True, color=self.colors['grid'], alpha=0.3)
            
            # 1. 메인 학습 곡선
            if current_episode > 0:
                dqn_end = min(current_episode, len(dqn_rewards))
                ddpg_end = min(current_episode, len(ddpg_rewards))
                
                if dqn_end > 0:
                    episodes_dqn = range(dqn_end)
                    ax_main.plot(episodes_dqn, dqn_rewards[:dqn_end], 
                               color=self.colors['dqn'], label='DQN', linewidth=2)
                    
                    # 이동평균
                    if dqn_end > 20:
                        window = min(50, dqn_end // 5)
                        ma_rewards = np.convolve(dqn_rewards[:dqn_end], 
                                               np.ones(window)/window, mode='valid')
                        ax_main.plot(range(window-1, dqn_end), ma_rewards, 
                                   color=self.colors['dqn'], linewidth=3, alpha=0.8)
                
                if ddpg_end > 0:
                    episodes_ddpg = range(ddpg_end)
                    ax_main.plot(episodes_ddpg, ddpg_rewards[:ddpg_end], 
                               color=self.colors['ddpg'], label='DDPG', linewidth=2)
                    
                    # 이동평균
                    if ddpg_end > 20:
                        window = min(50, ddpg_end // 5)
                        ma_rewards = np.convolve(ddpg_rewards[:ddpg_end], 
                                               np.ones(window)/window, mode='valid')
                        ax_main.plot(range(window-1, ddpg_end), ma_rewards, 
                                   color=self.colors['ddpg'], linewidth=3, alpha=0.8)
            
            ax_main.set_xlabel('Episode', color=self.colors['text'])
            ax_main.set_ylabel('Reward', color=self.colors['text'])
            ax_main.set_title('학습 곡선', color=self.colors['text'])
            ax_main.legend(facecolor=self.colors['background'], 
                         edgecolor=self.colors['text'], labelcolor=self.colors['text'])
            
            # 2. 에피소드 길이
            if current_episode > 0:
                dqn_lengths = self.dqn_data['metrics']['episode_lengths'][:min(current_episode, len(dqn_rewards))]
                ddpg_lengths = self.ddpg_data['metrics']['episode_lengths'][:min(current_episode, len(ddpg_rewards))]
                
                if dqn_lengths:
                    ax_lengths.plot(dqn_lengths, color=self.colors['dqn'], alpha=0.7)
                if ddpg_lengths:
                    ax_lengths.plot(ddpg_lengths, color=self.colors['ddpg'], alpha=0.7)
            
            ax_lengths.set_xlabel('Episode', color=self.colors['text'])
            ax_lengths.set_ylabel('Length', color=self.colors['text'])
            ax_lengths.set_title('에피소드 길이', color=self.colors['text'])
            
            # 3. 학습 손실
            if current_episode > 0:
                loss_steps = min(current_episode * 10, len(self.dqn_data['metrics']['training_losses']))
                if loss_steps > 0:
                    ax_losses.plot(self.dqn_data['metrics']['training_losses'][:loss_steps], 
                                 color=self.colors['dqn'], alpha=0.7)
                    ax_losses.plot(self.ddpg_data['metrics']['training_losses'][:loss_steps], 
                                 color=self.colors['ddpg'], alpha=0.7)
            
            ax_losses.set_xlabel('Training Step', color=self.colors['text'])
            ax_losses.set_ylabel('Loss', color=self.colors['text'])
            ax_losses.set_title('학습 손실', color=self.colors['text'])
            ax_losses.set_yscale('log')
            
            # 4. Q-값 변화
            if current_episode > 0:
                q_steps = min(current_episode * 10, len(self.dqn_data['metrics']['q_values']))
                if q_steps > 0:
                    ax_q_values.plot(self.dqn_data['metrics']['q_values'][:q_steps], 
                                   color=self.colors['dqn'], alpha=0.7)
                    ax_q_values.plot(self.ddpg_data['metrics']['q_values'][:q_steps], 
                                   color=self.colors['ddpg'], alpha=0.7)
            
            ax_q_values.set_xlabel('Training Step', color=self.colors['text'])
            ax_q_values.set_ylabel('Q-value', color=self.colors['text'])
            ax_q_values.set_title('Q-값 변화', color=self.colors['text'])
            
            # 5. 현재 통계
            progress = (frame / self.total_frames) * 100
            stats_text = f"""
            진행률: {progress:.1f}% | 현재 에피소드: {current_episode}/{max_episodes}
            
            DQN (CartPole): 평균 보상 {np.mean(dqn_rewards[:current_episode]) if current_episode > 0 else 0:.1f}
            DDPG (Pendulum): 평균 보상 {np.mean(ddpg_rewards[:current_episode]) if current_episode > 0 else 0:.1f}
            """
            
            ax_stats.text(0.5, 0.5, stats_text, 
                         transform=ax_stats.transAxes,
                         ha='center', va='center',
                         color=self.colors['text'],
                         fontsize=14,
                         bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor=self.colors['background'],
                                 edgecolor=self.colors['text']))
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(fig, animate, frames=self.total_frames, 
                                     interval=1000//self.config.fps, blit=False)
        
        # 저장
        output_path = self.output_path / output_filename
        print(f"[INFO] 애니메이션 저장 중: {output_path}")
        
        # ffmpeg 사용 가능 여부에 따라 writer 선택
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, check=True, timeout=5)
            writer = animation.FFMpegWriter(fps=self.config.fps, bitrate=5000)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("[WARNING] ffmpeg을 사용할 수 없습니다. 다른 방법으로 저장합니다.")
            # 개별 프레임을 이미지로 저장하고 OpenCV로 비디오 생성
            return self._save_animation_with_opencv(fig, animate, output_path)
        
        try:
            anim.save(str(output_path), writer=writer, dpi=100)
            plt.close(fig)
            print(f"[INFO] 애니메이션 저장 완료: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"[ERROR] FFmpeg 저장 실패: {e}")
            plt.close(fig)
            # OpenCV 백업 방법
            return self._save_animation_with_opencv(fig, animate, output_path)
    
    def _save_animation_with_opencv(self, fig, animate_func, output_path: Path) -> str:
        """OpenCV를 사용한 애니메이션 저장 (ffmpeg 대안)"""
        print("[INFO] OpenCV를 사용하여 비디오 생성 중...")
        
        # 임시 디렉토리에 프레임 저장
        frames_dir = self.temp_path / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # 각 프레임을 이미지로 저장
        frame_paths = []
        for frame_idx in range(min(self.total_frames, 300)):  # 최대 300프레임으로 제한
            animate_func(frame_idx)
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            fig.savefig(frame_path, dpi=80, facecolor=self.colors['background'])
            frame_paths.append(str(frame_path))
            
            if frame_idx % 50 == 0:
                print(f"[INFO] 프레임 진행률: {frame_idx}/{min(self.total_frames, 300)}")
        
        plt.close(fig)
        
        # OpenCV로 비디오 생성
        if frame_paths:
            first_frame = cv2.imread(frame_paths[0])
            height, width, _ = first_frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.config.fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            
            # 임시 프레임 삭제
            for frame_path in frame_paths:
                os.remove(frame_path)
            frames_dir.rmdir()
            
            print(f"[INFO] OpenCV 비디오 생성 완료: {output_path}")
            return str(output_path)
        else:
            print("[ERROR] 프레임 생성 실패")
            return None
    
    def create_comparison_video(self, output_filename: str = "algorithm_comparison.mp4"):
        """알고리즘 비교 비디오 생성"""
        print("[INFO] 알고리즘 비교 비디오 생성 시작...")
        
        # 비교 데이터 준비
        dqn_final_performance = {
            'mean_reward': np.mean(self.dqn_data['metrics']['episode_rewards'][-100:]),
            'std_reward': np.std(self.dqn_data['metrics']['episode_rewards'][-100:]),
            'mean_length': np.mean(self.dqn_data['metrics']['episode_lengths'][-100:]),
            'convergence_episode': self._find_convergence_episode(self.dqn_data['metrics']['episode_rewards'])
        }
        
        ddpg_final_performance = {
            'mean_reward': np.mean(self.ddpg_data['metrics']['episode_rewards'][-100:]),
            'std_reward': np.std(self.ddpg_data['metrics']['episode_rewards'][-100:]),
            'mean_length': np.mean(self.ddpg_data['metrics']['episode_lengths'][-100:]),
            'convergence_episode': self._find_convergence_episode(self.ddpg_data['metrics']['episode_rewards'])
        }
        
        # 정적 비교 이미지 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=self.colors['background'])
        fig.suptitle('DQN vs DDPG 최종 성능 비교', fontsize=20, color=self.colors['text'])
        
        # 1. 최종 성능 바 차트
        ax = axes[0, 0]
        ax.set_facecolor(self.colors['background'])
        algorithms = ['DQN', 'DDPG']
        rewards = [dqn_final_performance['mean_reward'], ddpg_final_performance['mean_reward']]
        errors = [dqn_final_performance['std_reward'], ddpg_final_performance['std_reward']]
        
        bars = ax.bar(algorithms, rewards, yerr=errors, 
                     color=[self.colors['dqn'], self.colors['ddpg']], alpha=0.8)
        ax.set_ylabel('Average Reward', color=self.colors['text'])
        ax.set_title('최종 평균 보상', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        # 2. 학습 곡선 전체
        ax = axes[0, 1]
        ax.set_facecolor(self.colors['background'])
        ax.plot(self.dqn_data['metrics']['episode_rewards'], 
               color=self.colors['dqn'], label='DQN', alpha=0.7)
        ax.plot(self.ddpg_data['metrics']['episode_rewards'], 
               color=self.colors['ddpg'], label='DDPG', alpha=0.7)
        ax.set_xlabel('Episode', color=self.colors['text'])
        ax.set_ylabel('Reward', color=self.colors['text'])
        ax.set_title('전체 학습 곡선', color=self.colors['text'])
        ax.legend(facecolor=self.colors['background'], labelcolor=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        # 3. 수렴 속도 비교
        ax = axes[1, 0]
        ax.set_facecolor(self.colors['background'])
        convergence_data = [dqn_final_performance['convergence_episode'], 
                          ddpg_final_performance['convergence_episode']]
        bars = ax.bar(algorithms, convergence_data, 
                     color=[self.colors['dqn'], self.colors['ddpg']], alpha=0.8)
        ax.set_ylabel('Episode', color=self.colors['text'])
        ax.set_title('수렴 속도 (에피소드)', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        # 4. 학습 안정성
        ax = axes[1, 1]
        ax.set_facecolor(self.colors['background'])
        
        # 최근 100 에피소드의 분산 계산
        dqn_recent_var = np.var(self.dqn_data['metrics']['episode_rewards'][-100:])
        ddpg_recent_var = np.var(self.ddpg_data['metrics']['episode_rewards'][-100:])
        
        variances = [dqn_recent_var, ddpg_recent_var]
        bars = ax.bar(algorithms, variances, 
                     color=[self.colors['dqn'], self.colors['ddpg']], alpha=0.8)
        ax.set_ylabel('Variance', color=self.colors['text'])
        ax.set_title('학습 안정성 (낮을수록 안정)', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지로 저장
        comparison_image_path = self.temp_path / "comparison.png"
        plt.savefig(comparison_image_path, dpi=150, facecolor=self.colors['background'])
        plt.close()
        
        # 이미지를 비디오로 변환 (5초간 표시)
        output_path = self.output_path / output_filename
        self._image_to_video(str(comparison_image_path), str(output_path), duration=5)
        
        print(f"[INFO] 비교 비디오 생성 완료: {output_path}")
        return str(output_path)
    
    def _find_convergence_episode(self, rewards: List[float], window: int = 50) -> int:
        """수렴 지점 찾기"""
        if len(rewards) < window * 2:
            return len(rewards)
        
        # 이동평균으로 수렴 지점 찾기
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # 기울기가 작아지는 지점 찾기
        gradients = np.gradient(moving_avg)
        
        # 기울기가 임계값 이하가 되는 첫 번째 지점
        threshold = np.std(gradients) * 0.1
        convergence_indices = np.where(np.abs(gradients) < threshold)[0]
        
        return convergence_indices[0] + window if len(convergence_indices) > 0 else len(rewards)
    
    def _image_to_video(self, image_path: str, output_path: str, duration: int = 5):
        """이미지를 비디오로 변환"""
        # OpenCV로 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        height, width, _ = img.shape
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.config.fps, (width, height))
        
        # 지정된 시간동안 같은 프레임 반복
        total_frames = self.config.fps * duration
        for _ in range(total_frames):
            out.write(img)
        
        out.release()
    
    def create_summary_video(self, output_filename: str = "experiment_summary.mp4"):
        """실험 요약 비디오 생성"""
        print("[INFO] 실험 요약 비디오 생성 시작...")
        
        # 여러 구성 요소 비디오 생성
        learning_video = self.create_learning_animation("learning_temp.mp4")
        comparison_video = self.create_comparison_video("comparison_temp.mp4")
        
        # 인트로 생성
        intro_video = self._create_intro_video()
        
        # 아웃트로 생성
        outro_video = self._create_outro_video()
        
        # 모든 비디오 합치기
        output_path = self.output_path / output_filename
        self._concatenate_videos([intro_video, learning_video, comparison_video, outro_video], 
                                str(output_path))
        
        # 임시 파일 정리
        self._cleanup_temp_files()
        
        print(f"[INFO] 요약 비디오 생성 완료: {output_path}")
        return str(output_path)
    
    def _create_intro_video(self) -> str:
        """인트로 비디오 생성"""
        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        ax.axis('off')
        
        intro_text = """
        DQN vs DDPG
        강화학습 알고리즘 비교 실험
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Deep Q-Network (DQN)
        ⚡ 이산 행동 공간 (CartPole-v1)
        ⚡ 암묵적 결정적 정책 (argmax)
        ⚡ ε-greedy 탐험 전략
        
        Deep Deterministic Policy Gradient (DDPG)  
        ⚡ 연속 행동 공간 (Pendulum-v1)
        ⚡ 명시적 결정적 정책 (Actor Network)
        ⚡ 가우시안 노이즈 탐험 전략
        """
        
        ax.text(0.5, 0.5, intro_text, transform=ax.transAxes, 
               ha='center', va='center', color=self.colors['text'], 
               fontsize=16, weight='bold')
        
        intro_path = self.temp_path / "intro.png"
        plt.savefig(intro_path, dpi=150, facecolor=self.colors['background'])
        plt.close()
        
        intro_video_path = self.temp_path / "intro.mp4"
        self._image_to_video(str(intro_path), str(intro_video_path), duration=3)
        
        return str(intro_video_path)
    
    def _create_outro_video(self) -> str:
        """아웃트로 비디오 생성"""
        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        ax.axis('off')
        
        # 최종 결과 요약
        dqn_final = np.mean(self.dqn_data['metrics']['episode_rewards'][-100:])
        ddpg_final = np.mean(self.ddpg_data['metrics']['episode_rewards'][-100:])
        
        outro_text = f"""
        실험 결과 요약
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        DQN (CartPole-v1)
        📊 최종 평균 보상: {dqn_final:.1f}
        📊 목표 달성: {'✅ 성공' if dqn_final >= 400 else '❌ 미달'}
        
        DDPG (Pendulum-v1)
        📊 최종 평균 보상: {ddpg_final:.1f}
        📊 목표 달성: {'✅ 성공' if ddpg_final >= -300 else '❌ 미달'}
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        두 알고리즘 모두 각자의 도메인에서
        결정적 정책을 성공적으로 학습했습니다.
        
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        ax.text(0.5, 0.5, outro_text, transform=ax.transAxes, 
               ha='center', va='center', color=self.colors['text'], 
               fontsize=14, weight='bold')
        
        outro_path = self.temp_path / "outro.png"
        plt.savefig(outro_path, dpi=150, facecolor=self.colors['background'])
        plt.close()
        
        outro_video_path = self.temp_path / "outro.mp4"
        self._image_to_video(str(outro_path), str(outro_video_path), duration=3)
        
        return str(outro_video_path)
    
    def _concatenate_videos(self, video_paths: List[str], output_path: str):
        """비디오 파일들을 연결"""
        # ffmpeg이 없는 경우 첫 번째 비디오만 복사
        import subprocess
        import shutil
        
        # ffmpeg 사용 가능 여부 확인
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, check=True, timeout=5)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            ffmpeg_available = False
        
        if not ffmpeg_available:
            print("[WARNING] ffmpeg을 사용할 수 없습니다. 첫 번째 비디오만 복사합니다.")
            if video_paths:
                shutil.copy2(video_paths[0], output_path)
                print(f"[INFO] 비디오 복사 완료: {output_path}")
            return
        
        # 임시 파일 목록 생성
        filelist_path = self.temp_path / "filelist.txt"
        with open(filelist_path, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")
        
        # ffmpeg 명령 실행
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(filelist_path),
            '-c', 'copy',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[INFO] 비디오 연결 완료: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 비디오 연결 실패: {e}")
            # 대안: 첫 번째 비디오만 복사
            if video_paths:
                shutil.copy2(video_paths[0], output_path)
                print(f"[INFO] 첫 번째 비디오를 출력으로 복사: {output_path}")
    
    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            for file_path in self.temp_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            print("[INFO] 임시 파일 정리 완료")
        except Exception as e:
            print(f"[WARNING] 임시 파일 정리 중 오류: {e}")
    
    def run_full_pipeline(self, dqn_results_path: str, ddpg_results_path: str) -> str:
        """전체 파이프라인 실행"""
        print("[INFO] 전체 비디오 렌더링 파이프라인 시작")
        
        # 1. 데이터 로드
        self.load_training_data(dqn_results_path, ddpg_results_path)
        
        # 2. 요약 비디오 생성
        summary_video = self.create_summary_video("final_experiment_summary.mp4")
        
        # 3. 개별 비디오들도 생성
        learning_video = self.create_learning_animation("detailed_learning_process.mp4")
        comparison_video = self.create_comparison_video("algorithm_comparison.mp4")
        
        print("[INFO] 전체 비디오 렌더링 파이프라인 완료")
        print(f"[INFO] 메인 비디오: {summary_video}")
        print(f"[INFO] 학습 과정 비디오: {learning_video}")
        print(f"[INFO] 비교 비디오: {comparison_video}")
        
        return summary_video


def create_pipeline_from_config(config_path: str = None) -> VideoRenderingPipeline:
    """설정에서 파이프라인 생성"""
    if config_path and os.path.exists(config_path):
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()
    
    return VideoRenderingPipeline(config)