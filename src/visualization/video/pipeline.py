"""
비디오 렌더링 파이프라인 모듈

학습 과정을 비디오로 시각화하는 파이프라인을 제공합니다.
기존 src/core/video_pipeline.py의 기능을 모듈화하여 개선합니다.

새로운 출력 구조:
- 비디오 확장자별 자동 디렉토리 분류 (MP4 -> output/mp4/pipeline/, GIF -> output/gif/pipeline/)
- create_structured_filename()으로 일관된 비디오 파일명 생성
- 비디오 타입별 세부 카테고리 지원 (learning/, animation/, comparison/)
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video generation will be limited.")

import numpy as np
import matplotlib.pyplot as plt
try:
    import matplotlib.animation as animation
except ImportError:
    animation = None
from typing import Dict, Any, Optional, List, Tuple, Callable
import os
import json
from pathlib import Path

from ..core.base import BaseVisualizer
from ..core.utils import (
    smooth_data, validate_experiment_data, ensure_path_exists,
    get_output_path_by_extension, create_structured_filename
)


class VideoRenderingPipeline(BaseVisualizer):
    """
    비디오 렌더링 파이프라인 클래스
    
    학습 데이터를 입력받아 교육용 비디오를 생성합니다.
    샘플 데이터 지원, 다양한 비디오 형식, 실시간 진행 표시 등을 포함합니다.
    """
    
    def __init__(self, 
                 output_dir: str = "videos",
                 *args, **kwargs):
        super().__init__(output_dir=output_dir, *args, **kwargs)
        
        # 비디오 설정
        self.video_config = self.config.video
        self.fps = self.video_config.fps
        self.resolution = self.video_config.resolution
        
        # 색상 설정
        self.dqn_color = self.config.chart.dqn_color
        self.ddpg_color = self.config.chart.ddpg_color
        
        # 비디오 작성기
        self.video_writer = None
        self.current_frame = 0
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        학습 과정 비디오 생성
        
        Args:
            data: 학습 데이터 {'dqn': dqn_data, 'ddpg': ddpg_data}
            **kwargs: 비디오 생성 옵션
            
        Returns:
            생성된 비디오 파일 경로
        """
        if not self.validate_data(data):
            raise ValueError("유효하지 않은 데이터입니다")
        
        dqn_data = data.get('dqn', {})
        ddpg_data = data.get('ddpg', {})
        
        return self.create_learning_animation(
            dqn_data, ddpg_data, **kwargs
        )
    
    def create_learning_animation(self,
                                dqn_data: Dict[str, Any],
                                ddpg_data: Dict[str, Any],
                                save_filename: str = "learning_animation.mp4",
                                duration_seconds: int = 60,
                                show_progress: bool = True) -> str:
        """
        학습 과정 애니메이션 생성
        
        Args:
            dqn_data: DQN 학습 데이터
            ddpg_data: DDPG 학습 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이 (초)
            show_progress: 진행 표시 여부
            
        Returns:
            생성된 비디오 파일 경로
        """
        # 데이터 검증
        if not validate_experiment_data(dqn_data) or not validate_experiment_data(ddpg_data):
            self.logger.warning("실험 데이터가 불완전합니다. 샘플 데이터를 생성합니다.")
            dqn_data, ddpg_data = self._generate_sample_data()
        
        # 확장자 기반 출력 구조 사용 - 비디오 파일을 적절한 디렉토리에 자동 저장
        video_path = get_output_path_by_extension(save_filename, "pipeline", self.config)
        ensure_path_exists(video_path)
        
        # 애니메이션 설정
        total_frames = duration_seconds * self.fps
        max_episodes = max(
            len(dqn_data.get('episode_rewards', [])),
            len(ddpg_data.get('episode_rewards', []))
        )
        
        # 프레임당 에피소드 수
        episodes_per_frame = max(1, max_episodes // total_frames)
        
        # Figure 설정
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DQN vs DDPG Learning Progress', fontsize=20, fontweight='bold')
        
        # 초기 플롯 설정
        self._setup_animation_axes(axes, dqn_data, ddpg_data)
        
        # 애니메이션 데이터 준비
        animation_data = self._prepare_animation_data(dqn_data, ddpg_data, total_frames)
        
        try:
            # 비디오 작성기 초기화
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution
            )
            
            self.logger.info(f"비디오 생성 시작: {total_frames} 프레임, {duration_seconds}초")
            
            # 프레임별 렌더링
            for frame_idx in range(total_frames):
                self._render_frame(
                    fig, axes, animation_data, frame_idx, 
                    show_progress=show_progress
                )
                
                if show_progress and frame_idx % (total_frames // 10) == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    self.logger.info(f"진행률: {progress:.1f}%")
            
            self.logger.info("비디오 생성 완료")
            
        finally:
            # 리소스 정리
            if self.video_writer:
                self.video_writer.release()
            plt.close(fig)
            cv2.destroyAllWindows()
        
        self.log_visualization_info(
            "Learning Animation",
            {
                "duration": duration_seconds,
                "total_frames": total_frames,
                "fps": self.fps,
                "resolution": self.resolution,
                "file_size_mb": os.path.getsize(video_path) / (1024*1024)
            }
        )
        
        return video_path
    
    def _setup_animation_axes(self, axes, dqn_data: Dict, ddpg_data: Dict):
        """애니메이션 축 초기 설정"""
        
        # 1. 에피소드 보상 (좌상)
        ax = axes[0, 0]
        ax.set_title('Episode Rewards', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # 2. 에피소드 길이 (우상)
        ax = axes[0, 1]
        ax.set_title('Episode Length', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.grid(True, alpha=0.3)
        
        # 3. 학습 손실 (좌하)
        ax = axes[1, 0]
        ax.set_title('Training Loss', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. 누적 보상 (우하)
        ax = axes[1, 1]
        ax.set_title('Cumulative Reward', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True, alpha=0.3)
    
    def _prepare_animation_data(self, dqn_data: Dict, ddpg_data: Dict, total_frames: int) -> Dict:
        """애니메이션 데이터 준비"""
        
        # 에피소드 데이터
        dqn_rewards = dqn_data.get('episode_rewards', [])
        ddpg_rewards = ddpg_data.get('episode_rewards', [])
        dqn_lengths = dqn_data.get('episode_lengths', [])
        ddpg_lengths = ddpg_data.get('episode_lengths', [])
        
        # 손실 데이터
        dqn_losses = dqn_data.get('training_losses', [])
        ddpg_losses = ddpg_data.get('training_losses', [])
        
        # 최대 에피소드 수
        max_episodes = max(len(dqn_rewards), len(ddpg_rewards))
        
        # 프레임별 데이터 포인트 계산
        episodes_per_frame = max(1, max_episodes // total_frames)
        
        # 누적 보상 계산
        dqn_cumulative = np.cumsum(dqn_rewards) if dqn_rewards else []
        ddpg_cumulative = np.cumsum(ddpg_rewards) if ddpg_rewards else []
        
        return {
            'dqn_rewards': dqn_rewards,
            'ddpg_rewards': ddpg_rewards,
            'dqn_lengths': dqn_lengths,
            'ddpg_lengths': ddpg_lengths,
            'dqn_losses': dqn_losses,
            'ddpg_losses': ddpg_losses,
            'dqn_cumulative': dqn_cumulative,
            'ddpg_cumulative': ddpg_cumulative,
            'episodes_per_frame': episodes_per_frame,
            'max_episodes': max_episodes
        }
    
    def _render_frame(self, fig, axes, animation_data: Dict, frame_idx: int, show_progress: bool = True):
        """개별 프레임 렌더링"""
        
        # 현재 프레임에서 표시할 에피소드 수
        current_episode = min(
            frame_idx * animation_data['episodes_per_frame'],
            animation_data['max_episodes']
        )
        
        # 모든 축 클리어
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        # 축 재설정
        self._setup_animation_axes(axes, {}, {})
        
        # 1. 에피소드 보상 플롯
        self._plot_rewards_frame(axes[0, 0], animation_data, current_episode)
        
        # 2. 에피소드 길이 플롯
        self._plot_lengths_frame(axes[0, 1], animation_data, current_episode)
        
        # 3. 학습 손실 플롯
        self._plot_losses_frame(axes[1, 0], animation_data, current_episode)
        
        # 4. 누적 보상 플롯
        self._plot_cumulative_frame(axes[1, 1], animation_data, current_episode)
        
        # 진행 표시
        if show_progress:
            progress = frame_idx / (len(animation_data) - 1) if len(animation_data) > 1 else 1.0
            fig.suptitle(f'DQN vs DDPG Learning Progress - Episode {current_episode} ({progress:.1%})', 
                        fontsize=20, fontweight='bold')
        
        # Figure를 이미지로 변환
        fig.canvas.draw()
        
        # OpenCV 형식으로 변환
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 해상도 조정
        img = cv2.resize(img, self.resolution)
        
        # 비디오에 프레임 쓰기
        if self.video_writer:
            self.video_writer.write(img)
    
    def _plot_rewards_frame(self, ax, animation_data: Dict, current_episode: int):
        """에피소드 보상 프레임 플롯"""
        dqn_rewards = animation_data['dqn_rewards'][:current_episode]
        ddpg_rewards = animation_data['ddpg_rewards'][:current_episode]
        
        if dqn_rewards:
            episodes = range(len(dqn_rewards))
            ax.plot(episodes, dqn_rewards, label='DQN', color=self.dqn_color, alpha=0.7)
            
            # 스무딩된 곡선
            if len(dqn_rewards) > 20:
                smoothed = smooth_data(dqn_rewards, min(20, len(dqn_rewards)//4))
                smooth_episodes = range(19, len(dqn_rewards))
                ax.plot(smooth_episodes, smoothed, color=self.dqn_color, linewidth=2)
        
        if ddpg_rewards:
            episodes = range(len(ddpg_rewards))
            ax.plot(episodes, ddpg_rewards, label='DDPG', color=self.ddpg_color, alpha=0.7)
            
            # 스무딩된 곡선
            if len(ddpg_rewards) > 20:
                smoothed = smooth_data(ddpg_rewards, min(20, len(ddpg_rewards)//4))
                smooth_episodes = range(19, len(ddpg_rewards))
                ax.plot(smooth_episodes, smoothed, color=self.ddpg_color, linewidth=2)
        
        ax.legend()
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    def _plot_lengths_frame(self, ax, animation_data: Dict, current_episode: int):
        """에피소드 길이 프레임 플롯"""
        dqn_lengths = animation_data['dqn_lengths'][:current_episode]
        ddpg_lengths = animation_data['ddpg_lengths'][:current_episode]
        
        if dqn_lengths:
            ax.plot(dqn_lengths, label='DQN', color=self.dqn_color, alpha=0.7)
        
        if ddpg_lengths:
            ax.plot(ddpg_lengths, label='DDPG', color=self.ddpg_color, alpha=0.7)
        
        ax.legend()
        ax.set_title('Episode Length')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.grid(True, alpha=0.3)
    
    def _plot_losses_frame(self, ax, animation_data: Dict, current_episode: int):
        """학습 손실 프레임 플롯"""
        # 손실 데이터는 에피소드가 아닌 스텝 기준이므로 다르게 처리
        total_frames = 1000  # 임시로 1000 프레임 기준
        
        dqn_losses = animation_data['dqn_losses']
        ddpg_losses = animation_data['ddpg_losses']
        
        if dqn_losses:
            # 현재 에피소드 비율로 손실 데이터 크기 계산
            max_episodes = animation_data['max_episodes']
            loss_ratio = current_episode / max_episodes if max_episodes > 0 else 0
            current_loss_idx = int(len(dqn_losses) * loss_ratio)
            
            current_losses = dqn_losses[:current_loss_idx]
            if current_losses:
                ax.plot(current_losses, label='DQN', color=self.dqn_color, alpha=0.7)
        
        if ddpg_losses:
            max_episodes = animation_data['max_episodes']
            loss_ratio = current_episode / max_episodes if max_episodes > 0 else 0
            current_loss_idx = int(len(ddpg_losses) * loss_ratio)
            
            current_losses = ddpg_losses[:current_loss_idx]
            if current_losses:
                ax.plot(current_losses, label='DDPG', color=self.ddpg_color, alpha=0.7)
        
        ax.legend()
        ax.set_title('Training Loss')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_frame(self, ax, animation_data: Dict, current_episode: int):
        """누적 보상 프레임 플롯"""
        dqn_cumulative = animation_data['dqn_cumulative'][:current_episode]
        ddpg_cumulative = animation_data['ddpg_cumulative'][:current_episode]
        
        if dqn_cumulative:
            ax.plot(dqn_cumulative, label='DQN', color=self.dqn_color, linewidth=2)
        
        if ddpg_cumulative:
            ax.plot(ddpg_cumulative, label='DDPG', color=self.ddpg_color, linewidth=2)
        
        ax.legend()
        ax.set_title('Cumulative Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True, alpha=0.3)
    
    def _generate_sample_data(self) -> Tuple[Dict, Dict]:
        """샘플 학습 데이터 생성"""
        
        # DQN 샘플 데이터 (CartPole 스타일)
        episodes = 500
        dqn_rewards = []
        dqn_lengths = []
        
        # 초기에는 낮은 성능, 점진적 개선
        for i in range(episodes):
            progress = i / episodes
            base_reward = 20 + progress * 480  # 20에서 500까지 증가
            noise = np.random.normal(0, 50 - progress * 30)  # 노이즈 감소
            reward = max(10, min(500, base_reward + noise))
            
            dqn_rewards.append(reward)
            dqn_lengths.append(int(reward))  # 길이는 보상과 유사
        
        # DDPG 샘플 데이터 (Pendulum 스타일)
        episodes = 300
        ddpg_rewards = []
        ddpg_lengths = []
        
        for i in range(episodes):
            progress = i / episodes
            base_reward = -1000 + progress * 800  # -1000에서 -200까지 증가
            noise = np.random.normal(0, 200 - progress * 150)
            reward = max(-1500, min(-100, base_reward + noise))
            
            ddpg_rewards.append(reward)
            ddpg_lengths.append(200)  # 고정 길이
        
        # 훈련 손실 (감소 추세)
        dqn_losses = [10 * np.exp(-i/1000) + np.random.normal(0, 0.1) 
                     for i in range(len(dqn_rewards) * 10)]
        ddpg_losses = [5 * np.exp(-i/800) + np.random.normal(0, 0.05) 
                      for i in range(len(ddpg_rewards) * 8)]
        
        dqn_data = {
            'episode_rewards': dqn_rewards,
            'episode_lengths': dqn_lengths,
            'training_losses': dqn_losses
        }
        
        ddpg_data = {
            'episode_rewards': ddpg_rewards,
            'episode_lengths': ddpg_lengths,
            'training_losses': ddpg_losses
        }
        
        self.logger.info("샘플 데이터 생성 완료")
        
        return dqn_data, ddpg_data
    
    def create_comparison_video(self,
                              dqn_data: Dict[str, Any],
                              ddpg_data: Dict[str, Any],
                              save_filename: str = "algorithm_comparison.mp4",
                              duration_seconds: int = 30) -> str:
        """
        알고리즘 비교 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        # 간단한 정적 비교 차트를 시간에 따라 표시
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DQN vs DDPG Algorithm Comparison', fontsize=20)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution
            )
            
            total_frames = duration_seconds * self.fps
            
            for frame_idx in range(total_frames):
                # 각 프레임에서 다른 비교 차트 표시
                chart_type = (frame_idx // (total_frames // 4)) % 4
                
                self._render_comparison_frame(fig, axes, dqn_data, ddpg_data, chart_type)
                
                # Figure를 이미지로 변환하여 비디오에 추가
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, self.resolution)
                
                if self.video_writer:
                    self.video_writer.write(img)
            
        finally:
            if self.video_writer:
                self.video_writer.release()
            plt.close(fig)
            cv2.destroyAllWindows()
        
        return video_path
    
    def _render_comparison_frame(self, fig, axes, dqn_data: Dict, ddpg_data: Dict, chart_type: int):
        """비교 프레임 렌더링"""
        # 모든 축 클리어
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        if chart_type == 0:
            # 성능 요약
            self._plot_performance_summary(axes, dqn_data, ddpg_data)
        elif chart_type == 1:
            # 학습 곡선
            self._plot_learning_curves_static(axes, dqn_data, ddpg_data)
        elif chart_type == 2:
            # 분포 비교
            self._plot_distributions(axes, dqn_data, ddpg_data)
        else:
            # 최종 결과
            self._plot_final_results(axes, dqn_data, ddpg_data)
    
    def _plot_performance_summary(self, axes, dqn_data: Dict, ddpg_data: Dict):
        """성능 요약 플롯"""
        # 구현 간소화
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'Performance Summary\n(Chart 1/4)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
    
    def _plot_learning_curves_static(self, axes, dqn_data: Dict, ddpg_data: Dict):
        """정적 학습 곡선 플롯"""
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'Learning Curves\n(Chart 2/4)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
    
    def _plot_distributions(self, axes, dqn_data: Dict, ddpg_data: Dict):
        """분포 비교 플롯"""
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'Reward Distributions\n(Chart 3/4)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
    
    def _plot_final_results(self, axes, dqn_data: Dict, ddpg_data: Dict):
        """최종 결과 플롯"""
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'Final Results\n(Chart 4/4)', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)