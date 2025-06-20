"""
학습 곡선 시각화 모듈

DQN과 DDPG의 학습 진행 과정을 다양한 관점에서 시각화합니다.
에피소드 보상, 길이, 손실, Q-값 변화 등을 포함합니다.

새로운 출력 구조:
- 확장자별 디렉토리 자동 분류 (PNG -> output/png/charts/, SVG -> output/svg/charts/ 등)
- 구조화된 파일명 생성 (learning_curves_chart_algorithm_environment_timestamp.png)
- BaseVisualizer.save_figure() 메서드가 자동으로 적절한 경로에 저장
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
import os

from ..core.base import MultiPlotVisualizer
from ..core.utils import (
    smooth_data, validate_experiment_data, calculate_statistics,
    get_output_path_by_extension, create_structured_filename
)


class LearningCurveVisualizer(MultiPlotVisualizer):
    """
    학습 곡선 시각화 클래스
    
    DQN과 DDPG의 학습 과정을 비교하여 시각화합니다.
    다양한 메트릭에 대한 학습 곡선을 생성할 수 있습니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 기본 색상 설정
        self.dqn_color = self.config.chart.dqn_color
        self.ddpg_color = self.config.chart.ddpg_color
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        학습 곡선 시각화 생성
        
        Args:
            data: 시각화할 데이터 {'dqn': dqn_metrics, 'ddpg': ddpg_metrics}
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 파일의 경로
        """
        if not self.validate_data(data):
            raise ValueError("유효하지 않은 데이터입니다")
        
        dqn_metrics = data.get('dqn', {})
        ddpg_metrics = data.get('ddpg', {})
        
        # 기본 학습 곡선 생성
        return self.plot_comprehensive_learning_curves(
            dqn_metrics, ddpg_metrics, **kwargs
        )
    
    def plot_comprehensive_learning_curves(self, 
                                         dqn_metrics: Dict[str, Any], 
                                         ddpg_metrics: Dict[str, Any],
                                         save_filename: str = "learning_curves_comparison.png",
                                         show_smoothed: bool = True,
                                         window_size: int = 50) -> str:
        """
        종합적인 학습 곡선 플롯 생성
        
        Args:
            dqn_metrics: DQN 실험 결과
            ddpg_metrics: DDPG 실험 결과
            save_filename: 저장할 파일명
            show_smoothed: 스무딩된 곡선 표시 여부
            window_size: 스무딩 윈도우 크기
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 2, 
            figsize=(15, 10),
            title='DQN vs DDPG Learning Performance Comparison'
        )
        
        # 1. 에피소드 보상
        self._plot_episode_rewards(axes[0], dqn_metrics, ddpg_metrics, 
                                  show_smoothed, window_size)
        
        # 2. 에피소드 길이
        self._plot_episode_lengths(axes[1], dqn_metrics, ddpg_metrics,
                                  show_smoothed, window_size)
        
        # 3. 학습 손실
        self._plot_training_losses(axes[2], dqn_metrics, ddpg_metrics)
        
        # 4. Q-값 변화
        self._plot_q_values(axes[3], dqn_metrics, ddpg_metrics)
        
        plt.tight_layout()
        
        # 확장자 기반 출력 구조 적용
        # save_figure 메서드가 자동으로 적절한 확장자별 디렉토리에 저장
        file_path = self.save_figure(fig, save_filename)
        
        self.log_visualization_info(
            "Comprehensive Learning Curves",
            {
                "dqn_episodes": len(dqn_metrics.get('episode_rewards', [])),
                "ddpg_episodes": len(ddpg_metrics.get('episode_rewards', [])),
                "smoothing_window": window_size,
                "show_smoothed": show_smoothed
            }
        )
        
        return file_path
    
    def _plot_episode_rewards(self, ax: plt.Axes, 
                            dqn_metrics: Dict, ddpg_metrics: Dict,
                            show_smoothed: bool = True,
                            window_size: int = 50):
        """에피소드 보상 플롯"""
        self.setup_subplot(ax, "Episode Rewards", "Episode", "Reward")
        
        # DQN 데이터 플롯
        if 'episode_rewards' in dqn_metrics:
            dqn_rewards = dqn_metrics['episode_rewards']
            episodes_dqn = range(len(dqn_rewards))
            
            # 원본 데이터 (투명하게)
            ax.plot(episodes_dqn, dqn_rewards, 
                   label='DQN (raw)', alpha=0.3, color=self.dqn_color, linewidth=1)
            
            # 스무딩된 데이터
            if show_smoothed and len(dqn_rewards) > window_size:
                window = min(window_size, len(dqn_rewards)//10)
                if window > 1:
                    smoothed = smooth_data(dqn_rewards, window)
                    ax.plot(range(window-1, len(dqn_rewards)), smoothed,
                           label='DQN (smoothed)', color=self.dqn_color, linewidth=2)
        
        # DDPG 데이터 플롯
        if 'episode_rewards' in ddpg_metrics:
            ddpg_rewards = ddpg_metrics['episode_rewards']
            episodes_ddpg = range(len(ddpg_rewards))
            
            # 원본 데이터 (투명하게)
            ax.plot(episodes_ddpg, ddpg_rewards,
                   label='DDPG (raw)', alpha=0.3, color=self.ddpg_color, linewidth=1)
            
            # 스무딩된 데이터
            if show_smoothed and len(ddpg_rewards) > window_size:
                window = min(window_size, len(ddpg_rewards)//10)
                if window > 1:
                    smoothed = smooth_data(ddpg_rewards, window)
                    ax.plot(range(window-1, len(ddpg_rewards)), smoothed,
                           label='DDPG (smoothed)', color=self.ddpg_color, linewidth=2)
        
        ax.legend()
    
    def _plot_episode_lengths(self, ax: plt.Axes,
                            dqn_metrics: Dict, ddpg_metrics: Dict,
                            show_smoothed: bool = True,
                            window_size: int = 50):
        """에피소드 길이 플롯"""
        self.setup_subplot(ax, "Episode Length", "Episode", "Length")
        
        if 'episode_lengths' in dqn_metrics:
            dqn_lengths = dqn_metrics['episode_lengths']
            ax.plot(dqn_lengths, label='DQN', alpha=0.7, color=self.dqn_color)
            
            if show_smoothed and len(dqn_lengths) > window_size:
                smoothed = smooth_data(dqn_lengths, window_size)
                ax.plot(range(window_size-1, len(dqn_lengths)), smoothed,
                       color=self.dqn_color, linewidth=2, alpha=0.9)
        
        if 'episode_lengths' in ddpg_metrics:
            ddpg_lengths = ddpg_metrics['episode_lengths']
            ax.plot(ddpg_lengths, label='DDPG', alpha=0.7, color=self.ddpg_color)
            
            if show_smoothed and len(ddpg_lengths) > window_size:
                smoothed = smooth_data(ddpg_lengths, window_size)
                ax.plot(range(window_size-1, len(ddpg_lengths)), smoothed,
                       color=self.ddpg_color, linewidth=2, alpha=0.9)
        
        ax.legend()
    
    def _plot_training_losses(self, ax: plt.Axes,
                            dqn_metrics: Dict, ddpg_metrics: Dict):
        """학습 손실 플롯"""
        self.setup_subplot(ax, "Training Loss", "Training Step", "Loss")
        
        has_data = False
        
        if 'training_losses' in dqn_metrics and dqn_metrics['training_losses']:
            ax.plot(dqn_metrics['training_losses'], 
                   label='DQN', alpha=0.7, color=self.dqn_color)
            has_data = True
        
        if 'training_losses' in ddpg_metrics and ddpg_metrics['training_losses']:
            ax.plot(ddpg_metrics['training_losses'], 
                   label='DDPG (Critic)', alpha=0.7, color=self.ddpg_color)
            has_data = True
        
        if has_data:
            ax.set_yscale('log')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No loss data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_q_values(self, ax: plt.Axes,
                      dqn_metrics: Dict, ddpg_metrics: Dict):
        """Q-값 변화 플롯"""
        self.setup_subplot(ax, "Q-Value Changes", "Training Step", "Average Q-value")
        
        has_data = False
        
        if 'q_values' in dqn_metrics and dqn_metrics['q_values']:
            ax.plot(dqn_metrics['q_values'], 
                   label='DQN Q-values', alpha=0.7, color=self.dqn_color)
            has_data = True
        
        if 'q_values' in ddpg_metrics and ddpg_metrics['q_values']:
            ax.plot(ddpg_metrics['q_values'], 
                   label='DDPG Q-values', alpha=0.7, color=self.ddpg_color)
            has_data = True
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Q-value data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def plot_reward_progression(self, 
                              dqn_metrics: Dict[str, Any],
                              ddpg_metrics: Dict[str, Any],
                              save_filename: str = "reward_progression.png",
                              show_confidence_bands: bool = True) -> str:
        """
        보상 진행 과정 상세 플롯
        
        Args:
            dqn_metrics: DQN 실험 결과
            ddpg_metrics: DDPG 실험 결과
            save_filename: 저장할 파일명
            show_confidence_bands: 신뢰구간 표시 여부
            
        Returns:
            저장된 파일 경로
        """
        fig, ax = self.create_figure(figsize=(12, 6), title="Reward Progression Analysis")
        
        # DQN 보상 진행
        if 'episode_rewards' in dqn_metrics:
            dqn_rewards = np.array(dqn_metrics['episode_rewards'])
            episodes = np.arange(len(dqn_rewards))
            
            # 이동 평균과 표준편차
            window = 50
            if len(dqn_rewards) > window:
                smoothed = smooth_data(dqn_rewards, window)
                episodes_smooth = episodes[window-1:]
                
                ax.plot(episodes_smooth, smoothed, 
                       label='DQN', color=self.dqn_color, linewidth=2)
                
                # 신뢰구간
                if show_confidence_bands:
                    rolling_std = np.array([np.std(dqn_rewards[max(0, i-window):i+1]) 
                                          for i in range(window-1, len(dqn_rewards))])
                    ax.fill_between(episodes_smooth, 
                                  smoothed - rolling_std, 
                                  smoothed + rolling_std,
                                  alpha=0.2, color=self.dqn_color)
        
        # DDPG 보상 진행
        if 'episode_rewards' in ddpg_metrics:
            ddpg_rewards = np.array(ddpg_metrics['episode_rewards'])
            episodes = np.arange(len(ddpg_rewards))
            
            # 이동 평균과 표준편차
            window = 50
            if len(ddpg_rewards) > window:
                smoothed = smooth_data(ddpg_rewards, window)
                episodes_smooth = episodes[window-1:]
                
                ax.plot(episodes_smooth, smoothed,
                       label='DDPG', color=self.ddpg_color, linewidth=2)
                
                # 신뢰구간
                if show_confidence_bands:
                    rolling_std = np.array([np.std(ddpg_rewards[max(0, i-window):i+1])
                                          for i in range(window-1, len(ddpg_rewards))])
                    ax.fill_between(episodes_smooth,
                                  smoothed - rolling_std,
                                  smoothed + rolling_std,
                                  alpha=0.2, color=self.ddpg_color)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self.save_figure(fig, save_filename)
    
    def plot_learning_efficiency(self,
                                dqn_metrics: Dict[str, Any],
                                ddpg_metrics: Dict[str, Any],
                                save_filename: str = "learning_efficiency.png") -> str:
        """
        학습 효율성 분석 플롯
        
        Args:
            dqn_metrics: DQN 실험 결과
            ddpg_metrics: DDPG 실험 결과  
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            1, 2,
            figsize=(15, 6),
            title="Learning Efficiency Analysis"
        )
        
        # 1. 누적 보상 비교
        ax1 = axes[0]
        self.setup_subplot(ax1, "Cumulative Reward", "Episode", "Cumulative Reward")
        
        if 'episode_rewards' in dqn_metrics:
            dqn_cumulative = np.cumsum(dqn_metrics['episode_rewards'])
            ax1.plot(dqn_cumulative, label='DQN', color=self.dqn_color, linewidth=2)
        
        if 'episode_rewards' in ddpg_metrics:
            ddpg_cumulative = np.cumsum(ddpg_metrics['episode_rewards'])
            ax1.plot(ddpg_cumulative, label='DDPG', color=self.ddpg_color, linewidth=2)
        
        ax1.legend()
        
        # 2. 학습률 (보상 개선 속도)
        ax2 = axes[1]
        self.setup_subplot(ax2, "Learning Rate", "Episode", "Reward Improvement Rate")
        
        window = 100
        
        if 'episode_rewards' in dqn_metrics and len(dqn_metrics['episode_rewards']) > window:
            dqn_rewards = np.array(dqn_metrics['episode_rewards'])
            learning_rate = []
            for i in range(window, len(dqn_rewards)):
                recent_avg = np.mean(dqn_rewards[i-window:i])
                prev_avg = np.mean(dqn_rewards[i-2*window:i-window]) if i >= 2*window else np.mean(dqn_rewards[:window])
                rate = (recent_avg - prev_avg) / window
                learning_rate.append(rate)
            
            ax2.plot(range(window, len(dqn_rewards)), learning_rate,
                    label='DQN', color=self.dqn_color, linewidth=2)
        
        if 'episode_rewards' in ddpg_metrics and len(ddpg_metrics['episode_rewards']) > window:
            ddpg_rewards = np.array(ddpg_metrics['episode_rewards'])
            learning_rate = []
            for i in range(window, len(ddpg_rewards)):
                recent_avg = np.mean(ddpg_rewards[i-window:i])
                prev_avg = np.mean(ddpg_rewards[i-2*window:i-window]) if i >= 2*window else np.mean(ddpg_rewards[:window])
                rate = (recent_avg - prev_avg) / window
                learning_rate.append(rate)
            
            ax2.plot(range(window, len(ddpg_rewards)), learning_rate,
                    label='DDPG', color=self.ddpg_color, linewidth=2)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        return self.save_figure(fig, save_filename)