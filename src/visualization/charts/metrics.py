"""
성능 지표 시각화 모듈

DQN과 DDPG의 다양한 성능 지표를 시각화합니다.
손실 함수, Q-값, 탐험률, 학습률 등의 메트릭을 포함합니다.

새로운 출력 구조:
- 확장자별 디렉토리 자동 분류 (PNG -> output/png/metrics/, PDF -> output/pdf/metrics/ 등)
- 구조화된 파일명 생성 기능으로 일관된 파일 관리
- get_output_path_by_extension() 함수를 통한 자동 경로 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
import os

from ..core.base import MultiPlotVisualizer
from ..core.utils import (
    smooth_data, validate_experiment_data, calculate_statistics,
    get_output_path_by_extension, create_structured_filename
)


class MetricsVisualizer(MultiPlotVisualizer):
    """
    성능 지표 시각화 클래스
    
    다양한 훈련 메트릭과 성능 지표를 시각화합니다.
    손실, Q-값, 탐험, 학습률 등의 변화를 추적합니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 메트릭별 색상 설정
        self.dqn_color = self.config.chart.dqn_color
        self.ddpg_color = self.config.chart.ddpg_color
        self.loss_color = '#d62728'
        self.q_value_color = '#2ca02c'
        self.exploration_color = '#9467bd'
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        메트릭 시각화 생성
        
        Args:
            data: 시각화할 데이터 {'dqn': dqn_metrics, 'ddpg': ddpg_metrics}
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 파일의 경로
        """
        if not self.validate_data(data):
            raise ValueError("유효하지 않은 데이터입니다")
        
        dqn_data = data.get('dqn', {})
        ddpg_data = data.get('ddpg', {})
        
        # 기본 훈련 메트릭 시각화 생성
        return self.plot_training_metrics(
            dqn_data, ddpg_data, **kwargs
        )
    
    def plot_training_metrics(self, 
                            dqn_data: Dict[str, Any], 
                            ddpg_data: Dict[str, Any],
                            save_filename: str = "training_metrics.png") -> str:
        """
        훈련 메트릭 종합 시각화
        
        Args:
            dqn_data: DQN 훈련 데이터
            ddpg_data: DDPG 훈련 데이터
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 3, 
            figsize=(18, 12),
            title='Training Metrics Dashboard'
        )
        
        # 1. 손실 함수 변화
        self._plot_loss_curves(axes[0], dqn_data, ddpg_data)
        
        # 2. Q-값 변화
        self._plot_q_value_trends(axes[1], dqn_data, ddpg_data)
        
        # 3. 탐험률 변화 (DQN) / 노이즈 레벨 (DDPG)
        self._plot_exploration_metrics(axes[2], dqn_data, ddpg_data)
        
        # 4. 학습률 변화
        self._plot_learning_rate_changes(axes[3], dqn_data, ddpg_data)
        
        # 5. 네트워크 업데이트 빈도
        self._plot_update_frequency(axes[4], dqn_data, ddpg_data)
        
        # 6. 메모리 사용량 (버퍼 크기 등)
        self._plot_memory_usage(axes[5], dqn_data, ddpg_data)
        
        plt.tight_layout()
        
        file_path = self.save_figure(fig, save_filename)
        
        self.log_visualization_info(
            "Training Metrics Dashboard",
            {
                "has_dqn_loss": 'training_losses' in dqn_data,
                "has_ddpg_loss": 'training_losses' in ddpg_data,
                "has_q_values": 'q_values' in dqn_data or 'q_values' in ddpg_data
            }
        )
        
        return file_path
    
    def _plot_loss_curves(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """손실 함수 변화 플롯"""
        self.setup_subplot(ax, "Training Loss", "Training Step", "Loss")
        
        has_data = False
        
        if 'training_losses' in dqn_data and dqn_data['training_losses']:
            losses = dqn_data['training_losses']
            ax.plot(losses, label='DQN Loss', color=self.dqn_color, alpha=0.7)
            
            # 스무딩된 손실
            if len(losses) > 100:
                smoothed = smooth_data(losses, 100)
                ax.plot(range(99, len(losses)), smoothed, 
                       color=self.dqn_color, linewidth=2)
            has_data = True
        
        if 'training_losses' in ddpg_data and ddpg_data['training_losses']:
            # DDPG는 actor와 critic 손실이 따로 있을 수 있음
            losses = ddpg_data['training_losses']
            if isinstance(losses, dict):
                if 'critic' in losses:
                    ax.plot(losses['critic'], label='DDPG Critic Loss', 
                           color=self.ddpg_color, alpha=0.7)
                if 'actor' in losses:
                    ax.plot(losses['actor'], label='DDPG Actor Loss', 
                           color=self.ddpg_color, alpha=0.7, linestyle='--')
            else:
                ax.plot(losses, label='DDPG Loss', color=self.ddpg_color, alpha=0.7)
            has_data = True
        
        if has_data:
            ax.set_yscale('log')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No loss data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_q_value_trends(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """Q-값 변화 추세 플롯"""
        self.setup_subplot(ax, "Q-Value Trends", "Training Step", "Average Q-Value")
        
        has_data = False
        
        if 'q_values' in dqn_data and dqn_data['q_values']:
            q_values = dqn_data['q_values']
            ax.plot(q_values, label='DQN Q-values', 
                   color=self.q_value_color, alpha=0.7)
            
            # 트렌드 라인
            if len(q_values) > 50:
                smoothed = smooth_data(q_values, 50)
                ax.plot(range(49, len(q_values)), smoothed,
                       color=self.q_value_color, linewidth=2)
            has_data = True
        
        if 'q_values' in ddpg_data and ddpg_data['q_values']:
            q_values = ddpg_data['q_values']
            ax.plot(q_values, label='DDPG Q-values', 
                   color=self.ddpg_color, alpha=0.7)
            
            # 트렌드 라인
            if len(q_values) > 50:
                smoothed = smooth_data(q_values, 50)
                ax.plot(range(49, len(q_values)), smoothed,
                       color=self.ddpg_color, linewidth=2)
            has_data = True
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Q-value data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_exploration_metrics(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """탐험 메트릭 플롯"""
        self.setup_subplot(ax, "Exploration Metrics", "Episode/Step", "Exploration Level")
        
        has_data = False
        
        # DQN 엡실론 변화
        if 'epsilon_values' in dqn_data:
            epsilon_values = dqn_data['epsilon_values']
            ax.plot(epsilon_values, label='DQN ε-greedy', 
                   color=self.exploration_color, linewidth=2)
            has_data = True
        
        # DDPG 노이즈 레벨
        if 'noise_levels' in ddpg_data:
            noise_levels = ddpg_data['noise_levels']
            ax.plot(noise_levels, label='DDPG Noise Level', 
                   color=self.ddpg_color, linewidth=2)
            has_data = True
        
        if has_data:
            ax.legend()
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, 'No exploration data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_learning_rate_changes(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """학습률 변화 플롯"""
        self.setup_subplot(ax, "Learning Rate", "Training Step", "Learning Rate")
        
        has_data = False
        
        if 'learning_rates' in dqn_data:
            lr_values = dqn_data['learning_rates']
            ax.plot(lr_values, label='DQN LR', color=self.dqn_color, linewidth=2)
            has_data = True
        
        if 'learning_rates' in ddpg_data:
            lr_data = ddpg_data['learning_rates']
            if isinstance(lr_data, dict):
                if 'actor' in lr_data:
                    ax.plot(lr_data['actor'], label='DDPG Actor LR', 
                           color=self.ddpg_color, linewidth=2)
                if 'critic' in lr_data:
                    ax.plot(lr_data['critic'], label='DDPG Critic LR', 
                           color=self.ddpg_color, linewidth=2, linestyle='--')
            else:
                ax.plot(lr_data, label='DDPG LR', color=self.ddpg_color, linewidth=2)
            has_data = True
        
        if has_data:
            ax.set_yscale('log')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No learning rate data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_update_frequency(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """네트워크 업데이트 빈도 플롯"""
        self.setup_subplot(ax, "Network Updates", "Episode", "Updates per Episode")
        
        has_data = False
        
        if 'update_counts' in dqn_data:
            updates = dqn_data['update_counts']
            ax.plot(updates, label='DQN Updates', color=self.dqn_color, alpha=0.7)
            has_data = True
        
        if 'update_counts' in ddpg_data:
            updates = ddpg_data['update_counts']
            ax.plot(updates, label='DDPG Updates', color=self.ddpg_color, alpha=0.7)
            has_data = True
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No update frequency data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_memory_usage(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """메모리 사용량 플롯"""
        self.setup_subplot(ax, "Replay Buffer Usage", "Episode", "Buffer Fill %")
        
        has_data = False
        
        if 'buffer_sizes' in dqn_data:
            buffer_usage = dqn_data['buffer_sizes']
            if 'max_buffer_size' in dqn_data:
                max_size = dqn_data['max_buffer_size']
                buffer_percentage = [size/max_size*100 for size in buffer_usage]
                ax.plot(buffer_percentage, label='DQN Buffer', 
                       color=self.dqn_color, linewidth=2)
                has_data = True
        
        if 'buffer_sizes' in ddpg_data:
            buffer_usage = ddpg_data['buffer_sizes']
            if 'max_buffer_size' in ddpg_data:
                max_size = ddpg_data['max_buffer_size']
                buffer_percentage = [size/max_size*100 for size in buffer_usage]
                ax.plot(buffer_percentage, label='DDPG Buffer', 
                       color=self.ddpg_color, linewidth=2)
                has_data = True
        
        if has_data:
            ax.set_ylim([0, 100])
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No buffer usage data available', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def plot_loss_analysis(self,
                          dqn_data: Dict[str, Any],
                          ddpg_data: Dict[str, Any],
                          save_filename: str = "loss_analysis.png") -> str:
        """
        손실 함수 상세 분석
        
        Args:
            dqn_data: DQN 훈련 데이터
            ddpg_data: DDPG 훈련 데이터
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 2,
            figsize=(15, 10),
            title="Loss Function Analysis"
        )
        
        # 1. 원본 손실 곡선
        self._plot_raw_losses(axes[0], dqn_data, ddpg_data)
        
        # 2. 스무딩된 손실 곡선
        self._plot_smoothed_losses(axes[1], dqn_data, ddpg_data)
        
        # 3. 손실 분포
        self._plot_loss_distribution(axes[2], dqn_data, ddpg_data)
        
        # 4. 손실 변화율
        self._plot_loss_gradients(axes[3], dqn_data, ddpg_data)
        
        plt.tight_layout()
        
        return self.save_figure(fig, save_filename)
    
    def _plot_raw_losses(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """원본 손실 곡선 플롯"""
        self.setup_subplot(ax, "Raw Loss Curves", "Training Step", "Loss")
        
        if 'training_losses' in dqn_data and dqn_data['training_losses']:
            ax.plot(dqn_data['training_losses'], label='DQN', 
                   color=self.dqn_color, alpha=0.6)
        
        if 'training_losses' in ddpg_data and ddpg_data['training_losses']:
            ax.plot(ddpg_data['training_losses'], label='DDPG', 
                   color=self.ddpg_color, alpha=0.6)
        
        ax.set_yscale('log')
        ax.legend()
    
    def _plot_smoothed_losses(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """스무딩된 손실 곡선 플롯"""
        self.setup_subplot(ax, "Smoothed Loss Curves", "Training Step", "Smoothed Loss")
        
        window = 100
        
        if 'training_losses' in dqn_data and len(dqn_data['training_losses']) > window:
            losses = dqn_data['training_losses']
            smoothed = smooth_data(losses, window)
            ax.plot(range(window-1, len(losses)), smoothed, 
                   label='DQN', color=self.dqn_color, linewidth=2)
        
        if 'training_losses' in ddpg_data and len(ddpg_data['training_losses']) > window:
            losses = ddpg_data['training_losses']
            smoothed = smooth_data(losses, window)
            ax.plot(range(window-1, len(losses)), smoothed, 
                   label='DDPG', color=self.ddpg_color, linewidth=2)
        
        ax.set_yscale('log')
        ax.legend()
    
    def _plot_loss_distribution(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """손실 분포 플롯"""
        self.setup_subplot(ax, "Loss Distribution", "Loss Value", "Density")
        
        if 'training_losses' in dqn_data and dqn_data['training_losses']:
            losses = np.array(dqn_data['training_losses'])
            # 극값 제거 (상위 5% 제거)
            losses = losses[losses <= np.percentile(losses, 95)]
            ax.hist(losses, bins=50, alpha=0.7, label='DQN', 
                   color=self.dqn_color, density=True)
        
        if 'training_losses' in ddpg_data and ddpg_data['training_losses']:
            losses = np.array(ddpg_data['training_losses'])
            losses = losses[losses <= np.percentile(losses, 95)]
            ax.hist(losses, bins=50, alpha=0.7, label='DDPG', 
                   color=self.ddpg_color, density=True)
        
        ax.set_xscale('log')
        ax.legend()
    
    def _plot_loss_gradients(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """손실 변화율 플롯"""
        self.setup_subplot(ax, "Loss Gradients", "Training Step", "Loss Change Rate")
        
        if 'training_losses' in dqn_data and len(dqn_data['training_losses']) > 1:
            losses = np.array(dqn_data['training_losses'])
            gradients = np.diff(losses)
            # 스무딩
            if len(gradients) > 50:
                gradients = smooth_data(gradients, 50)
            ax.plot(gradients, label='DQN', color=self.dqn_color, alpha=0.7)
        
        if 'training_losses' in ddpg_data and len(ddpg_data['training_losses']) > 1:
            losses = np.array(ddpg_data['training_losses'])
            gradients = np.diff(losses)
            if len(gradients) > 50:
                gradients = smooth_data(gradients, 50)
            ax.plot(gradients, label='DDPG', color=self.ddpg_color, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
    
    def plot_q_value_analysis(self,
                            dqn_data: Dict[str, Any],
                            ddpg_data: Dict[str, Any],
                            save_filename: str = "q_value_analysis.png") -> str:
        """
        Q-값 상세 분석
        
        Args:
            dqn_data: DQN 훈련 데이터
            ddpg_data: DDPG 훈련 데이터
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 2,
            figsize=(15, 10),
            title="Q-Value Analysis"
        )
        
        # 1. Q-값 진화
        self._plot_q_value_evolution(axes[0], dqn_data, ddpg_data)
        
        # 2. Q-값 분포
        self._plot_q_value_distribution(axes[1], dqn_data, ddpg_data)
        
        # 3. Q-값 안정성
        self._plot_q_value_stability(axes[2], dqn_data, ddpg_data)
        
        # 4. Q-값 수렴
        self._plot_q_value_convergence(axes[3], dqn_data, ddpg_data)
        
        plt.tight_layout()
        
        return self.save_figure(fig, save_filename)
    
    def _plot_q_value_evolution(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """Q-값 진화 플롯"""
        self.setup_subplot(ax, "Q-Value Evolution", "Training Step", "Average Q-Value")
        
        if 'q_values' in dqn_data and dqn_data['q_values']:
            ax.plot(dqn_data['q_values'], label='DQN', 
                   color=self.dqn_color, alpha=0.7)
        
        if 'q_values' in ddpg_data and ddpg_data['q_values']:
            ax.plot(ddpg_data['q_values'], label='DDPG', 
                   color=self.ddpg_color, alpha=0.7)
        
        ax.legend()
    
    def _plot_q_value_distribution(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """Q-값 분포 플롯"""
        self.setup_subplot(ax, "Q-Value Distribution", "Q-Value", "Density")
        
        if 'q_values' in dqn_data and dqn_data['q_values']:
            ax.hist(dqn_data['q_values'], bins=50, alpha=0.7, 
                   label='DQN', color=self.dqn_color, density=True)
        
        if 'q_values' in ddpg_data and ddpg_data['q_values']:
            ax.hist(ddpg_data['q_values'], bins=50, alpha=0.7, 
                   label='DDPG', color=self.ddpg_color, density=True)
        
        ax.legend()
    
    def _plot_q_value_stability(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """Q-값 안정성 플롯"""
        self.setup_subplot(ax, "Q-Value Stability", "Training Step", "Q-Value Variance")
        
        window = 100
        
        if 'q_values' in dqn_data and len(dqn_data['q_values']) > window:
            q_values = np.array(dqn_data['q_values'])
            rolling_var = []
            for i in range(window, len(q_values)):
                variance = np.var(q_values[i-window:i])
                rolling_var.append(variance)
            ax.plot(range(window, len(q_values)), rolling_var, 
                   label='DQN', color=self.dqn_color)
        
        if 'q_values' in ddpg_data and len(ddpg_data['q_values']) > window:
            q_values = np.array(ddpg_data['q_values'])
            rolling_var = []
            for i in range(window, len(q_values)):
                variance = np.var(q_values[i-window:i])
                rolling_var.append(variance)
            ax.plot(range(window, len(q_values)), rolling_var, 
                   label='DDPG', color=self.ddpg_color)
        
        ax.legend()
    
    def _plot_q_value_convergence(self, ax: plt.Axes, dqn_data: Dict, ddpg_data: Dict):
        """Q-값 수렴 플롯"""
        self.setup_subplot(ax, "Q-Value Convergence", "Training Step", "Change Rate")
        
        if 'q_values' in dqn_data and len(dqn_data['q_values']) > 1:
            q_values = np.array(dqn_data['q_values'])
            changes = np.abs(np.diff(q_values))
            if len(changes) > 50:
                changes = smooth_data(changes, 50)
            ax.plot(changes, label='DQN', color=self.dqn_color)
        
        if 'q_values' in ddpg_data and len(ddpg_data['q_values']) > 1:
            q_values = np.array(ddpg_data['q_values'])
            changes = np.abs(np.diff(q_values))
            if len(changes) > 50:
                changes = smooth_data(changes, 50)
            ax.plot(changes, label='DDPG', color=self.ddpg_color)
        
        ax.set_yscale('log')
        ax.legend()