"""
실시간 그래프를 표시하는 환경 래퍼
에피소드별 보상, Q-값, 손실 등의 지표를 실시간으로 그래프로 표시합니다.
"""

import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정
import matplotlib.pyplot as plt
from collections import deque
import io
from PIL import Image


class RealtimeGraphWrapper(gym.Wrapper):
    """실시간 그래프 오버레이 래퍼
    
    게임 화면 옆에 실시간으로 업데이트되는 그래프를 표시합니다.
    - 에피소드별 보상 추이
    - 최근 N개 에피소드의 이동평균
    - 현재 에피소드의 누적 보상
    """
    
    def __init__(self, env: gym.Env,
                 graph_width: int = 400,
                 graph_height: int = 300,
                 history_length: int = 100,
                 ma_window: int = 10,
                 show_current_episode: bool = True,
                 algorithm_name: str = "Algorithm"):
        """
        Args:
            env: 래핑할 환경
            graph_width: 그래프 영역 너비
            graph_height: 그래프 영역 높이
            history_length: 표시할 에피소드 히스토리 길이
            ma_window: 이동평균 윈도우 크기
            show_current_episode: 현재 에피소드 진행상황 표시 여부
            algorithm_name: 알고리즘 이름 (그래프 제목용)
        """
        super().__init__(env)
        
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.history_length = history_length
        self.ma_window = ma_window
        self.show_current_episode = show_current_episode
        self.algorithm_name = algorithm_name
        
        # 데이터 저장용
        self.episode_rewards = deque(maxlen=history_length)
        self.episode_lengths = deque(maxlen=history_length)
        self.current_episode_rewards = []
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0
        
        # 그래프 설정
        self._setup_graph()
    
    def _setup_graph(self):
        """matplotlib 그래프 초기 설정"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, 
                                                       figsize=(self.graph_width/100, 
                                                               self.graph_height/100))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # 상단 그래프: 에피소드별 총 보상
        self.ax1.set_title(f'{self.algorithm_name} - Episode Rewards', fontsize=10)
        self.ax1.set_xlabel('Episode', fontsize=8)
        self.ax1.set_ylabel('Total Reward', fontsize=8)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(labelsize=6)
        
        # 하단 그래프: 현재 에피소드 진행상황
        if self.show_current_episode:
            self.ax2.set_title('Current Episode Progress', fontsize=10)
            self.ax2.set_xlabel('Steps', fontsize=8)
            self.ax2.set_ylabel('Cumulative Reward', fontsize=8)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(labelsize=6)
        else:
            self.ax2.set_visible(False)
        
        plt.tight_layout()
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """환경 리셋 및 새 에피소드 시작"""
        # 이전 에피소드 데이터 저장
        if self.current_episode > 0 and self.total_reward != 0:
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.current_step)
        
        # 현재 에피소드 초기화
        self.current_episode += 1
        self.current_step = 0
        self.total_reward = 0
        self.current_episode_rewards = []
        
        return self.env.reset(**kwargs)
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행 및 데이터 업데이트"""
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # 데이터 업데이트
        self.current_step += 1
        self.total_reward += reward
        self.current_episode_rewards.append(self.total_reward)
        
        return state, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """게임 화면과 그래프를 결합하여 렌더링"""
        # 원본 게임 화면
        game_frame = self.env.render()
        if game_frame is None:
            return None
        
        # 그래프 렌더링
        graph_frame = self._render_graph()
        
        # 프레임 결합
        combined_frame = self._combine_frames(game_frame, graph_frame)
        
        return combined_frame
    
    def _render_graph(self) -> np.ndarray:
        """현재 데이터로 그래프를 렌더링"""
        # 그래프 초기화
        self.ax1.clear()
        if self.show_current_episode:
            self.ax2.clear()
        
        # 에피소드별 보상 그래프
        if len(self.episode_rewards) > 0:
            episodes = list(range(len(self.episode_rewards)))
            rewards = list(self.episode_rewards)
            
            # 실제 보상
            self.ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='Reward')
            
            # 이동평균
            if len(rewards) >= self.ma_window:
                ma = self._moving_average(rewards, self.ma_window)
                ma_episodes = episodes[self.ma_window-1:]
                self.ax1.plot(ma_episodes, ma, 'r-', linewidth=2, 
                             label=f'MA({self.ma_window})')
            
            self.ax1.legend(fontsize=8, loc='upper left')
            self.ax1.set_title(f'{self.algorithm_name} - Episode Rewards', fontsize=10)
            self.ax1.set_xlabel('Episode', fontsize=8)
            self.ax1.set_ylabel('Total Reward', fontsize=8)
            self.ax1.grid(True, alpha=0.3)
        
        # 현재 에피소드 진행상황
        if self.show_current_episode and len(self.current_episode_rewards) > 0:
            steps = list(range(len(self.current_episode_rewards)))
            self.ax2.plot(steps, self.current_episode_rewards, 'g-', linewidth=2)
            self.ax2.fill_between(steps, 0, self.current_episode_rewards, alpha=0.3)
            
            # 현재 상태 표시
            self.ax2.text(0.02, 0.98, f'Episode: {self.current_episode}', 
                         transform=self.ax2.transAxes, fontsize=8,
                         verticalalignment='top')
            self.ax2.text(0.02, 0.88, f'Current Reward: {self.total_reward:.1f}', 
                         transform=self.ax2.transAxes, fontsize=8,
                         verticalalignment='top')
            
            self.ax2.set_title('Current Episode Progress', fontsize=10)
            self.ax2.set_xlabel('Steps', fontsize=8)
            self.ax2.set_ylabel('Cumulative Reward', fontsize=8)
            self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # matplotlib figure를 numpy array로 변환
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # PIL Image로 읽어서 numpy array로 변환
        img = Image.open(buf)
        graph_array = np.array(img)
        buf.close()
        
        # RGBA를 RGB로 변환 (알파 채널 제거)
        if graph_array.shape[2] == 4:
            graph_array = graph_array[:, :, :3]
        
        return graph_array
    
    def _combine_frames(self, game_frame: np.ndarray, 
                       graph_frame: np.ndarray) -> np.ndarray:
        """게임 화면과 그래프를 나란히 결합"""
        # 높이를 맞추기 위해 리사이즈
        game_height = game_frame.shape[0]
        graph_height = graph_frame.shape[0]
        
        if game_height != graph_height:
            # 그래프를 게임 화면 높이에 맞춤
            scale = game_height / graph_height
            new_width = int(graph_frame.shape[1] * scale)
            graph_frame = cv2.resize(graph_frame, (new_width, game_height))
        
        # 프레임 결합
        combined = np.hstack([game_frame, graph_frame])
        
        return combined
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """이동평균 계산"""
        ma = []
        for i in range(window - 1, len(data)):
            ma.append(np.mean(data[i - window + 1:i + 1]))
        return ma
    
    def close(self):
        """리소스 정리"""
        plt.close(self.fig)
        super().close()


class AdvancedMetricsWrapper(RealtimeGraphWrapper):
    """고급 메트릭을 표시하는 확장 래퍼
    
    Q-값, 손실, 탐색률 등의 추가 지표를 실시간으로 표시합니다.
    """
    
    def __init__(self, env: gym.Env, 
                 show_q_values: bool = True,
                 show_loss: bool = True,
                 show_exploration: bool = True,
                 **kwargs):
        self.show_q_values = show_q_values
        self.show_loss = show_loss
        self.show_exploration = show_exploration
        
        super().__init__(env, **kwargs)
        
        # 추가 데이터 저장용
        self.q_values = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        self.exploration_rates = deque(maxlen=1000)
        
        # 외부에서 업데이트할 수 있는 메트릭
        self.latest_q_value = None
        self.latest_loss = None
        self.latest_exploration_rate = None
    
    def update_metrics(self, q_value: Optional[float] = None,
                      loss: Optional[float] = None,
                      exploration_rate: Optional[float] = None):
        """외부에서 메트릭 업데이트"""
        if q_value is not None:
            self.latest_q_value = q_value
            self.q_values.append(q_value)
        
        if loss is not None:
            self.latest_loss = loss
            self.losses.append(loss)
        
        if exploration_rate is not None:
            self.latest_exploration_rate = exploration_rate
            self.exploration_rates.append(exploration_rate)
    
    def _setup_graph(self):
        """확장된 그래프 설정"""
        num_plots = 2  # 기본 2개
        if self.show_q_values:
            num_plots += 1
        if self.show_loss:
            num_plots += 1
        
        self.fig, self.axes = plt.subplots(num_plots, 1,
                                          figsize=(self.graph_width/100,
                                                  self.graph_height/100))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        if num_plots == 1:
            self.axes = [self.axes]
        
        # 그래프 할당
        plot_idx = 0
        self.ax1 = self.axes[plot_idx]  # 에피소드 보상
        plot_idx += 1
        
        if self.show_current_episode:
            self.ax2 = self.axes[plot_idx]  # 현재 에피소드
            plot_idx += 1
        
        if self.show_q_values:
            self.ax_q = self.axes[plot_idx]
            plot_idx += 1
        
        if self.show_loss:
            self.ax_loss = self.axes[plot_idx]
            plot_idx += 1
        
        plt.tight_layout()
    
    def _render_graph(self) -> np.ndarray:
        """확장된 그래프 렌더링"""
        # 기본 그래프 렌더링
        super()._render_graph()
        
        # Q-값 그래프
        if self.show_q_values and len(self.q_values) > 0:
            self.ax_q.clear()
            self.ax_q.plot(list(self.q_values), 'purple', alpha=0.7)
            self.ax_q.set_title('Q-Values', fontsize=10)
            self.ax_q.set_ylabel('Q-Value', fontsize=8)
            self.ax_q.grid(True, alpha=0.3)
            self.ax_q.tick_params(labelsize=6)
            
            if self.latest_q_value is not None:
                self.ax_q.text(0.98, 0.02, f'Latest: {self.latest_q_value:.3f}',
                             transform=self.ax_q.transAxes, fontsize=8,
                             horizontalalignment='right')
        
        # 손실 그래프
        if self.show_loss and len(self.losses) > 0:
            self.ax_loss.clear()
            self.ax_loss.semilogy(list(self.losses), 'orange', alpha=0.7)
            self.ax_loss.set_title('Training Loss', fontsize=10)
            self.ax_loss.set_ylabel('Loss (log scale)', fontsize=8)
            self.ax_loss.grid(True, alpha=0.3)
            self.ax_loss.tick_params(labelsize=6)
            
            if self.latest_loss is not None:
                self.ax_loss.text(0.98, 0.02, f'Latest: {self.latest_loss:.3e}',
                                transform=self.ax_loss.transAxes, fontsize=8,
                                horizontalalignment='right')
        
        plt.tight_layout()
        
        # matplotlib figure를 numpy array로 변환
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        graph_array = np.array(img)
        buf.close()
        
        if graph_array.shape[2] == 4:
            graph_array = graph_array[:, :, :3]
        
        return graph_array