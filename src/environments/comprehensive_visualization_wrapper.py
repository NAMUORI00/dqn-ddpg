"""
종합적인 시각화 래퍼 - 게임플레이, 실시간 그래프, 비교분석을 모두 포함
DQN과 DDPG의 차이점을 한 화면에서 완전하게 시각화합니다.
"""

import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import io
from PIL import Image
import time


class ComprehensiveVisualizationWrapper(gym.Wrapper):
    """종합 시각화 래퍼
    
    한 화면에서 다음을 모두 표시:
    1. 게임플레이 영상 (상단)
    2. 실시간 성능 그래프 (중단)
    3. 비교 분석 지표 (하단)
    4. 통계 정보 오버레이
    """
    
    def __init__(self, env: gym.Env,
                 algorithm_name: str = "Algorithm",
                 partner_wrapper: Optional['ComprehensiveVisualizationWrapper'] = None,
                 frame_width: int = 600,
                 frame_height: int = 400,
                 graph_height: int = 300,
                 stats_height: int = 150):
        """
        Args:
            env: 래핑할 환경
            algorithm_name: 알고리즘 이름 (DQN, DDPG)
            partner_wrapper: 비교할 상대방 래퍼 (DQN vs DDPG 비교용)
            frame_width: 게임 프레임 너비
            frame_height: 게임 프레임 높이
            graph_height: 그래프 영역 높이
            stats_height: 통계 영역 높이
        """
        super().__init__(env)
        
        self.algorithm_name = algorithm_name
        self.partner_wrapper = partner_wrapper
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.graph_height = graph_height
        self.stats_height = stats_height
        
        # 전체 프레임 크기
        self.total_height = frame_height + graph_height + stats_height
        
        # 데이터 저장용
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.q_values = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        self.exploration_rates = deque(maxlen=1000)
        
        # 현재 에피소드 정보
        self.current_episode = 0
        self.current_step = 0
        self.total_reward = 0
        self.current_episode_rewards = []
        self.start_time = time.time()
        
        # 외부에서 업데이트할 메트릭
        self.latest_q_value = None
        self.latest_loss = None
        self.latest_exploration_rate = None
        
        # 성능 통계
        self.best_reward = float('-inf')
        self.worst_reward = float('inf')
        self.total_episodes = 0
        self.total_steps = 0
        
        # matplotlib 설정
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """matplotlib 그래프 설정"""
        # 그래프 크기 계산 (인치 단위)
        fig_width = self.frame_width / 100
        fig_height = self.graph_height / 100
        
        self.fig, ((self.ax_reward, self.ax_q), (self.ax_loss, self.ax_current)) = plt.subplots(
            2, 2, figsize=(fig_width, fig_height), facecolor='white'
        )
        
        # 각 그래프 기본 설정
        self._setup_subplot(self.ax_reward, "Episode Rewards", "Episode", "Reward")
        self._setup_subplot(self.ax_q, "Q-Values", "Step", "Q-Value")
        self._setup_subplot(self.ax_loss, "Training Loss", "Step", "Loss (log)")
        self._setup_subplot(self.ax_current, "Current Episode", "Step", "Cumulative Reward")
        
        plt.tight_layout(pad=1.0)
    
    def _setup_subplot(self, ax, title, xlabel, ylabel):
        """개별 서브플롯 설정"""
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    
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
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """환경 리셋 및 새 에피소드 시작"""
        # 이전 에피소드 데이터 저장
        if self.current_episode > 0:
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.current_step)
            
            # 성능 통계 업데이트
            self.best_reward = max(self.best_reward, self.total_reward)
            self.worst_reward = min(self.worst_reward, self.total_reward)
            self.total_episodes += 1
            self.total_steps += self.current_step
        
        # 새 에피소드 초기화
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
        """종합 시각화 프레임 생성"""
        # 1. 게임 화면 렌더링
        game_frame = self._render_game_frame()
        
        # 2. 그래프 렌더링
        graph_frame = self._render_graphs()
        
        # 3. 통계 정보 렌더링
        stats_frame = self._render_statistics()
        
        # 4. 모든 프레임 수직으로 결합
        if game_frame is not None and graph_frame is not None and stats_frame is not None:
            combined_frame = np.vstack([game_frame, graph_frame, stats_frame])
            return combined_frame
        
        return None
    
    def _render_game_frame(self) -> Optional[np.ndarray]:
        """게임 화면 렌더링 및 오버레이 추가"""
        frame = self.env.render()
        if frame is None:
            return None
        
        # 지정된 크기로 리사이즈
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # 오버레이 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 알고리즘 이름
        cv2.putText(frame, f"{self.algorithm_name}", (10, 30), 
                   font, 1.0, (255, 255, 255), 2)
        
        # 현재 상태 정보
        cv2.putText(frame, f"Episode: {self.current_episode}", (10, 60), 
                   font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Steps: {self.current_step}", (10, 80), 
                   font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Reward: {self.total_reward:.1f}", (10, 100), 
                   font, 0.6, (255, 255, 255), 1)
        
        # 최신 메트릭 표시
        if self.latest_q_value is not None:
            cv2.putText(frame, f"Q-Value: {self.latest_q_value:.2f}", (10, 120), 
                       font, 0.5, (255, 255, 0), 1)
        
        if self.latest_exploration_rate is not None:
            cv2.putText(frame, f"Exploration: {self.latest_exploration_rate:.3f}", (10, 140), 
                       font, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def _render_graphs(self) -> Optional[np.ndarray]:
        """실시간 그래프들 렌더링"""
        # 모든 서브플롯 초기화
        for ax in [self.ax_reward, self.ax_q, self.ax_loss, self.ax_current]:
            ax.clear()
        
        # 1. 에피소드 보상 그래프
        if len(self.episode_rewards) > 0:
            episodes = list(range(len(self.episode_rewards)))
            rewards = list(self.episode_rewards)
            
            self.ax_reward.plot(episodes, rewards, 'b-', alpha=0.7, label='Reward')
            
            # 이동평균
            if len(rewards) >= 10:
                ma = self._moving_average(rewards, 10)
                ma_episodes = episodes[9:]
                self.ax_reward.plot(ma_episodes, ma, 'r-', linewidth=2, label='MA(10)')
            
            self.ax_reward.legend(fontsize=6)
        
        self._setup_subplot(self.ax_reward, "Episode Rewards", "Episode", "Reward")
        
        # 2. Q-값 그래프
        if len(self.q_values) > 0:
            self.ax_q.plot(list(self.q_values), 'purple', alpha=0.7)
            if self.latest_q_value is not None:
                self.ax_q.axhline(y=self.latest_q_value, color='red', linestyle='--', alpha=0.5)
        
        self._setup_subplot(self.ax_q, "Q-Values", "Step", "Q-Value")
        
        # 3. 손실 그래프
        if len(self.losses) > 0:
            self.ax_loss.semilogy(list(self.losses), 'orange', alpha=0.7)
        
        self._setup_subplot(self.ax_loss, "Training Loss", "Step", "Loss (log)")
        
        # 4. 현재 에피소드 진행상황
        if len(self.current_episode_rewards) > 0:
            steps = list(range(len(self.current_episode_rewards)))
            self.ax_current.plot(steps, self.current_episode_rewards, 'g-', linewidth=2)
            self.ax_current.fill_between(steps, 0, self.current_episode_rewards, alpha=0.3)
        
        self._setup_subplot(self.ax_current, "Current Episode Progress", "Step", "Cumulative Reward")
        
        plt.tight_layout(pad=1.0)
        
        # matplotlib figure를 numpy array로 변환
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        graph_array = np.array(img)
        buf.close()
        
        # RGBA를 RGB로 변환
        if graph_array.shape[2] == 4:
            graph_array = graph_array[:, :, :3]
        
        # 지정된 크기로 리사이즈
        graph_array = cv2.resize(graph_array, (self.frame_width, self.graph_height))
        
        return graph_array
    
    def _render_statistics(self) -> np.ndarray:
        """통계 정보 패널 렌더링"""
        # 흰색 배경 생성
        stats_frame = np.ones((self.stats_height, self.frame_width, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)  # 검은색 텍스트
        
        # 기본 통계
        y_pos = 25
        line_height = 20
        
        # 왼쪽 열
        stats_left = [
            f"Algorithm: {self.algorithm_name}",
            f"Total Episodes: {self.total_episodes}",
            f"Total Steps: {self.total_steps}",
            f"Current Episode: {self.current_episode}",
            f"Current Steps: {self.current_step}"
        ]
        
        for i, text in enumerate(stats_left):
            cv2.putText(stats_frame, text, (10, y_pos + i * line_height), 
                       font, 0.5, color, 1)
        
        # 중간 열
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(list(self.episode_rewards))
            std_reward = np.std(list(self.episode_rewards))
            
            stats_middle = [
                f"Best Reward: {self.best_reward:.1f}",
                f"Worst Reward: {self.worst_reward:.1f}",
                f"Average Reward: {avg_reward:.1f}",
                f"Std Reward: {std_reward:.1f}",
                f"Current Reward: {self.total_reward:.1f}"
            ]
            
            for i, text in enumerate(stats_middle):
                cv2.putText(stats_frame, text, (200, y_pos + i * line_height), 
                           font, 0.5, color, 1)
        
        # 오른쪽 열 - 최신 메트릭
        stats_right = []
        if self.latest_q_value is not None:
            stats_right.append(f"Latest Q-Value: {self.latest_q_value:.3f}")
        if self.latest_loss is not None:
            stats_right.append(f"Latest Loss: {self.latest_loss:.3e}")
        if self.latest_exploration_rate is not None:
            stats_right.append(f"Exploration Rate: {self.latest_exploration_rate:.3f}")
        
        # 실행 시간
        elapsed_time = time.time() - self.start_time
        stats_right.append(f"Elapsed Time: {elapsed_time/60:.1f}min")
        
        # 평균 에피소드 길이
        if len(self.episode_lengths) > 0:
            avg_length = np.mean(list(self.episode_lengths))
            stats_right.append(f"Avg Episode Length: {avg_length:.1f}")
        
        for i, text in enumerate(stats_right):
            cv2.putText(stats_frame, text, (400, y_pos + i * line_height), 
                       font, 0.5, color, 1)
        
        return stats_frame
    
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


def create_side_by_side_comparison(env1_wrapper: ComprehensiveVisualizationWrapper,
                                 env2_wrapper: ComprehensiveVisualizationWrapper,
                                 separator_width: int = 20) -> np.ndarray:
    """두 래퍼의 시각화를 나란히 결합"""
    frame1 = env1_wrapper.render()
    frame2 = env2_wrapper.render()
    
    if frame1 is None or frame2 is None:
        return None
    
    # 높이 맞추기
    height = max(frame1.shape[0], frame2.shape[0])
    if frame1.shape[0] != height:
        frame1 = cv2.resize(frame1, (frame1.shape[1], height))
    if frame2.shape[0] != height:
        frame2 = cv2.resize(frame2, (frame2.shape[1], height))
    
    # 중간 구분선
    separator = np.ones((height, separator_width, 3), dtype=np.uint8) * 128
    
    # 결합
    combined = np.hstack([frame1, separator, frame2])
    
    # 상단에 비교 제목 추가
    title_height = 50
    title_frame = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 240
    
    # 제목 텍스트
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "DQN vs DDPG Comprehensive Comparison"
    text_size = cv2.getTextSize(title_text, font, 1.2, 2)[0]
    text_x = (combined.shape[1] - text_size[0]) // 2
    
    cv2.putText(title_frame, title_text, (text_x, 35), font, 1.2, (0, 0, 0), 2)
    
    # 최종 결합
    final_frame = np.vstack([title_frame, combined])
    
    return final_frame