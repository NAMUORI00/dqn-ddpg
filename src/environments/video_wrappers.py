"""
비디오 녹화를 위한 환경 래퍼들
기존 환경을 수정하지 않고 녹화 기능을 투명하게 추가합니다.
"""

import gymnasium as gym
import numpy as np
import cv2
import os
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import json
import time


class VideoRecordingWrapper(gym.Wrapper):
    """기본 비디오 녹화 래퍼
    
    환경의 렌더링을 캡처하여 비디오로 저장합니다.
    기존 환경 동작에 전혀 영향을 주지 않습니다.
    """
    
    def __init__(self, env: gym.Env, 
                 save_path: str,
                 episode_id: int,
                 fps: int = 30,
                 resolution: Tuple[int, int] = (640, 480),
                 quality: str = "medium"):
        """
        Args:
            env: 래핑할 환경
            save_path: 비디오 저장 경로
            episode_id: 에피소드 ID
            fps: 프레임률
            resolution: 해상도 (width, height)
            quality: 품질 ('low', 'medium', 'high')
        """
        super().__init__(env)
        
        self.save_path = Path(save_path)
        self.episode_id = episode_id
        self.fps = fps
        self.resolution = resolution
        self.quality = quality
        
        # 비디오 라이터 설정
        self.video_writer = None
        self.frames = []
        self.metadata = {
            'episode_id': episode_id,
            'fps': fps,
            'resolution': resolution,
            'quality': quality,
            'start_time': None,
            'end_time': None,
            'total_frames': 0,
            'total_reward': 0,
            'episode_length': 0
        }
        
        # 품질별 설정
        self.quality_settings = {
            'low': {'bitrate': 500, 'crf': 28},
            'medium': {'bitrate': 1500, 'crf': 23},
            'high': {'bitrate': 5000, 'crf': 18}
        }
        
        self._setup_recording()
    
    def _setup_recording(self):
        """녹화 설정 초기화"""
        # 저장 디렉토리 생성
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 환경이 렌더링을 지원하는지 확인
        try:
            # 테스트 렌더링
            test_frame = self.env.render()
            if test_frame is not None:
                self.render_mode = 'rgb_array'
            else:
                # render 모드 설정 시도
                self.env.unwrapped.render_mode = 'rgb_array'
                test_frame = self.env.render()
        except:
            print(f"[WARNING] 환경 {self.env.spec.id}에서 렌더링을 사용할 수 없습니다.")
            self.render_mode = None
            return
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = self.save_path / f"episode_{self.episode_id:03d}.mp4"
        
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps,
            self.resolution
        )
        
        print(f"[INFO] 녹화 시작: {video_path}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """환경 리셋 및 첫 프레임 캡처"""
        self.metadata['start_time'] = time.time()
        self.metadata['total_reward'] = 0
        self.metadata['episode_length'] = 0
        
        state, info = self.env.reset(**kwargs)
        
        # 첫 프레임 캡처
        self._capture_frame()
        
        return state, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행 및 프레임 캡처"""
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # 메타데이터 업데이트
        self.metadata['total_reward'] += reward
        self.metadata['episode_length'] += 1
        
        # 프레임 캡처
        self._capture_frame()
        
        # 에피소드 종료 시 녹화 완료
        if terminated or truncated:
            self._finish_recording()
        
        return state, reward, terminated, truncated, info
    
    def _capture_frame(self):
        """현재 프레임을 캡처하고 저장"""
        if self.video_writer is None:
            return
        
        try:
            # 환경 렌더링
            frame = self.env.render()
            if frame is None:
                return
            
            # 프레임 처리
            if len(frame.shape) == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 해상도 조정
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            
            # 프레임 저장
            self.video_writer.write(frame)
            self.metadata['total_frames'] += 1
            
        except Exception as e:
            print(f"[WARNING] 프레임 캡처 실패: {e}")
    
    def _finish_recording(self):
        """녹화 완료 및 정리"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']
        
        # 메타데이터 저장
        metadata_path = self.save_path / f"episode_{self.episode_id:03d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"[INFO] 녹화 완료: {self.metadata['total_frames']} 프레임, "
              f"보상: {self.metadata['total_reward']:.2f}")
    
    def close(self):
        """리소스 정리"""
        if self.video_writer:
            self._finish_recording()
        super().close()


class RenderableEnv(gym.Wrapper):
    """렌더링 최적화 래퍼
    
    일관된 렌더링 품질과 화면 크기를 보장합니다.
    """
    
    def __init__(self, env: gym.Env, render_width: int = 640, render_height: int = 480):
        super().__init__(env)
        self.render_width = render_width
        self.render_height = render_height
        
        # 환경별 렌더링 설정
        if hasattr(env.unwrapped, 'render_mode'):
            env.unwrapped.render_mode = 'rgb_array'
        
        # CartPole 특별 설정
        if 'CartPole' in str(env.spec.id):
            env.unwrapped.screen_width = render_width
            env.unwrapped.screen_height = render_height
    
    def render(self, mode='rgb_array'):
        """최적화된 렌더링"""
        frame = self.env.render()
        
        if frame is not None and len(frame.shape) == 3:
            # 크기 조정
            if frame.shape[:2] != (self.render_height, self.render_width):
                frame = cv2.resize(frame, (self.render_width, self.render_height))
        
        return frame


class OverlayRenderer(gym.Wrapper):
    """메타데이터 오버레이 렌더러
    
    렌더링된 프레임에 학습 정보를 오버레이합니다.
    """
    
    def __init__(self, env: gym.Env, show_episode: bool = True, 
                 show_reward: bool = True, show_steps: bool = True):
        super().__init__(env)
        self.show_episode = show_episode
        self.show_reward = show_reward  
        self.show_steps = show_steps
        
        # 현재 에피소드 정보
        self.current_episode = 0
        self.current_reward = 0
        self.current_steps = 0
    
    def reset(self, **kwargs):
        """리셋 시 에피소드 정보 업데이트"""
        self.current_episode += 1
        self.current_reward = 0
        self.current_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """스텝 시 정보 업데이트"""
        state, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward += reward
        self.current_steps += 1
        return state, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        """오버레이가 추가된 렌더링"""
        frame = self.env.render(mode)
        
        if frame is not None and len(frame.shape) == 3:
            frame = self._add_overlay(frame)
        
        return frame
    
    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """프레임에 오버레이 추가"""
        frame = frame.copy()
        
        # 텍스트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # 흰색
        thickness = 2
        
        y_offset = 30
        
        # 에피소드 번호
        if self.show_episode:
            text = f"Episode: {self.current_episode}"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        # 현재 보상
        if self.show_reward:
            text = f"Reward: {self.current_reward:.1f}"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        # 스텝 수
        if self.show_steps:
            text = f"Steps: {self.current_steps}"
            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
        
        return frame


def create_video_env(env_name: str, video_config: Optional[Dict] = None) -> gym.Env:
    """비디오 녹화가 가능한 환경 생성
    
    Args:
        env_name: 환경 이름
        video_config: 비디오 설정 (None이면 녹화 안함)
    
    Returns:
        비디오 녹화 기능이 추가된 환경
    """
    # 기본 환경 생성
    env = gym.make(env_name)
    
    # 렌더링 최적화
    env = RenderableEnv(env)
    
    # 비디오 녹화 설정이 있으면 적용
    if video_config:
        # 오버레이 추가
        if video_config.get('show_overlay', True):
            env = OverlayRenderer(env)
        
        # 비디오 녹화 래퍼 추가
        env = VideoRecordingWrapper(
            env,
            save_path=video_config['save_path'],
            episode_id=video_config['episode_id'],
            fps=video_config.get('fps', 30),
            resolution=video_config.get('resolution', (640, 480)),
            quality=video_config.get('quality', 'medium')
        )
    
    return env