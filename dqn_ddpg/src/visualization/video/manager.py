"""
비디오 관리 모듈

실시간 비디오 녹화 및 관리 기능을 제공합니다.
기존 src/core/video_manager.py의 기능을 모듈화하여 개선합니다.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os
import threading
import time
from datetime import datetime

from ..core.base import BaseVisualizer
from ..core.utils import ensure_path_exists, get_timestamp


class VideoManager(BaseVisualizer):
    """
    비디오 관리 클래스
    
    훈련 중 실시간 비디오 녹화, 이중 품질 녹화, 
    성능 기반 녹화 스케줄링 등을 제공합니다.
    """
    
    def __init__(self, 
                 output_dir: str = "videos",
                 enable_dual_recording: bool = True,
                 *args, **kwargs):
        super().__init__(output_dir=output_dir, *args, **kwargs)
        
        # 비디오 설정
        self.video_config = self.config.video
        self.fps = self.video_config.fps
        self.resolution = self.video_config.resolution
        
        # 녹화 설정
        self.enable_dual_recording = enable_dual_recording
        self.is_recording = False
        self.video_writers = {}
        
        # 성능 추적
        self.performance_history = []
        self.recording_triggers = {
            'milestone_episodes': [100, 250, 500, 1000],
            'performance_threshold': 0.8,  # 상위 80% 성능일 때 녹화
            'regular_interval': 50  # 50 에피소드마다 정기 녹화
        }
        
        # 스레드 안전성
        self.recording_lock = threading.Lock()
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """비디오 관리는 별도 인터페이스 사용"""
        return self.get_output_path("video_manager_active.txt")
    
    def start_recording(self, 
                       episode: int,
                       algorithm: str = "dqn",
                       quality: str = "high") -> bool:
        """
        녹화 시작
        
        Args:
            episode: 현재 에피소드 번호
            algorithm: 알고리즘 이름 (dqn, ddpg)
            quality: 비디오 품질 (low, medium, high)
            
        Returns:
            녹화 시작 성공 여부
        """
        with self.recording_lock:
            if self.is_recording:
                self.logger.warning("이미 녹화 중입니다")
                return False
            
            try:
                # 파일명 생성
                timestamp = get_timestamp()
                filename = f"{algorithm}_episode_{episode}_{quality}_{timestamp}.mp4"
                video_path = self.get_output_path(filename, f"{algorithm}/{quality}")
                
                # 비디오 작성기 초기화
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # 품질별 해상도 설정
                resolution = self._get_quality_resolution(quality)
                fps = self._get_quality_fps(quality)
                
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)
                
                if not video_writer.isOpened():
                    self.logger.error(f"비디오 작성기 초기화 실패: {video_path}")
                    return False
                
                # 녹화 정보 저장
                recording_info = {
                    'writer': video_writer,
                    'path': video_path,
                    'episode': episode,
                    'algorithm': algorithm,
                    'quality': quality,
                    'start_time': time.time(),
                    'frame_count': 0
                }
                
                self.video_writers[f"{algorithm}_{quality}"] = recording_info
                self.is_recording = True
                
                self.logger.info(f"녹화 시작: {video_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"녹화 시작 실패: {e}")
                return False
    
    def add_frame(self, 
                  frame: np.ndarray,
                  algorithm: str = "dqn",
                  quality: str = "high") -> bool:
        """
        프레임 추가
        
        Args:
            frame: 비디오 프레임 (OpenCV 형식)
            algorithm: 알고리즘 이름
            quality: 비디오 품질
            
        Returns:
            프레임 추가 성공 여부
        """
        with self.recording_lock:
            recording_key = f"{algorithm}_{quality}"
            
            if recording_key not in self.video_writers:
                return False
            
            recording_info = self.video_writers[recording_key]
            writer = recording_info['writer']
            
            try:
                # 해상도 조정
                target_resolution = self._get_quality_resolution(quality)
                if frame.shape[:2][::-1] != target_resolution:
                    frame = cv2.resize(frame, target_resolution)
                
                # 프레임 쓰기
                writer.write(frame)
                recording_info['frame_count'] += 1
                
                return True
                
            except Exception as e:
                self.logger.error(f"프레임 추가 실패: {e}")
                return False
    
    def stop_recording(self, 
                      algorithm: str = "dqn",
                      quality: str = "high") -> Optional[str]:
        """
        녹화 중지
        
        Args:
            algorithm: 알고리즘 이름
            quality: 비디오 품질
            
        Returns:
            저장된 비디오 파일 경로 (실패시 None)
        """
        with self.recording_lock:
            recording_key = f"{algorithm}_{quality}"
            
            if recording_key not in self.video_writers:
                self.logger.warning(f"활성 녹화를 찾을 수 없습니다: {recording_key}")
                return None
            
            recording_info = self.video_writers[recording_key]
            
            try:
                # 비디오 작성기 해제
                recording_info['writer'].release()
                
                # 녹화 정보 로깅
                duration = time.time() - recording_info['start_time']
                frame_count = recording_info['frame_count']
                video_path = recording_info['path']
                
                self.logger.info(
                    f"녹화 완료: {video_path} "
                    f"({frame_count} 프레임, {duration:.1f}초)"
                )
                
                # 녹화 정보 제거
                del self.video_writers[recording_key]
                
                # 모든 녹화가 종료되었는지 확인
                if not self.video_writers:
                    self.is_recording = False
                
                return video_path
                
            except Exception as e:
                self.logger.error(f"녹화 중지 실패: {e}")
                return None
    
    def should_record(self, episode: int, performance: float) -> Dict[str, bool]:
        """
        녹화 여부 결정
        
        Args:
            episode: 현재 에피소드
            performance: 현재 성능 점수 (0-1 사이)
            
        Returns:
            녹화 여부 딕셔너리 {'high_quality': bool, 'low_quality': bool}
        """
        # 성능 이력 업데이트
        self.performance_history.append(performance)
        
        # 최근 성능의 상위 백분위 계산
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            performance_percentile = (
                sum(1 for p in recent_performance if p <= performance) / len(recent_performance)
            )
        else:
            performance_percentile = 0.5
        
        # 녹화 결정 로직
        record_high = (
            episode in self.recording_triggers['milestone_episodes'] or
            performance_percentile >= self.recording_triggers['performance_threshold']
        )
        
        record_low = (
            episode % self.recording_triggers['regular_interval'] == 0 or
            record_high
        )
        
        return {
            'high_quality': record_high,
            'low_quality': record_low
        }
    
    def start_dual_recording(self, episode: int, algorithm: str = "dqn") -> bool:
        """
        이중 품질 녹화 시작
        
        Args:
            episode: 에피소드 번호
            algorithm: 알고리즘 이름
            
        Returns:
            녹화 시작 성공 여부
        """
        if not self.enable_dual_recording:
            return self.start_recording(episode, algorithm, "high")
        
        success_high = self.start_recording(episode, algorithm, "high")
        success_low = self.start_recording(episode, algorithm, "low")
        
        return success_high or success_low
    
    def stop_dual_recording(self, algorithm: str = "dqn") -> Dict[str, Optional[str]]:
        """
        이중 품질 녹화 중지
        
        Args:
            algorithm: 알고리즘 이름
            
        Returns:
            저장된 파일 경로들
        """
        results = {}
        
        if self.enable_dual_recording:
            results['high'] = self.stop_recording(algorithm, "high")
            results['low'] = self.stop_recording(algorithm, "low")
        else:
            results['high'] = self.stop_recording(algorithm, "high")
        
        return results
    
    def cleanup(self):
        """리소스 정리"""
        with self.recording_lock:
            for recording_key, recording_info in self.video_writers.items():
                try:
                    recording_info['writer'].release()
                    self.logger.info(f"녹화 강제 종료: {recording_info['path']}")
                except:
                    pass
            
            self.video_writers.clear()
            self.is_recording = False
    
    def _get_quality_resolution(self, quality: str) -> Tuple[int, int]:
        """품질별 해상도 반환"""
        quality_map = {
            'low': (640, 480),
            'medium': (1280, 720),
            'high': (1920, 1080)
        }
        return quality_map.get(quality, self.resolution)
    
    def _get_quality_fps(self, quality: str) -> int:
        """품질별 프레임률 반환"""
        quality_map = {
            'low': 15,
            'medium': 24,
            'high': 30
        }
        return quality_map.get(quality, self.fps)
    
    def get_recording_status(self) -> Dict[str, Any]:
        """현재 녹화 상태 반환"""
        with self.recording_lock:
            status = {
                'is_recording': self.is_recording,
                'active_recordings': len(self.video_writers),
                'recordings': {}
            }
            
            for key, info in self.video_writers.items():
                status['recordings'][key] = {
                    'episode': info['episode'],
                    'algorithm': info['algorithm'],
                    'quality': info['quality'],
                    'duration': time.time() - info['start_time'],
                    'frame_count': info['frame_count']
                }
            
            return status
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.cleanup()