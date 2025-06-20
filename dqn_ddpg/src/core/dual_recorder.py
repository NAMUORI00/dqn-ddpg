"""
이중 비디오 녹화 시스템
전체 에피소드 저품질 녹화 + 선택적 고품질 녹화를 동시에 수행
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import cv2
import json

from .video_manager import VideoConfig, VideoManager
from ..environments.video_wrappers import VideoRecordingWrapper, RenderableEnv, OverlayRenderer


@dataclass
class DualRecordingConfig:
    """이중 녹화 설정"""
    # 전체 녹화 설정 (저품질)
    full_fps: int = 15
    full_resolution: Tuple[int, int] = (320, 240)
    full_quality: str = "low"
    
    # 선택적 녹화 설정 (고품질)
    selective_fps: int = 30
    selective_resolution: Tuple[int, int] = (640, 480)
    selective_quality: str = "high"
    
    # 버퍼 관리
    max_buffer_size: int = 300  # 최대 프레임 버퍼 크기
    buffer_cleanup_threshold: float = 0.8  # 버퍼 정리 임계값
    
    # 성능 최적화
    async_writing: bool = True
    compression_threads: int = 2
    
    @classmethod
    def from_yaml_config(cls, config_dict: Dict) -> 'DualRecordingConfig':
        """YAML 설정에서 이중 녹화 설정 생성"""
        dual_config = config_dict.get('dual_recording', {})
        
        full_config = dual_config.get('full_recording', {})
        selective_config = dual_config.get('selective_recording', {})
        
        return cls(
            full_fps=full_config.get('fps', 15),
            full_resolution=tuple(full_config.get('resolution', [320, 240])),
            full_quality=full_config.get('quality', 'low'),
            
            selective_fps=selective_config.get('fps', 30),
            selective_resolution=tuple(selective_config.get('resolution', [640, 480])),
            selective_quality=selective_config.get('quality', 'high')
        )


class AsyncVideoWriter:
    """비동기 비디오 작성기
    
    별도 스레드에서 프레임을 저장하여 메인 스레드 성능 향상
    """
    
    def __init__(self, output_path: str, fps: int, resolution: Tuple[int, int], 
                 quality: str = "medium"):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.quality = quality
        
        # 스레드 관리
        self.frame_queue = []
        self.is_recording = False
        self.writer_thread = None
        self.lock = threading.Lock()
        
        # 비디오 라이터
        self.video_writer = None
        self._setup_writer()
    
    def _setup_writer(self):
        """비디오 라이터 초기화"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.resolution
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"비디오 라이터를 열 수 없습니다: {self.output_path}")
    
    def start_recording(self):
        """녹화 시작"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.daemon = True
        self.writer_thread.start()
    
    def add_frame(self, frame: np.ndarray):
        """프레임 추가 (비동기)"""
        if not self.is_recording:
            return
        
        # 프레임 전처리
        processed_frame = self._preprocess_frame(frame)
        
        with self.lock:
            self.frame_queue.append(processed_frame)
            
            # 큐 크기 제한
            if len(self.frame_queue) > 300:  # 10초 분량 (30fps 기준)
                self.frame_queue.pop(0)  # 오래된 프레임 제거
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        if frame is None:
            return None
        
        # RGB to BGR 변환
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 해상도 조정
        if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
            frame = cv2.resize(frame, self.resolution)
        
        return frame
    
    def _writer_worker(self):
        """백그라운드 프레임 작성 작업자"""
        while self.is_recording or len(self.frame_queue) > 0:
            frames_to_write = []
            
            # 프레임 배치 수집
            with self.lock:
                if self.frame_queue:
                    frames_to_write = self.frame_queue.copy()
                    self.frame_queue.clear()
            
            # 프레임 작성
            for frame in frames_to_write:
                if frame is not None and self.video_writer:
                    self.video_writer.write(frame)
            
            # CPU 부하 방지
            time.sleep(0.01)
    
    def stop_recording(self):
        """녹화 종료"""
        self.is_recording = False
        
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def __del__(self):
        """소멸자"""
        self.stop_recording()


class DualVideoRecorder:
    """이중 비디오 녹화기
    
    전체 에피소드 저품질 녹화와 선택적 고품질 녹화를 동시에 수행
    """
    
    def __init__(self, video_manager: VideoManager, config: DualRecordingConfig):
        self.video_manager = video_manager
        self.config = config
        
        # 녹화 상태
        self.is_recording = False
        self.current_episode = 0
        self.current_algorithm = None
        
        # 비디오 라이터들
        self.full_writer = None  # 전체 녹화 (항상 활성)
        self.selective_writer = None  # 선택적 녹화 (필요시만 활성)
        
        # 성능 측정
        self.recording_stats = {
            'total_episodes': 0,
            'full_recordings': 0,
            'selective_recordings': 0,
            'total_frames_processed': 0,
            'average_processing_time': 0
        }
        
        print("[INFO] 이중 비디오 녹화기 초기화 완료")
    
    def start_episode_recording(self, algorithm: str, episode_id: int, 
                              record_selective: bool = False) -> bool:
        """에피소드 녹화 시작
        
        Args:
            algorithm: 알고리즘 이름 ('dqn' 또는 'ddpg')
            episode_id: 에피소드 ID
            record_selective: 선택적 고품질 녹화 여부
            
        Returns:
            녹화 시작 성공 여부
        """
        if self.is_recording:
            self.stop_episode_recording()
        
        self.current_episode = episode_id
        self.current_algorithm = algorithm
        
        try:
            # 전체 녹화 (항상 실행)
            full_path = self._get_video_path(algorithm, "full", episode_id)
            self.full_writer = AsyncVideoWriter(
                str(full_path),
                self.config.full_fps,
                self.config.full_resolution,
                self.config.full_quality
            )
            self.full_writer.start_recording()
            
            # 선택적 녹화 (필요시만)
            if record_selective:
                selective_path = self._get_video_path(algorithm, "highlights", episode_id)
                self.selective_writer = AsyncVideoWriter(
                    str(selective_path),
                    self.config.selective_fps,
                    self.config.selective_resolution,
                    self.config.selective_quality
                )
                self.selective_writer.start_recording()
                print(f"[INFO] 선택적 고품질 녹화 시작: Episode {episode_id}")
            
            self.is_recording = True
            self.recording_stats['total_episodes'] += 1
            
            if record_selective:
                self.recording_stats['selective_recordings'] += 1
            
            print(f"[INFO] 이중 녹화 시작: {algorithm} Episode {episode_id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 녹화 시작 실패: {e}")
            self.stop_episode_recording()
            return False
    
    def add_frame(self, frame: np.ndarray):
        """프레임 추가"""
        if not self.is_recording or frame is None:
            return
        
        start_time = time.time()
        
        try:
            # 전체 녹화에 프레임 추가
            if self.full_writer:
                self.full_writer.add_frame(frame)
            
            # 선택적 녹화에 프레임 추가
            if self.selective_writer:
                self.selective_writer.add_frame(frame)
            
            # 통계 업데이트
            self.recording_stats['total_frames_processed'] += 1
            processing_time = time.time() - start_time
            self._update_processing_time(processing_time)
            
        except Exception as e:
            print(f"[WARNING] 프레임 처리 오류: {e}")
    
    def stop_episode_recording(self, episode_metadata: Optional[Dict] = None):
        """에피소드 녹화 종료"""
        if not self.is_recording:
            return
        
        try:
            # 비디오 라이터 종료
            if self.full_writer:
                self.full_writer.stop_recording()
                self.recording_stats['full_recordings'] += 1
            
            if self.selective_writer:
                self.selective_writer.stop_recording()
            
            # 메타데이터 저장
            if episode_metadata:
                self._save_episode_metadata(episode_metadata)
            
            # 정리
            self.full_writer = None
            self.selective_writer = None
            self.is_recording = False
            
            print(f"[INFO] 이중 녹화 완료: {self.current_algorithm} Episode {self.current_episode}")
            
        except Exception as e:
            print(f"[ERROR] 녹화 종료 오류: {e}")
    
    def _get_video_path(self, algorithm: str, video_type: str, episode_id: int) -> Path:
        """비디오 파일 경로 생성"""
        base_path = Path(self.video_manager.config.save_base_path)
        return base_path / algorithm / video_type / f"episode_{episode_id:03d}.mp4"
    
    def _save_episode_metadata(self, metadata: Dict):
        """에피소드 메타데이터 저장"""
        try:
            # 기본 메타데이터에 녹화 정보 추가
            recording_metadata = {
                **metadata,
                'recording_config': {
                    'dual_recording': True,
                    'full_recording': {
                        'fps': self.config.full_fps,
                        'resolution': self.config.full_resolution,
                        'quality': self.config.full_quality
                    },
                    'selective_recording': {
                        'fps': self.config.selective_fps,
                        'resolution': self.config.selective_resolution,
                        'quality': self.config.selective_quality,
                        'enabled': self.selective_writer is not None
                    }
                }
            }
            
            # 전체 녹화 메타데이터
            full_metadata_path = self._get_video_path(
                self.current_algorithm, "full", self.current_episode
            ).with_suffix('.json')
            
            with open(full_metadata_path, 'w') as f:
                json.dump(recording_metadata, f, indent=2)
            
            # 선택적 녹화 메타데이터 (있는 경우)
            if self.selective_writer:
                selective_metadata_path = self._get_video_path(
                    self.current_algorithm, "highlights", self.current_episode
                ).with_suffix('.json')
                
                with open(selective_metadata_path, 'w') as f:
                    json.dump(recording_metadata, f, indent=2)
            
        except Exception as e:
            print(f"[WARNING] 메타데이터 저장 실패: {e}")
    
    def _update_processing_time(self, processing_time: float):
        """프레임 처리 시간 통계 업데이트"""
        current_avg = self.recording_stats['average_processing_time']
        total_frames = self.recording_stats['total_frames_processed']
        
        # 이동 평균 계산
        new_avg = ((current_avg * (total_frames - 1)) + processing_time) / total_frames
        self.recording_stats['average_processing_time'] = new_avg
    
    def get_recording_stats(self) -> Dict:
        """녹화 통계 반환"""
        return {
            **self.recording_stats,
            'current_episode': self.current_episode,
            'current_algorithm': self.current_algorithm,
            'is_recording': self.is_recording
        }
    
    def cleanup_old_recordings(self, keep_recent: int = 10):
        """오래된 녹화 파일 정리"""
        print(f"[INFO] 오래된 녹화 파일 정리 시작 (최근 {keep_recent}개 유지)")
        
        for algorithm in ['dqn', 'ddpg']:
            for video_type in ['full', 'highlights']:
                video_dir = Path(self.video_manager.config.save_base_path) / algorithm / video_type
                
                if not video_dir.exists():
                    continue
                
                # 비디오 파일 목록 수집
                video_files = list(video_dir.glob("episode_*.mp4"))
                video_files.sort(key=lambda x: x.name)  # 이름순 정렬
                
                # 오래된 파일 삭제
                if len(video_files) > keep_recent:
                    files_to_delete = video_files[:-keep_recent]
                    
                    for video_file in files_to_delete:
                        try:
                            # 비디오 파일 삭제
                            video_file.unlink()
                            
                            # 메타데이터 파일도 삭제
                            metadata_file = video_file.with_suffix('.json')
                            if metadata_file.exists():
                                metadata_file.unlink()
                            
                            print(f"[INFO] 삭제됨: {video_file.name}")
                            
                        except Exception as e:
                            print(f"[WARNING] 파일 삭제 실패 {video_file}: {e}")
        
        print("[INFO] 녹화 파일 정리 완료")
    
    def __del__(self):
        """소멸자"""
        if self.is_recording:
            self.stop_episode_recording()


class DualRecordingEnvironmentWrapper:
    """이중 녹화 환경 래퍼
    
    기존 환경에 이중 녹화 기능을 투명하게 추가
    """
    
    def __init__(self, env, dual_recorder: DualVideoRecorder, 
                 algorithm: str, episode_id: int, record_selective: bool = False):
        self.env = env
        self.dual_recorder = dual_recorder
        self.algorithm = algorithm
        self.episode_id = episode_id
        self.record_selective = record_selective
        
        # 에피소드 메타데이터
        self.episode_metadata = {
            'algorithm': algorithm,
            'episode_id': episode_id,
            'start_time': None,
            'end_time': None,
            'total_reward': 0,
            'episode_length': 0,
            'record_selective': record_selective
        }
    
    def reset(self, **kwargs):
        """환경 리셋 및 녹화 시작"""
        # 녹화 시작
        success = self.dual_recorder.start_episode_recording(
            self.algorithm, self.episode_id, self.record_selective
        )
        
        if not success:
            print(f"[WARNING] 녹화 시작 실패: Episode {self.episode_id}")
        
        # 환경 리셋
        result = self.env.reset(**kwargs)
        
        # 메타데이터 초기화
        self.episode_metadata['start_time'] = time.time()
        self.episode_metadata['total_reward'] = 0
        self.episode_metadata['episode_length'] = 0
        
        # 첫 프레임 캡처
        self._capture_frame()
        
        return result
    
    def step(self, action):
        """환경 스텝 및 프레임 캡처"""
        result = self.env.step(action)
        state, reward, terminated, truncated, info = result
        
        # 메타데이터 업데이트
        self.episode_metadata['total_reward'] += reward
        self.episode_metadata['episode_length'] += 1
        
        # 프레임 캡처
        self._capture_frame()
        
        # 에피소드 종료 시 녹화 완료
        if terminated or truncated:
            self.episode_metadata['end_time'] = time.time()
            self.dual_recorder.stop_episode_recording(self.episode_metadata)
        
        return result
    
    def _capture_frame(self):
        """현재 프레임 캡처"""
        try:
            frame = self.env.render()
            if frame is not None:
                self.dual_recorder.add_frame(frame)
        except Exception as e:
            print(f"[WARNING] 프레임 캡처 실패: {e}")
    
    def close(self):
        """환경 종료"""
        if self.dual_recorder.is_recording:
            self.dual_recorder.stop_episode_recording(self.episode_metadata)
        
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __getattr__(self, name):
        """환경 속성 위임"""
        return getattr(self.env, name)


def create_dual_recording_env(env_name: str, dual_recorder: DualVideoRecorder,
                            algorithm: str, episode_id: int, 
                            record_selective: bool = False):
    """이중 녹화 환경 생성
    
    Args:
        env_name: 환경 이름
        dual_recorder: 이중 녹화기
        algorithm: 알고리즘 이름
        episode_id: 에피소드 ID
        record_selective: 선택적 고품질 녹화 여부
    
    Returns:
        이중 녹화 기능을 가진 환경
    """
    import gymnasium as gym
    
    # 기본 환경 생성 (렌더링 모드 명시)
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 렌더링 최적화
    env = RenderableEnv(env)
    
    # 오버레이 추가
    env = OverlayRenderer(env)
    
    # 이중 녹화 래퍼 추가
    env = DualRecordingEnvironmentWrapper(
        env, dual_recorder, algorithm, episode_id, record_selective
    )
    
    return env