"""
이중 품질 비디오 녹화 모듈

고품질과 저품질 비디오를 동시에 녹화하는 기능을 제공합니다.
기존 src/core/dual_recorder.py의 기능을 모듈화하여 개선합니다.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
from typing import Dict, Any, Optional, Tuple
import threading
import time

from ..core.base import BaseVisualizer
from ..core.utils import ensure_path_exists, get_timestamp


class DualQualityRecorder(BaseVisualizer):
    """
    이중 품질 비디오 녹화 클래스
    
    동시에 고품질과 저품질 비디오를 녹화하여
    저장 공간과 품질 사이의 균형을 제공합니다.
    """
    
    def __init__(self, 
                 output_dir: str = "videos",
                 *args, **kwargs):
        super().__init__(output_dir=output_dir, *args, **kwargs)
        
        # 녹화 설정
        self.high_quality_writer = None
        self.low_quality_writer = None
        self.is_recording = False
        
        # 품질 설정
        self.high_res = (1920, 1080)
        self.low_res = (640, 480)
        self.high_fps = 30
        self.low_fps = 15
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """이중 녹화는 별도 인터페이스 사용"""
        return self.get_output_path("dual_recorder_active.txt")
    
    def start_recording(self, 
                       base_filename: str,
                       algorithm: str = "dqn") -> bool:
        """
        이중 품질 녹화 시작
        
        Args:
            base_filename: 기본 파일명
            algorithm: 알고리즘 이름
            
        Returns:
            녹화 시작 성공 여부
        """
        with self.lock:
            if self.is_recording:
                self.logger.warning("이미 녹화 중입니다")
                return False
            
            try:
                timestamp = get_timestamp()
                
                # 고품질 비디오 파일
                high_path = self.get_output_path(
                    f"{base_filename}_high_{timestamp}.mp4",
                    f"{algorithm}/high"
                )
                
                # 저품질 비디오 파일
                low_path = self.get_output_path(
                    f"{base_filename}_low_{timestamp}.mp4", 
                    f"{algorithm}/low"
                )
                
                # 비디오 작성기 초기화
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                self.high_quality_writer = cv2.VideoWriter(
                    high_path, fourcc, self.high_fps, self.high_res
                )
                
                self.low_quality_writer = cv2.VideoWriter(
                    low_path, fourcc, self.low_fps, self.low_res
                )
                
                # 초기화 확인
                if not self.high_quality_writer.isOpened():
                    self.logger.error(f"고품질 비디오 작성기 초기화 실패: {high_path}")
                    return False
                    
                if not self.low_quality_writer.isOpened():
                    self.logger.error(f"저품질 비디오 작성기 초기화 실패: {low_path}")
                    return False
                
                self.is_recording = True
                self.logger.info(f"이중 품질 녹화 시작: {base_filename}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"이중 녹화 시작 실패: {e}")
                self.cleanup()
                return False
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        프레임 추가 (두 품질 모두)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            프레임 추가 성공 여부
        """
        with self.lock:
            if not self.is_recording:
                return False
            
            try:
                # 고품질 프레임
                if self.high_quality_writer:
                    high_frame = cv2.resize(frame, self.high_res)
                    self.high_quality_writer.write(high_frame)
                
                # 저품질 프레임 (프레임 스키핑으로 FPS 조절)
                if self.low_quality_writer:
                    # 간단한 프레임 스키핑 (2프레임마다 1개)
                    if hasattr(self, '_frame_count'):
                        self._frame_count += 1
                    else:
                        self._frame_count = 1
                    
                    if self._frame_count % 2 == 0:
                        low_frame = cv2.resize(frame, self.low_res)
                        self.low_quality_writer.write(low_frame)
                
                return True
                
            except Exception as e:
                self.logger.error(f"프레임 추가 실패: {e}")
                return False
    
    def stop_recording(self) -> Dict[str, Optional[str]]:
        """
        녹화 중지
        
        Returns:
            저장된 파일 경로들
        """
        with self.lock:
            results = {'high': None, 'low': None}
            
            try:
                if self.high_quality_writer:
                    self.high_quality_writer.release()
                    results['high'] = "high_quality_video_saved"
                
                if self.low_quality_writer:
                    self.low_quality_writer.release()
                    results['low'] = "low_quality_video_saved"
                
                self.is_recording = False
                self.logger.info("이중 품질 녹화 완료")
                
            except Exception as e:
                self.logger.error(f"녹화 중지 실패: {e}")
            
            finally:
                self.cleanup()
            
            return results
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.high_quality_writer:
                self.high_quality_writer.release()
                self.high_quality_writer = None
            
            if self.low_quality_writer:
                self.low_quality_writer.release()
                self.low_quality_writer = None
            
            self.is_recording = False
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.cleanup()