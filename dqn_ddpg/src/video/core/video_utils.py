"""
비디오 처리 공통 유틸리티
모든 비디오 관련 기능에서 공통으로 사용하는 함수들을 모아놓은 모듈
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os


class VideoEncoder:
    """비디오 인코딩 관련 유틸리티"""
    
    @staticmethod
    def save_with_opencv(frames: List[np.ndarray], 
                        output_path: str, 
                        fps: int = 30) -> bool:
        """
        OpenCV를 사용하여 프레임들을 비디오로 저장
        
        Args:
            frames: 프레임 이미지 배열 리스트
            output_path: 출력 파일 경로
            fps: 프레임 레이트
            
        Returns:
            bool: 저장 성공 여부
        """
        if not frames:
            print("[ERROR] 저장할 프레임이 없습니다.")
            return False
        
        # 출력 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 첫 번째 프레임에서 크기 정보 가져오기
        height, width = frames[0].shape[:2]
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"[ERROR] 비디오 라이터를 열 수 없습니다: {output_path}")
            return False
        
        # 프레임들 쓰기
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    
    @staticmethod
    def save_frames_to_video(frames_dir: str, 
                           output_path: str, 
                           fps: int = 30,
                           frame_pattern: str = "frame_*.png") -> bool:
        """
        디렉토리의 프레임 이미지들을 비디오로 저장
        
        Args:
            frames_dir: 프레임 이미지가 있는 디렉토리
            output_path: 출력 비디오 경로
            fps: 프레임 레이트
            frame_pattern: 프레임 파일 패턴
            
        Returns:
            bool: 저장 성공 여부
        """
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob(frame_pattern))
        
        if not frame_files:
            print(f"[ERROR] {frames_dir}에서 프레임을 찾을 수 없습니다.")
            return False
        
        # 첫 번째 프레임에서 크기 정보
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"[ERROR] 첫 번째 프레임을 읽을 수 없습니다: {frame_files[0]}")
            return False
            
        height, width = first_frame.shape[:2]
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 모든 프레임 처리
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
        
        out.release()
        
        # 프레임 파일들 정리 (선택적)
        for frame_file in frame_files:
            frame_file.unlink()
        frames_path.rmdir()
        
        return True


class VideoLayoutUtils:
    """비디오 레이아웃 관련 유틸리티"""
    
    @staticmethod
    def create_side_by_side(left_frame: np.ndarray, 
                           right_frame: np.ndarray,
                           gap: int = 20,
                           labels: Tuple[str, str] = None) -> np.ndarray:
        """
        두 프레임을 나란히 배치
        
        Args:
            left_frame: 왼쪽 프레임
            right_frame: 오른쪽 프레임  
            gap: 프레임 사이 간격
            labels: (왼쪽 라벨, 오른쪽 라벨)
            
        Returns:
            np.ndarray: 합성된 프레임
        """
        h1, w1 = left_frame.shape[:2]
        h2, w2 = right_frame.shape[:2]
        
        # 높이 맞추기
        max_height = max(h1, h2)
        total_width = w1 + w2 + gap
        
        # 합성 프레임 생성
        result = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        
        # 프레임 배치
        y1 = (max_height - h1) // 2
        y2 = (max_height - h2) // 2
        
        result[y1:y1+h1, 0:w1] = left_frame
        result[y2:y2+h2, w1+gap:w1+gap+w2] = right_frame
        
        # 라벨 추가
        if labels:
            left_label, right_label = labels
            
            # 왼쪽 라벨
            cv2.putText(result, left_label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 오른쪽 라벨  
            cv2.putText(result, right_label, (w1 + gap + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result
    
    @staticmethod
    def add_text_overlay(frame: np.ndarray, 
                        text: str, 
                        position: Tuple[int, int] = (10, 30),
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 2) -> np.ndarray:
        """
        프레임에 텍스트 오버레이 추가
        
        Args:
            frame: 원본 프레임
            text: 추가할 텍스트
            position: 텍스트 위치 (x, y)
            font_scale: 폰트 크기
            color: 텍스트 색상 (B, G, R)
            thickness: 텍스트 두께
            
        Returns:
            np.ndarray: 텍스트가 추가된 프레임
        """
        result = frame.copy()
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return result
    
    @staticmethod
    def add_progress_bar(frame: np.ndarray,
                        progress: float,
                        position: Tuple[int, int] = (10, None),
                        size: Tuple[int, int] = (200, 10),
                        bg_color: Tuple[int, int, int] = (50, 50, 50),
                        fg_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        프레임에 진행률 바 추가
        
        Args:
            frame: 원본 프레임
            progress: 진행률 (0.0 ~ 1.0)
            position: 바 위치 (x, y) - y가 None이면 하단에 배치
            size: 바 크기 (width, height)
            bg_color: 배경 색상
            fg_color: 전경 색상
            
        Returns:
            np.ndarray: 진행률 바가 추가된 프레임
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        bar_w, bar_h = size
        x, y = position
        
        if y is None:
            y = h - bar_h - 20  # 하단에서 20px 위
        
        # 배경 바
        cv2.rectangle(result, (x, y), (x + bar_w, y + bar_h), bg_color, -1)
        
        # 진행률 바
        progress_w = int(bar_w * max(0, min(1, progress)))
        if progress_w > 0:
            cv2.rectangle(result, (x, y), (x + progress_w, y + bar_h), fg_color, -1)
        
        # 진행률 텍스트
        progress_text = f"{progress * 100:.1f}%"
        cv2.putText(result, progress_text, (x + bar_w + 10, y + bar_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result


class SampleDataGenerator:
    """샘플 데이터 생성 유틸리티"""
    
    @staticmethod
    def create_learning_curves(algorithm: str, episodes: int = 200) -> dict:
        """
        샘플 학습 곡선 데이터 생성
        
        Args:
            algorithm: 'dqn' 또는 'ddpg'
            episodes: 에피소드 수
            
        Returns:
            dict: 학습 메트릭 데이터
        """
        np.random.seed(42)
        
        if algorithm.lower() == 'dqn':
            # CartPole 학습 곡선
            rewards = []
            for i in range(episodes):
                base_reward = min(450, 50 + i * 2)
                noise = np.random.normal(0, 20)
                reward = max(0, base_reward + noise)
                rewards.append(reward)
        else:  # ddpg
            # Pendulum 학습 곡선
            rewards = []
            for i in range(episodes):
                base_reward = max(-200, -800 + i * 3)
                noise = np.random.normal(0, 30)
                reward = min(0, base_reward + noise)
                rewards.append(reward)
        
        return {
            'episode_rewards': rewards,
            'episode_lengths': [np.random.randint(50, 200) for _ in range(episodes)],
            'training_losses': [np.random.exponential(0.1) for _ in range(episodes * 5)],
            'q_values': [np.random.normal(0, 1) for _ in range(episodes * 5)]
        }
    
    @staticmethod
    def create_sample_frame(algorithm: str, 
                           episode: int,
                           step: int,
                           size: Tuple[int, int] = (600, 400)) -> np.ndarray:
        """
        샘플 게임 프레임 생성
        
        Args:
            algorithm: 'dqn' 또는 'ddpg'
            episode: 에피소드 번호
            step: 스텝 번호
            size: 프레임 크기 (width, height)
            
        Returns:
            np.ndarray: 생성된 프레임
        """
        width, height = size
        
        if algorithm.lower() == 'dqn':
            # CartPole 시뮬레이션
            frame = np.full((height, width, 3), (20, 120, 20), dtype=np.uint8)
            
            # 카트 위치 (진동)
            cart_x = width // 2 + int(50 * np.sin(step * 0.05))
            cart_y = height - 100
            
            # 폴 각도
            pole_angle = 0.3 * np.sin(step * 0.08)
            
            # 카트 그리기
            cv2.rectangle(frame, (cart_x - 30, cart_y - 20),
                         (cart_x + 30, cart_y), (100, 100, 100), -1)
            
            # 폴 그리기
            pole_length = 100
            pole_end_x = cart_x + int(pole_length * np.sin(pole_angle))
            pole_end_y = cart_y - 20 - int(pole_length * np.cos(pole_angle))
            cv2.line(frame, (cart_x, cart_y - 20), 
                    (pole_end_x, pole_end_y), (200, 100, 50), 5)
            
            # 정보 표시
            env_name = "CartPole-v1"
            color = (0, 255, 136)
            
        else:  # ddpg
            # Pendulum 시뮬레이션
            frame = np.full((height, width, 3), (50, 20, 20), dtype=np.uint8)
            
            center_x, center_y = width // 2, height // 2
            angle = step * 0.05
            
            # 원 그리기
            cv2.circle(frame, (center_x, center_y), 100, (80, 80, 80), 2)
            
            # 진자 그리기
            pend_x = center_x + int(80 * np.sin(angle))
            pend_y = center_y + int(80 * np.cos(angle))
            cv2.line(frame, (center_x, center_y), (pend_x, pend_y), (200, 200, 200), 4)
            cv2.circle(frame, (pend_x, pend_y), 15, (150, 150, 250), -1)
            
            # 정보 표시
            env_name = "Pendulum-v1"
            color = (255, 107, 107)
        
        # 텍스트 정보 추가
        cv2.putText(frame, f"{algorithm.upper()} - {env_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Episode: {episode}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Step: {step}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


def cleanup_temp_files(temp_dir: str):
    """임시 파일들 정리"""
    temp_path = Path(temp_dir)
    if temp_path.exists():
        try:
            for file_path in temp_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            temp_path.rmdir()
            print(f"[INFO] 임시 파일 정리 완료: {temp_dir}")
        except Exception as e:
            print(f"[WARNING] 임시 파일 정리 중 오류: {e}")


def check_video_dependencies() -> dict:
    """비디오 처리에 필요한 의존성 확인"""
    deps = {}
    
    try:
        import cv2
        deps['opencv'] = cv2.__version__
    except ImportError:
        deps['opencv'] = None
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
    
    try:
        import matplotlib
        deps['matplotlib'] = matplotlib.__version__
    except ImportError:
        deps['matplotlib'] = None
    
    # ffmpeg 확인
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        deps['ffmpeg'] = result.returncode == 0
    except:
        deps['ffmpeg'] = False
    
    return deps