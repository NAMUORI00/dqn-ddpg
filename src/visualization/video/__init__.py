"""
비디오 생성 및 관리 모듈

DQN vs DDPG 학습 과정과 결과를 비디오로 시각화합니다.

모듈 구성:
- pipeline: 학습 과정 비디오 생성 파이프라인
- manager: 실시간 비디오 녹화 관리
- recorder: 이중 품질 비디오 녹화
- generator: 다양한 타입의 비디오 생성기
"""

from .pipeline import VideoRenderingPipeline
from .manager import VideoManager  
from .recorder import DualQualityRecorder
from .generator import ComparisonVideoGenerator, LearningVideoGenerator

__all__ = [
    'VideoRenderingPipeline',
    'VideoManager',
    'DualQualityRecorder', 
    'ComparisonVideoGenerator',
    'LearningVideoGenerator'
]