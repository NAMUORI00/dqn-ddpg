"""
비디오 생성기 모듈

다양한 타입의 교육용 비디오를 생성하는 클래스들을 제공합니다.
비교 비디오, 학습 과정 비디오, 프레젠테이션 비디오 등을 포함합니다.

새로운 출력 구조:
- 비디오 파일 확장자별 자동 분류 (MP4 -> output/mp4/videos/, AVI -> output/avi/videos/ 등)
- 비디오 타입별 세부 디렉토리 지원 (comparison/, learning/, presentation/)
- get_output_path_by_extension()으로 비디오 파일 경로 자동 관리
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import os

from ..core.base import BaseVisualizer
from ..core.utils import (
    smooth_data, validate_experiment_data,
    get_output_path_by_extension, create_structured_filename
)
from .pipeline import VideoRenderingPipeline


class ComparisonVideoGenerator(BaseVisualizer):
    """
    알고리즘 비교 비디오 생성 클래스
    
    DQN과 DDPG의 성능을 시각적으로 비교하는 비디오를 생성합니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = VideoRenderingPipeline(output_dir=self.output_dir, config=self.config)
    
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        비교 비디오 생성
        
        Args:
            data: 비교할 데이터
            **kwargs: 비디오 생성 옵션
            
        Returns:
            생성된 비디오 파일 경로
        """
        return self.create_side_by_side_comparison(
            data.get('dqn', {}), 
            data.get('ddpg', {}), 
            **kwargs
        )
    
    def create_side_by_side_comparison(self,
                                     dqn_data: Dict[str, Any],
                                     ddpg_data: Dict[str, Any],
                                     save_filename: str = "side_by_side_comparison.mp4",
                                     duration_seconds: int = 30) -> str:
        """
        나란히 비교 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이
            
        Returns:
            생성된 비디오 파일 경로
        """
        # 확장자 기반 출력 구조 사용 - 비디오 파일을 적절한 디렉토리에 자동 저장
        video_path = get_output_path_by_extension(save_filename, "comparison", self.config)
        
        # 간단한 비교 비디오 생성 (실제 구현에서는 더 복잡할 것)
        self.logger.info(f"나란히 비교 비디오 생성 시작: {save_filename}")
        
        # Pipeline을 사용하여 비교 비디오 생성
        return self.pipeline.create_comparison_video(
            dqn_data, ddpg_data, save_filename, duration_seconds
        )
    
    def create_performance_race(self,
                              dqn_data: Dict[str, Any],
                              ddpg_data: Dict[str, Any],
                              save_filename: str = "performance_race.mp4") -> str:
        """
        성능 경주 스타일 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            save_filename: 저장할 파일명
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"성능 경주 비디오 생성: {save_filename}")
        
        # 성능 경주 비디오 로직 (간소화)
        # 실제로는 막대 그래프가 경주하는 형태의 애니메이션
        
        return video_path


class LearningVideoGenerator(BaseVisualizer):
    """
    학습 과정 비디오 생성 클래스
    
    학습 진행 과정을 단계별로 보여주는 교육용 비디오를 생성합니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = VideoRenderingPipeline(output_dir=self.output_dir, config=self.config)
    
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        학습 과정 비디오 생성
        
        Args:
            data: 학습 데이터
            **kwargs: 비디오 생성 옵션
            
        Returns:
            생성된 비디오 파일 경로
        """
        return self.create_complete_learning_journey(
            data.get('dqn', {}),
            data.get('ddpg', {}),
            **kwargs
        )
    
    def create_complete_learning_journey(self,
                                       dqn_data: Dict[str, Any],
                                       ddpg_data: Dict[str, Any],
                                       save_filename: str = "complete_learning_journey.mp4",
                                       duration_seconds: int = 120) -> str:
        """
        완전한 학습 여정 비디오 생성
        
        Args:
            dqn_data: DQN 학습 데이터
            ddpg_data: DDPG 학습 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이
            
        Returns:
            생성된 비디오 파일 경로
        """
        self.logger.info(f"완전한 학습 여정 비디오 생성: {save_filename}")
        
        # Pipeline을 사용하여 학습 애니메이션 생성
        return self.pipeline.create_learning_animation(
            dqn_data, ddpg_data, save_filename, duration_seconds
        )
    
    def create_milestone_highlights(self,
                                  dqn_data: Dict[str, Any],
                                  ddpg_data: Dict[str, Any],
                                  milestones: List[int] = [100, 250, 500],
                                  save_filename: str = "milestone_highlights.mp4") -> str:
        """
        마일스톤 하이라이트 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            milestones: 하이라이트할 에피소드 리스트
            save_filename: 저장할 파일명
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"마일스톤 하이라이트 비디오 생성: {save_filename}")
        
        # 마일스톤별 성능 변화를 보여주는 비디오 생성 로직
        # 간소화된 구현
        
        return video_path
    
    def create_algorithm_explanation(self,
                                   save_filename: str = "algorithm_explanation.mp4",
                                   include_theory: bool = True) -> str:
        """
        알고리즘 설명 비디오 생성
        
        Args:
            save_filename: 저장할 파일명
            include_theory: 이론 설명 포함 여부
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"알고리즘 설명 비디오 생성: {save_filename}")
        
        # DQN과 DDPG의 핵심 개념을 설명하는 비디오 생성
        # 실제로는 텍스트 애니메이션, 다이어그램 등을 포함
        
        return video_path


class PresentationVideoGenerator(BaseVisualizer):
    """
    프레젠테이션용 비디오 생성 클래스
    
    학술 발표나 교육용 프레젠테이션에 적합한 고품질 비디오를 생성합니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 프레젠테이션용 고품질 설정
        self.presentation_config = self.config.presentation
        self.high_dpi = 300
        self.professional_colors = {
            'primary': self.presentation_config.primary_color,
            'secondary': self.presentation_config.secondary_color,
            'accent': self.presentation_config.accent_color
        }
    
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        프레젠테이션 비디오 생성
        
        Args:
            data: 프레젠테이션 데이터
            **kwargs: 비디오 생성 옵션
            
        Returns:
            생성된 비디오 파일 경로
        """
        return self.create_executive_summary(
            data.get('dqn', {}),
            data.get('ddpg', {}),
            **kwargs
        )
    
    def create_executive_summary(self,
                               dqn_data: Dict[str, Any],
                               ddpg_data: Dict[str, Any],
                               save_filename: str = "executive_summary.mp4",
                               duration_seconds: int = 60) -> str:
        """
        경영진 요약 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"경영진 요약 비디오 생성: {save_filename}")
        
        # 핵심 결과와 인사이트를 간결하게 보여주는 비디오
        # 프로페셔널한 스타일과 명확한 메시지 전달
        
        return video_path
    
    def create_technical_deep_dive(self,
                                 dqn_data: Dict[str, Any],
                                 ddpg_data: Dict[str, Any],
                                 save_filename: str = "technical_deep_dive.mp4",
                                 duration_seconds: int = 180) -> str:
        """
        기술적 심화 분석 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            save_filename: 저장할 파일명
            duration_seconds: 비디오 길이
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"기술적 심화 분석 비디오 생성: {save_filename}")
        
        # 상세한 기술적 분석과 메트릭을 포함하는 전문가용 비디오
        
        return video_path
    
    def create_conference_presentation(self,
                                     dqn_data: Dict[str, Any],
                                     ddpg_data: Dict[str, Any],
                                     conference_style: str = "academic",
                                     save_filename: str = "conference_presentation.mp4") -> str:
        """
        컨퍼런스 발표용 비디오 생성
        
        Args:
            dqn_data: DQN 데이터
            ddpg_data: DDPG 데이터
            conference_style: 컨퍼런스 스타일 (academic, industry)
            save_filename: 저장할 파일명
            
        Returns:
            생성된 비디오 파일 경로
        """
        video_path = self.get_output_path(save_filename)
        
        self.logger.info(f"컨퍼런스 발표 비디오 생성: {save_filename} ({conference_style} 스타일)")
        
        # 컨퍼런스 발표에 적합한 형식의 비디오 생성
        # 학술 vs 산업 스타일에 따른 차별화
        
        return video_path