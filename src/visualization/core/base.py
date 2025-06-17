"""
기본 시각화 클래스

모든 시각화 클래스가 상속받는 기본 클래스를 정의합니다.
공통 기능과 설정을 제공하여 코드 중복을 방지합니다.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging

from .config import VisualizationConfig
from .utils import setup_matplotlib_korean, create_output_directory, get_timestamp


class BaseVisualizer(ABC):
    """
    모든 시각화 클래스의 기본 클래스
    
    공통 기능:
    - 출력 디렉토리 관리
    - 한글 폰트 설정
    - 기본 스타일 적용
    - 파일 저장 관리
    - 로깅 시스템
    
    Attributes:
        config (VisualizationConfig): 시각화 설정
        output_dir (str): 출력 디렉토리 경로
        logger (logging.Logger): 로거 인스턴스
    """
    
    def __init__(self, 
                 output_dir: str = "output/charts",
                 config: Optional[VisualizationConfig] = None,
                 korean_font: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        기본 시각화 클래스 초기화
        
        Args:
            output_dir: 결과 저장 디렉토리
            config: 시각화 설정 객체
            korean_font: 한글 폰트 사용 여부
            logger: 로거 객체 (None이면 기본 로거 생성)
        """
        self.output_dir = output_dir
        self.config = config or VisualizationConfig()
        
        # 출력 디렉토리 생성
        create_output_directory(self.output_dir)
        
        # 한글 폰트 설정
        if korean_font:
            setup_matplotlib_korean()
        
        # 로거 설정
        self.logger = logger or self._setup_logger()
        
        # 기본 스타일 적용
        self._setup_matplotlib_style()
        
        self.logger.info(f"{self.__class__.__name__} 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"visualization.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_matplotlib_style(self):
        """matplotlib 기본 스타일 설정"""
        # 기본 그래프 스타일
        plt.style.use('default')
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 그래프 품질 설정
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        # 색상 팔레트
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        )
        
        # 격자 스타일
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.5
        
        self.logger.debug("matplotlib 스타일 설정 완료")
    
    def get_output_path(self, filename: str, subdir: str = "") -> str:
        """
        출력 파일 경로 생성
        
        Args:
            filename: 파일명
            subdir: 하위 디렉토리 (선택사항)
            
        Returns:
            완전한 파일 경로
        """
        if subdir:
            full_dir = os.path.join(self.output_dir, subdir)
            create_output_directory(full_dir)
        else:
            full_dir = self.output_dir
            
        return os.path.join(full_dir, filename)
    
    def get_timestamped_filename(self, base_name: str, extension: str = ".png") -> str:
        """
        타임스탬프가 포함된 파일명 생성
        
        Args:
            base_name: 기본 파일명
            extension: 파일 확장자
            
        Returns:
            타임스탬프가 포함된 파일명
        """
        timestamp = get_timestamp()
        return f"{base_name}_{timestamp}{extension}"
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   filename: str,
                   content_type: str = "charts",
                   close_after_save: bool = True,
                   **kwargs) -> str:
        """
        그래프 저장 (새로운 구조 적용)
        
        Args:
            fig: matplotlib Figure 객체
            filename: 저장할 파일명
            content_type: 콘텐츠 유형 (charts, plots, diagrams 등)
            close_after_save: 저장 후 figure 닫기 여부
            **kwargs: savefig 추가 매개변수
            
        Returns:
            저장된 파일의 전체 경로
        """
        from .utils import get_output_path_by_extension
        
        # 새로운 구조에 따른 경로 생성
        file_path = get_output_path_by_extension(filename, content_type, self.config)
        
        # 기본 저장 옵션
        save_options = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_options.update(kwargs)
        
        try:
            fig.savefig(file_path, **save_options)
            self.logger.info(f"그래프 저장 완료: {file_path}")
            
            if close_after_save:
                plt.close(fig)
                
            return file_path
            
        except Exception as e:
            self.logger.error(f"그래프 저장 실패: {e}")
            if close_after_save:
                plt.close(fig)
            raise
    
    def create_figure(self, 
                     figsize: Tuple[float, float] = (12, 8),
                     title: str = "",
                     **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, list]]:
        """
        Figure 생성
        
        Args:
            figsize: 그래프 크기
            title: 그래프 제목
            **kwargs: plt.subplots 추가 매개변수
            
        Returns:
            (Figure, Axes) 튜플
        """
        fig, axes = plt.subplots(figsize=figsize, **kwargs)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
        return fig, axes
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        데이터 유효성 검사
        
        Args:
            data: 검사할 데이터
            
        Returns:
            유효성 검사 결과
        """
        if not isinstance(data, dict):
            self.logger.error("데이터가 딕셔너리 형태가 아닙니다")
            return False
            
        if not data:
            self.logger.warning("빈 데이터입니다")
            return False
            
        return True
    
    def log_visualization_info(self, viz_type: str, details: Dict[str, Any]):
        """
        시각화 정보 로깅
        
        Args:
            viz_type: 시각화 타입
            details: 상세 정보
        """
        self.logger.info(f"시각화 생성: {viz_type}")
        for key, value in details.items():
            self.logger.debug(f"  {key}: {value}")
    
    @abstractmethod
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        시각화 생성 (추상 메서드)
        
        하위 클래스에서 반드시 구현해야 합니다.
        
        Args:
            data: 시각화할 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 파일의 경로
        """
        pass
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        # 열린 모든 figure 닫기
        plt.close('all')
        
        if exc_type is not None:
            self.logger.error(f"시각화 중 오류 발생: {exc_type.__name__}: {exc_val}")
        
        self.logger.info(f"{self.__class__.__name__} 종료")


class MultiPlotVisualizer(BaseVisualizer):
    """
    여러 개의 서브플롯을 가진 시각화를 위한 확장 클래스
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_subplot_grid(self, 
                           rows: int, 
                           cols: int,
                           figsize: Tuple[float, float] = None,
                           title: str = "",
                           **kwargs) -> Tuple[plt.Figure, list]:
        """
        서브플롯 격자 생성
        
        Args:
            rows: 행 수
            cols: 열 수  
            figsize: 그래프 크기
            title: 전체 제목
            **kwargs: 추가 매개변수
            
        Returns:
            (Figure, Axes 리스트) 튜플
        """
        if figsize is None:
            figsize = (6 * cols, 4 * rows)
            
        fig, axes = self.create_figure(
            figsize=figsize,
            title=title,
            nrows=rows,
            ncols=cols,
            **kwargs
        )
        
        # axes를 항상 리스트로 만들기
        if rows * cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        return fig, axes
    
    def setup_subplot(self, 
                     ax: plt.Axes,
                     title: str = "",
                     xlabel: str = "",
                     ylabel: str = "",
                     grid: bool = True):
        """
        개별 서브플롯 설정
        
        Args:
            ax: Axes 객체
            title: 서브플롯 제목
            xlabel: X축 라벨
            ylabel: Y축 라벨  
            grid: 격자 표시 여부
        """
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=0.3)
            
        # 축 스타일 설정
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)