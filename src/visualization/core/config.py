"""
시각화 설정 관리 모듈

모든 시각화 관련 설정을 중앙에서 관리합니다.
YAML 파일 지원 및 동적 설정 변경 기능을 제공합니다.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


@dataclass
class ChartConfig:
    """
    차트 시각화 설정
    """
    # 기본 설정
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 300
    style: str = 'default'
    
    # 색상 설정
    color_palette: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ])
    dqn_color: str = '#1f77b4'  # 파란색
    ddpg_color: str = '#ff7f0e'  # 주황색
    
    # 폰트 설정
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    
    # 격자 설정
    grid: bool = True
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5
    
    # 범례 설정
    legend_location: str = 'best'
    legend_frameon: bool = True
    legend_fancybox: bool = True
    legend_shadow: bool = False
    
    # 축 설정
    spine_linewidth: float = 0.5
    hide_top_right_spines: bool = True


@dataclass 
class VideoConfig:
    """
    비디오 생성 설정
    """
    # 기본 비디오 설정
    fps: int = 30
    duration_seconds: int = 180
    resolution: Tuple[int, int] = (1280, 720)
    codec: str = 'mp4v'
    quality: str = 'high'  # 'low', 'medium', 'high'
    
    # 비디오 스타일
    background_color: str = 'white'
    text_color: str = 'black'
    progress_bar_color: str = '#1f77b4'
    
    # 애니메이션 설정
    transition_duration: float = 1.0
    fade_duration: float = 0.5
    
    # 텍스트 설정
    title_fontsize: int = 20
    subtitle_fontsize: int = 14
    info_fontsize: int = 12
    
    # 레이아웃
    margin: float = 0.1
    spacing: float = 0.05
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """품질별 설정 반환"""
        quality_map = {
            'low': {'dpi': 100, 'bitrate': '1000k'},
            'medium': {'dpi': 200, 'bitrate': '2000k'},
            'high': {'dpi': 300, 'bitrate': '4000k'}
        }
        return quality_map.get(self.quality, quality_map['medium'])


@dataclass
class RealtimeConfig:
    """
    실시간 시각화 설정
    """
    # 업데이트 설정
    update_interval: float = 1.0  # 초
    max_points: int = 1000
    smooth_updates: bool = True
    
    # 디스플레이 설정
    window_title: str = "DQN vs DDPG 실시간 모니터링"
    figsize: Tuple[float, float] = (15, 10)
    
    # 그래프 설정
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    
    # 색상 설정
    background_color: str = 'white'
    grid_color: str = 'gray'
    
    # 성능 설정
    use_blitting: bool = True
    optimize_drawing: bool = True


@dataclass
class PresentationConfig:
    """
    프레젠테이션 자료 생성 설정
    """
    # 템플릿 설정
    template_style: str = 'professional'  # 'simple', 'professional', 'academic'
    language: str = 'korean'  # 'korean', 'english'
    
    # 레이아웃 설정
    slide_size: Tuple[float, float] = (16, 9)  # 16:9 비율
    margin: float = 0.1
    
    # 색상 테마
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    accent_color: str = '#2ca02c'
    text_color: str = '#333333'
    
    # 폰트 설정
    title_font: str = 'Arial'
    body_font: str = 'Arial'
    code_font: str = 'Courier New'
    
    # 내용 설정
    include_code: bool = False
    include_technical_details: bool = True
    include_references: bool = True


@dataclass
class VisualizationConfig:
    """
    통합 시각화 설정
    """
    # 기본 디렉토리 설정 (새로운 콘텐츠 기반 구조)
    output_dir: str = "output"
    charts_dir: str = "output/charts"
    tables_dir: str = "output/tables"
    videos_dir: str = "output/videos"
    logs_dir: str = "output/logs"
    reports_dir: str = "output/reports"
    temp_dir: str = "temp"
    
    # 콘텐츠 타입별 출력 경로 매핑
    content_paths: Dict[str, str] = field(default_factory=lambda: {
        'charts': 'output/charts/',
        'tables': 'output/tables/',
        'videos': 'output/videos/',
        'logs': 'output/logs/',
        'reports': 'output/reports/'
    })
    
    # 확장자별 출력 매핑 (새 구조)
    extension_mapping: Dict[str, str] = field(default_factory=lambda: {
        'png': 'charts',  # 차트 이미지
        'jpg': 'charts',  # 차트 이미지
        'svg': 'charts',  # 벡터 차트
        'pdf': 'reports', # PDF 리포트
        'mp4': 'videos',  # 비디오
        'avi': 'videos',  # 비디오
        'gif': 'videos',  # 애니메이션
        'json': 'logs',   # 로그 데이터
        'csv': 'logs',    # 데이터
        'yaml': 'logs',   # 설정 백업
        'pkl': 'logs',    # 모델 데이터
        'md': 'reports',  # 마크다운 리포트
        'html': 'reports', # HTML 리포트
        'txt': 'reports'  # 텍스트 리포트
    })
    
    # 콘텐츠별 세부 디렉토리
    chart_subdirs: Dict[str, str] = field(default_factory=lambda: {
        'learning_curves': 'learning_curves/',
        'comparison': 'performance_comparison/',
        'metrics': 'metrics/',
        'policy_analysis': 'policy_analysis/'
    })
    
    video_subdirs: Dict[str, str] = field(default_factory=lambda: {
        'training': 'training/',
        'comparison': 'comparison/',
        'pipeline': 'pipeline/'
    })
    
    log_subdirs: Dict[str, str] = field(default_factory=lambda: {
        'experiments': 'experiments/',
        'configs': 'configs/'
    })
    
    # 파일 설정
    image_format: str = 'png'
    video_format: str = 'mp4'
    save_intermediate: bool = False
    
    # 한글 지원
    korean_font: bool = True
    font_family: str = 'DejaVu Sans'
    
    # 품질 설정
    high_quality: bool = True
    compression: bool = False
    
    # 하위 설정들
    chart: ChartConfig = field(default_factory=ChartConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    presentation: PresentationConfig = field(default_factory=PresentationConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VisualizationConfig':
        """
        YAML 파일에서 설정 로드
        
        Args:
            config_path: YAML 설정 파일 경로
            
        Returns:
            VisualizationConfig 인스턴스
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        return cls(**config_data)
    
    def to_yaml(self, config_path: str):
        """
        설정을 YAML 파일로 저장
        
        Args:
            config_path: 저장할 YAML 파일 경로
        """
        # dataclass를 딕셔너리로 변환
        config_dict = {
            'output_dir': self.output_dir,
            'charts_dir': self.charts_dir,
            'tables_dir': self.tables_dir,
            'videos_dir': self.videos_dir,
            'logs_dir': self.logs_dir,
            'reports_dir': self.reports_dir,
            'temp_dir': self.temp_dir,
            'image_format': self.image_format,
            'video_format': self.video_format,
            'save_intermediate': self.save_intermediate,
            'korean_font': self.korean_font,
            'font_family': self.font_family,
            'high_quality': self.high_quality,
            'compression': self.compression,
            'chart': self._dataclass_to_dict(self.chart),
            'video': self._dataclass_to_dict(self.video),
            'realtime': self._dataclass_to_dict(self.realtime),
            'presentation': self._dataclass_to_dict(self.presentation)
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """dataclass를 딕셔너리로 변환"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return obj
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        딕셔너리에서 설정 업데이트
        
        Args:
            config_dict: 업데이트할 설정 딕셔너리
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key in ['chart', 'video', 'realtime', 'presentation']:
                    # 하위 설정 객체 업데이트
                    sub_config = getattr(self, key)
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_config, sub_key):
                                setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def get_output_path(self, filename: str, content_type: str = "", subdir: str = "") -> str:
        """
        출력 파일 경로 생성 (새 구조)
        
        Args:
            filename: 파일명
            content_type: 콘텐츠 타입 ('charts', 'videos', 'logs', 'tables', 'reports')
            subdir: 하위 디렉토리
            
        Returns:
            완전한 파일 경로
        """
        # 파일 확장자로부터 콘텐츠 타입 추론
        if not content_type:
            file_ext = filename.split('.')[-1].lower()
            content_type = self.extension_mapping.get(file_ext, 'charts')
        
        # 기본 디렉토리 설정
        base_dir = self.content_paths.get(content_type, self.output_dir)
        
        # 하위 디렉토리 추가
        if subdir:
            base_dir = os.path.join(base_dir, subdir)
            
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, filename)
    
    def validate(self) -> bool:
        """
        설정 유효성 검사
        
        Returns:
            유효성 검사 결과
        """
        # 필수 디렉토리 확인
        required_dirs = [self.output_dir, self.video_dir, self.presentation_dir]
        for dir_path in required_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"디렉토리 생성 실패: {dir_path} - {e}")
                return False
        
        # 이미지 형식 확인
        if self.image_format not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            print(f"지원되지 않는 이미지 형식: {self.image_format}")
            return False
        
        # 비디오 형식 확인
        if self.video_format not in ['mp4', 'avi', 'mkv', 'mov']:
            print(f"지원되지 않는 비디오 형식: {self.video_format}")
            return False
        
        return True


class ConfigManager:
    """
    설정 관리자 클래스
    
    여러 설정 파일을 관리하고 동적으로 설정을 변경할 수 있습니다.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        설정 관리자 초기화
        
        Args:
            config_dir: 설정 파일 디렉토리
        """
        self.config_dir = config_dir
        self.current_config = VisualizationConfig()
        self._config_history = []
        
        # 설정 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
    
    def load_config(self, config_name: str) -> VisualizationConfig:
        """
        설정 파일 로드
        
        Args:
            config_name: 설정 파일명 (확장자 제외)
            
        Returns:
            VisualizationConfig 인스턴스
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if os.path.exists(config_path):
            self.current_config = VisualizationConfig.from_yaml(config_path)
        else:
            print(f"설정 파일이 없어 기본 설정을 사용합니다: {config_path}")
            self.current_config = VisualizationConfig()
            
        return self.current_config
    
    def save_config(self, config_name: str, config: Optional[VisualizationConfig] = None):
        """
        설정 파일 저장
        
        Args:
            config_name: 설정 파일명 (확장자 제외)
            config: 저장할 설정 (None이면 현재 설정 사용)
        """
        config = config or self.current_config
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        config.to_yaml(config_path)
        print(f"설정 저장 완료: {config_path}")
    
    def update_config(self, **kwargs):
        """
        현재 설정 업데이트
        
        Args:
            **kwargs: 업데이트할 설정 키-값 쌍
        """
        # 현재 설정을 히스토리에 저장
        self._config_history.append(self.current_config)
        
        # 설정 업데이트
        self.current_config.update_from_dict(kwargs)
    
    def restore_config(self):
        """이전 설정으로 복원"""
        if self._config_history:
            self.current_config = self._config_history.pop()
            print("이전 설정으로 복원되었습니다")
        else:
            print("복원할 이전 설정이 없습니다")
    
    def create_preset_configs(self):
        """미리 정의된 설정 프리셋 생성"""
        # 빠른 테스트용 설정
        quick_config = VisualizationConfig()
        quick_config.chart.figsize = (8, 6)
        quick_config.video.duration_seconds = 60
        quick_config.video.quality = 'medium'
        self.save_config("quick_test", quick_config)
        
        # 고품질 프레젠테이션용 설정
        presentation_config = VisualizationConfig()
        presentation_config.high_quality = True
        presentation_config.chart.dpi = 300
        presentation_config.video.quality = 'high'
        presentation_config.video.resolution = (1920, 1080)
        self.save_config("presentation", presentation_config)
        
        # 개발/디버깅용 설정
        debug_config = VisualizationConfig()
        debug_config.save_intermediate = True
        debug_config.video.duration_seconds = 30
        debug_config.chart.figsize = (10, 6)
        self.save_config("debug", debug_config)
        
        print("프리셋 설정 파일들이 생성되었습니다")


# 전역 설정 인스턴스
_global_config = VisualizationConfig()
_config_manager = ConfigManager()

def get_global_config() -> VisualizationConfig:
    """전역 설정 반환"""
    return _global_config

def set_global_config(config: VisualizationConfig):
    """전역 설정 업데이트"""
    global _global_config
    _global_config = config

def get_config_manager() -> ConfigManager:
    """설정 관리자 반환"""
    return _config_manager