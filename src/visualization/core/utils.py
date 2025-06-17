"""
시각화 공통 유틸리티 함수들

모든 시각화 모듈에서 사용하는 공통 기능들을 제공합니다.
파일 관리, 데이터 검증, 한글 폰트 설정 등의 기능을 포함합니다.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import warnings


def setup_matplotlib_korean():
    """
    matplotlib 한글 폰트 설정
    
    시스템에서 사용 가능한 한글 폰트를 찾아 설정합니다.
    한글이 깨지지 않도록 폰트와 마이너스 기호 설정을 조정합니다.
    """
    # 한글 폰트 목록 (우선순위 순)
    korean_fonts = [
        'Malgun Gothic',      # Windows
        'AppleGothic',        # macOS  
        'Noto Sans CJK KR',   # Linux
        'NanumGothic',        # 나눔고딕
        'DejaVu Sans',        # 기본 폰트
        'Arial'               # 대체 폰트
    ]
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font is None:
        selected_font = 'DejaVu Sans'
        warnings.warn("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    
    # matplotlib 설정
    plt.rcParams['font.family'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    # 폰트 캐시 갱신 (버전별 호환성 처리)
    try:
        if hasattr(fm.fontManager, '_rebuild'):
            fm.fontManager._rebuild()
        elif hasattr(fm, '_rebuild'):
            fm._rebuild()
    except:
        pass  # 폰트 캐시 갱신 실패시 무시
    
    print(f"한글 폰트 설정 완료: {selected_font}")


def create_output_directory(path: str, parents: bool = True) -> str:
    """
    출력 디렉토리 생성
    
    Args:
        path: 생성할 디렉토리 경로
        parents: 상위 디렉토리도 함께 생성할지 여부
        
    Returns:
        생성된 디렉토리의 절대 경로
    """
    try:
        Path(path).mkdir(parents=parents, exist_ok=True)
        return os.path.abspath(path)
    except Exception as e:
        raise OSError(f"디렉토리 생성 실패: {path} - {e}")


def ensure_path_exists(file_path: str) -> str:
    """
    파일 경로의 디렉토리가 존재하는지 확인하고 생성
    
    Args:
        file_path: 확인할 파일 경로
        
    Returns:
        파일 경로
    """
    directory = os.path.dirname(file_path)
    if directory:
        create_output_directory(directory)
    return file_path


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    현재 시간 기반 타임스탬프 생성
    
    Args:
        format_str: 시간 형식 문자열
        
    Returns:
        타임스탬프 문자열
    """
    return datetime.now().strftime(format_str)


def save_plot_safely(fig: plt.Figure, 
                     file_path: str,
                     dpi: int = 300,
                     close_after_save: bool = True,
                     **kwargs) -> str:
    """
    안전한 그래프 저장
    
    Args:
        fig: matplotlib Figure 객체
        file_path: 저장할 파일 경로
        dpi: 해상도
        close_after_save: 저장 후 figure 닫기 여부
        **kwargs: savefig 추가 매개변수
        
    Returns:
        저장된 파일 경로
    """
    # 디렉토리 생성
    ensure_path_exists(file_path)
    
    # 기본 저장 옵션
    save_options = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
        'transparent': False
    }
    save_options.update(kwargs)
    
    try:
        fig.savefig(file_path, **save_options)
        
        if close_after_save:
            plt.close(fig)
            
        return file_path
        
    except Exception as e:
        if close_after_save:
            plt.close(fig)
        raise IOError(f"그래프 저장 실패: {file_path} - {e}")


def validate_data(data: Any, 
                 required_keys: Optional[List[str]] = None,
                 data_type: type = dict) -> bool:
    """
    데이터 유효성 검사
    
    Args:
        data: 검사할 데이터
        required_keys: 필수 키 목록 (딕셔너리인 경우)
        data_type: 예상 데이터 타입
        
    Returns:
        유효성 검사 결과
    """
    # 타입 검사
    if not isinstance(data, data_type):
        logging.error(f"데이터 타입이 맞지 않습니다. 예상: {data_type}, 실제: {type(data)}")
        return False
    
    # 빈 데이터 검사
    if not data:
        logging.warning("빈 데이터입니다")
        return False
    
    # 딕셔너리인 경우 필수 키 검사
    if isinstance(data, dict) and required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logging.error(f"필수 키가 누락되었습니다: {missing_keys}")
            return False
    
    return True


def validate_experiment_data(data: Dict[str, Any]) -> bool:
    """
    실험 데이터 유효성 검사
    
    Args:
        data: 실험 데이터 딕셔너리
        
    Returns:
        유효성 검사 결과
    """
    required_keys = ['episode_rewards', 'episode_lengths']
    
    if not validate_data(data, required_keys):
        return False
    
    # 리스트 데이터 검사
    for key in required_keys:
        if not isinstance(data[key], (list, np.ndarray)):
            logging.error(f"{key}가 리스트나 배열이 아닙니다")
            return False
        
        if len(data[key]) == 0:
            logging.warning(f"{key}가 비어있습니다")
            return False
    
    return True


def smooth_data(data: Union[List, np.ndarray], 
               window_size: int = 50,
               method: str = 'moving_average') -> np.ndarray:
    """
    데이터 스무딩
    
    Args:
        data: 스무딩할 데이터
        window_size: 윈도우 크기
        method: 스무딩 방법 ('moving_average', 'exponential')
        
    Returns:
        스무딩된 데이터
    """
    data = np.array(data)
    
    if len(data) < window_size:
        return data
    
    if method == 'moving_average':
        # 이동 평균
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    elif method == 'exponential':
        # 지수 이동 평균
        alpha = 2.0 / (window_size + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        
        return result
    
    else:
        raise ValueError(f"지원되지 않는 스무딩 방법: {method}")


def calculate_statistics(data: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    기본 통계 계산
    
    Args:
        data: 통계를 계산할 데이터
        
    Returns:
        통계 정보 딕셔너리
    """
    data = np.array(data)
    
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'count': len(data)
    }


def prepare_comparison_data(dqn_data: Dict[str, Any], 
                          ddpg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    비교 분석을 위한 데이터 준비
    
    Args:
        dqn_data: DQN 실험 데이터
        ddpg_data: DDPG 실험 데이터
        
    Returns:
        비교 분석용 데이터
    """
    if not (validate_experiment_data(dqn_data) and validate_experiment_data(ddpg_data)):
        raise ValueError("유효하지 않은 실험 데이터입니다")
    
    # 기본 통계 계산
    dqn_stats = calculate_statistics(dqn_data['episode_rewards'])
    ddpg_stats = calculate_statistics(ddpg_data['episode_rewards'])
    
    # 학습 진행률 계산
    dqn_smoothed = smooth_data(dqn_data['episode_rewards'])
    ddpg_smoothed = smooth_data(ddpg_data['episode_rewards'])
    
    return {
        'dqn': {
            'raw_data': dqn_data,
            'statistics': dqn_stats,
            'smoothed_rewards': dqn_smoothed
        },
        'ddpg': {
            'raw_data': ddpg_data,
            'statistics': ddpg_stats,
            'smoothed_rewards': ddpg_smoothed
        },
        'comparison': {
            'reward_improvement': {
                'dqn': dqn_stats['mean'] - dqn_stats['min'],
                'ddpg': ddpg_stats['mean'] - ddpg_stats['min']
            },
            'stability': {
                'dqn': dqn_stats['std'] / abs(dqn_stats['mean']) if dqn_stats['mean'] != 0 else float('inf'),
                'ddpg': ddpg_stats['std'] / abs(ddpg_stats['mean']) if ddpg_stats['mean'] != 0 else float('inf')
            }
        }
    }


def load_experiment_results(file_path: str) -> Dict[str, Any]:
    """
    실험 결과 파일 로드
    
    Args:
        file_path: 결과 파일 경로 (.json)
        
    Returns:
        실험 결과 데이터
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not validate_experiment_data(data):
            raise ValueError("유효하지 않은 실험 결과 형식입니다")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 오류: {e}")


def save_experiment_results(data: Dict[str, Any], file_path: str):
    """
    실험 결과 저장
    
    Args:
        data: 저장할 실험 데이터
        file_path: 저장할 파일 경로
    """
    if not validate_experiment_data(data):
        raise ValueError("유효하지 않은 실험 데이터입니다")
    
    ensure_path_exists(file_path)
    
    # numpy 배열을 리스트로 변환
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_data[key] = value.item()
        else:
            serializable_data[key] = value
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"파일 저장 실패: {file_path} - {e}")


def create_color_palette(n_colors: int, 
                        palette_name: str = 'tab10') -> List[str]:
    """
    색상 팔레트 생성
    
    Args:
        n_colors: 필요한 색상 개수
        palette_name: matplotlib 팔레트 이름
        
    Returns:
        색상 코드 리스트
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if palette_name in plt.colormaps():
        cmap = cm.get_cmap(palette_name)
        colors = [mcolors.to_hex(cmap(i / max(1, n_colors - 1))) for i in range(n_colors)]
    else:
        # 기본 색상 팔레트
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf'
        ]
        colors = (default_colors * ((n_colors // len(default_colors)) + 1))[:n_colors]
    
    return colors


def format_number(value: float, precision: int = 2) -> str:
    """
    숫자 포맷팅
    
    Args:
        value: 포맷팅할 숫자
        precision: 소수점 자릿수
        
    Returns:
        포맷팅된 문자열
    """
    if abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def create_summary_table(dqn_stats: Dict[str, float], 
                        ddpg_stats: Dict[str, float]) -> str:
    """
    요약 테이블 생성
    
    Args:
        dqn_stats: DQN 통계
        ddpg_stats: DDPG 통계
        
    Returns:
        마크다운 형식의 테이블 문자열
    """
    metrics = ['mean', 'std', 'min', 'max', 'median']
    
    table = "| Metric | DQN | DDPG | Difference |\n"
    table += "|--------|-----|------|------------|\n"
    
    for metric in metrics:
        dqn_val = dqn_stats.get(metric, 0)
        ddpg_val = ddpg_stats.get(metric, 0)
        diff = dqn_val - ddpg_val
        
        table += f"| {metric.capitalize()} | {format_number(dqn_val)} | {format_number(ddpg_val)} | {format_number(diff)} |\n"
    
    return table


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
        
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def memory_efficient_plot(data: np.ndarray, 
                         max_points: int = 10000) -> np.ndarray:
    """
    메모리 효율적인 플롯 데이터 생성
    
    큰 데이터셋을 시각화할 때 성능을 위해 데이터 포인트를 줄입니다.
    
    Args:
        data: 원본 데이터
        max_points: 최대 포인트 수
        
    Returns:
        줄어든 데이터
    """
    if len(data) <= max_points:
        return data
    
    # 균등한 간격으로 샘플링
    indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
    return data[indices]


def get_output_path_by_extension(filename: str, 
                                content_type: str = "charts", 
                                config=None) -> str:
    """
    파일 확장자에 따른 출력 경로 생성
    
    Args:
        filename: 저장할 파일명
        content_type: 콘텐츠 유형 (charts, plots, diagrams, etc.)
        config: 시각화 설정 객체
        
    Returns:
        확장자별 출력 경로
    """
    from .config import VisualizationConfig
    
    if config is None:
        config = VisualizationConfig()
    
    # 파일 확장자 추출
    ext = filename.split('.')[-1].lower()
    
    # 기본 경로 가져오기 (extension_mapping 사용)
    if ext in config.extension_mapping:
        content_category = config.extension_mapping[ext]
        base_path = config.content_paths.get(content_category, config.output_dir) + "/"
    else:
        # 알 수 없는 확장자는 기본 output 디렉토리로
        base_path = config.output_dir + "/"
    
    # 콘텐츠 유형별 세부 경로 추가
    if ext in ['png', 'jpg', 'svg', 'pdf']:
        # 이미지 파일
        content_path = base_path + content_type + "/"
    elif ext in ['mp4', 'avi', 'gif']:
        # 비디오 파일
        if content_type in config.video_subdirs:
            content_path = base_path + config.video_subdirs[content_type]
        else:
            content_path = base_path + content_type + "/"
    elif ext in ['json', 'csv', 'yaml', 'pkl']:
        # 데이터 파일
        content_path = base_path + content_type + "/"
    elif ext in ['md', 'html', 'txt']:
        # 문서 파일
        content_path = base_path + content_type + "/"
    else:
        content_path = base_path
    
    # 디렉토리 생성
    ensure_path_exists(content_path + filename)
    
    return content_path + filename


def create_structured_filename(prefix: str,
                              content_type: str,
                              algorithm: str = "",
                              environment: str = "",
                              extension: str = "png",
                              timestamp: bool = True) -> str:
    """
    구조화된 파일명 생성
    
    Args:
        prefix: 파일 접두사 (예: "learning_curves", "comparison")
        content_type: 콘텐츠 유형 (예: "chart", "video", "data")
        algorithm: 알고리즘 이름 (예: "dqn", "ddpg", "dqn_vs_ddpg")
        environment: 환경 이름 (예: "cartpole", "pendulum")
        extension: 파일 확장자
        timestamp: 타임스탬프 포함 여부
        
    Returns:
        구조화된 파일명
    """
    # 기본 구조: prefix_content-type_algorithm_environment_timestamp.ext
    parts = [prefix, content_type]
    
    if algorithm:
        parts.append(algorithm)
    
    if environment:
        parts.append(environment)
    
    if timestamp:
        parts.append(get_timestamp())
    
    filename = "_".join(parts) + "." + extension
    
    return filename


def migrate_file_to_new_structure(old_path: str, 
                                 content_type: str = "charts",
                                 config=None) -> Optional[str]:
    """
    기존 파일을 새로운 구조로 이동
    
    Args:
        old_path: 기존 파일 경로
        content_type: 콘텐츠 유형
        config: 시각화 설정
        
    Returns:
        새로운 파일 경로 (이동 성공시)
    """
    import shutil
    
    if not os.path.exists(old_path):
        print(f"파일이 존재하지 않습니다: {old_path}")
        return None
    
    # 파일명 추출
    filename = os.path.basename(old_path)
    
    # 새로운 경로 생성
    new_path = get_output_path_by_extension(filename, content_type, config)
    
    try:
        # 파일 복사 (원본 유지)
        shutil.copy2(old_path, new_path)
        print(f"파일 마이그레이션 완료: {old_path} -> {new_path}")
        return new_path
    except Exception as e:
        print(f"파일 마이그레이션 실패: {old_path} - {e}")
        return None