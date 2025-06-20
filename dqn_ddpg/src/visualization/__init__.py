"""
통합 시각화 모듈

DQN vs DDPG 비교 분석을 위한 모든 시각화 기능을 제공합니다.

주요 기능:
- 학습 곡선 시각화
- 알고리즘 성능 비교
- 결정적 정책 분석
- 비디오 생성 및 관리
- 실시간 모니터링
- 프레젠테이션 자료 생성

사용법:
    from src.visualization import quick_comparison, generate_presentation_materials
    
    # 빠른 비교 차트 생성
    quick_comparison(dqn_results, ddpg_results)
    
    # 프레젠테이션 자료 생성
    generate_presentation_materials(dqn_results, ddpg_results)
"""

from .core.config import VisualizationConfig, get_global_config, set_global_config
from .core.utils import (
    setup_matplotlib_korean, 
    validate_experiment_data,
    smooth_data,
    calculate_statistics
)
from .core.base import BaseVisualizer, MultiPlotVisualizer

# 차트 모듈
from .charts import (
    LearningCurveVisualizer,
    ComparisonChartVisualizer,
    MetricsVisualizer,
    PolicyAnalysisVisualizer
)

# 비디오 모듈 (선택적)
try:
    from .video import (
        VideoRenderingPipeline,
        VideoManager,
        DualQualityRecorder,
        ComparisonVideoGenerator,
        LearningVideoGenerator
    )
    VIDEO_MODULES_AVAILABLE = True
except ImportError:
    VIDEO_MODULES_AVAILABLE = False
    # print("Warning: Video modules not available. Install OpenCV for video functionality.")

# 프레젠테이션 기능은 charts, core, video 모듈로 통합됨

# 편의 함수들
def quick_comparison(dqn_results, ddpg_results, output_dir='output/charts'):
    """
    빠른 비교 분석 생성
    
    Args:
        dqn_results: DQN 실험 결과
        ddpg_results: DDPG 실험 결과
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 파일 경로들의 딕셔너리
    """
    config = get_global_config()
    results = {}
    
    # 학습 곡선 생성
    learning_viz = LearningCurveVisualizer(output_dir=output_dir, config=config)
    results['learning_curves'] = learning_viz.create_visualization({
        'dqn': dqn_results, 'ddpg': ddpg_results
    })
    
    # 성능 비교 차트
    comparison_viz = ComparisonChartVisualizer(output_dir=output_dir, config=config)
    results['performance_comparison'] = comparison_viz.create_visualization({
        'dqn': dqn_results, 'ddpg': ddpg_results
    })
    
    return results

def generate_presentation_materials(dqn_results, ddpg_results, output_dir='output'):
    """
    프레젠테이션 자료 생성
    
    Args:
        dqn_results: DQN 실험 결과
        ddpg_results: DDPG 실험 결과
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 자료들의 딕셔너리
    """
    config = get_global_config()
    config.high_quality = True  # 고품질 설정
    
    materials = {}
    
    # 차트 생성
    learning_viz = LearningCurveVisualizer(output_dir=output_dir, config=config)
    comparison_viz = ComparisonChartVisualizer(output_dir=output_dir, config=config)
    metrics_viz = MetricsVisualizer(output_dir=output_dir, config=config)
    
    data = {'dqn': dqn_results, 'ddpg': ddpg_results}
    
    materials['learning_curves'] = learning_viz.create_visualization(data)
    materials['performance_comparison'] = comparison_viz.create_visualization(data)
    materials['training_metrics'] = metrics_viz.create_visualization(data)
    
    # 비디오 생성 (선택적)
    if VIDEO_MODULES_AVAILABLE:
        try:
            video_gen = ComparisonVideoGenerator(output_dir=output_dir, config=config)
            materials['comparison_video'] = video_gen.create_visualization(data)
        except Exception as e:
            print(f"비디오 생성 실패: {e}")
    
    return materials

def create_comparison_video(dqn_results, ddpg_results, output_path='output/videos/comparison.mp4'):
    """
    비교 비디오 생성
    
    Args:
        dqn_results: DQN 실험 결과
        ddpg_results: DDPG 실험 결과
        output_path: 출력 파일 경로
        
    Returns:
        생성된 비디오 파일 경로
    """
    if not VIDEO_MODULES_AVAILABLE:
        raise ImportError("비디오 모듈을 사용할 수 없습니다. OpenCV를 설치하세요.")
    
    config = get_global_config()
    
    video_gen = ComparisonVideoGenerator(config=config)
    return video_gen.create_visualization({
        'dqn': dqn_results, 'ddpg': ddpg_results
    }, save_filename=output_path)

def analyze_deterministic_policies(dqn_agent, ddpg_agent, dqn_env, ddpg_env, 
                                 output_dir='output/charts'):
    """
    결정적 정책 분석
    
    Args:
        dqn_agent: DQN 에이전트
        ddpg_agent: DDPG 에이전트
        dqn_env: DQN 환경
        ddpg_env: DDPG 환경
        output_dir: 출력 디렉토리
        
    Returns:
        분석 결과 파일 경로
    """
    config = get_global_config()
    
    policy_viz = PolicyAnalysisVisualizer(output_dir=output_dir, config=config)
    return policy_viz.create_visualization({
        'dqn_agent': dqn_agent,
        'ddpg_agent': ddpg_agent,
        'dqn_env': dqn_env,
        'ddpg_env': ddpg_env
    })

# 모든 공개 API
__all__ = [
    # 설정
    'VisualizationConfig',
    'get_global_config', 
    'set_global_config',
    
    # 기본 클래스
    'BaseVisualizer',
    'MultiPlotVisualizer',
    
    # 차트 클래스
    'LearningCurveVisualizer',
    'ComparisonChartVisualizer', 
    'MetricsVisualizer',
    'PolicyAnalysisVisualizer',
    
    # 프레젠테이션 (통합됨)
    
    # 유틸리티
    'setup_matplotlib_korean',
    'validate_experiment_data',
    'smooth_data',
    'calculate_statistics',
    
    # 편의 함수
    'quick_comparison',
    'generate_presentation_materials',
    'create_comparison_video',
    'analyze_deterministic_policies'
]

# 비디오 클래스들을 조건부로 추가
if VIDEO_MODULES_AVAILABLE:
    __all__.extend([
        'VideoRenderingPipeline',
        'VideoManager',
        'DualQualityRecorder',
        'ComparisonVideoGenerator',
        'LearningVideoGenerator'
    ])