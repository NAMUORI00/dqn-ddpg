"""
Video System Package
Unified video processing, recording, and management system
"""

# Import main components for easy access
from .core.video_manager import VideoManager
from .core.video_utils import VideoEncoder, VideoConfig
from .processing.video_pipeline import VideoRenderingPipeline

# Legacy imports for backward compatibility
from .core.video_manager import VideoManager as LegacyVideoManager

__all__ = [
    'VideoManager',
    'VideoEncoder', 
    'VideoConfig',
    'VideoRenderingPipeline',
    'LegacyVideoManager'  # For backward compatibility
]