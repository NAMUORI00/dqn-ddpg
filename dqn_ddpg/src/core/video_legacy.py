"""
Legacy Video Imports
Provides backward compatibility for existing code that imports video components from src.core
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing video components from src.core is deprecated. "
    "Please use 'from src.video import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import video components from new location
try:
    from ..video.core.video_manager import VideoManager
    from ..video.core.video_utils import VideoEncoder, VideoConfig  
    from ..video.processing.video_pipeline import VideoRenderingPipeline
    
    # Legacy aliases
    video_manager = VideoManager
    video_utils = VideoEncoder
    
except ImportError as e:
    # Fallback for cases where video system is not available
    warnings.warn(f"Video system not available: {e}", UserWarning)
    VideoManager = None
    VideoEncoder = None
    VideoConfig = None
    VideoRenderingPipeline = None