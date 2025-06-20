"""
Unified Video System

This module consolidates all video-related functionality from:
- src/core/video_*.py
- src/visualization/video/

Provides a unified interface for:
- Video recording during training
- Video generation from data
- Video pipeline processing
- Video management and utilities
"""

# Import core video functionality
from .manager import VideoManager, VideoConfig
from .pipeline import VideoPipeline, PipelineConfig  
from .recorder import VideoRecorder, RecorderConfig
from .utils import VideoUtils

# Import legacy compatibility
from .legacy import LegacyVideoManager, LegacyPipeline

__all__ = [
    'VideoManager',
    'VideoConfig', 
    'VideoPipeline',
    'PipelineConfig',
    'VideoRecorder',
    'RecorderConfig',
    'VideoUtils',
    'LegacyVideoManager',
    'LegacyPipeline',
]