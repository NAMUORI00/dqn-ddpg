"""
Legacy Video System Compatibility Layer

This module provides backward compatibility for existing code that relies on
the old video system structure. It wraps the new unified video system to
maintain compatibility while encouraging migration to the new API.
"""

import warnings
from typing import Any, Dict, Optional

from .manager import VideoManager as NewVideoManager, VideoConfig as NewVideoConfig
from .pipeline import VideoPipeline as NewVideoPipeline


class LegacyVideoManager:
    """Legacy wrapper for the old video manager"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LegacyVideoManager is deprecated. Use src.core.video.VideoManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._manager = NewVideoManager(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._manager, name)


class LegacyPipeline:
    """Legacy wrapper for the old video pipeline"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LegacyPipeline is deprecated. Use src.core.video.VideoPipeline instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._pipeline = NewVideoPipeline(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._pipeline, name)


# Provide backward compatibility imports
def VideoConfig(*args, **kwargs):
    """Legacy VideoConfig constructor"""
    warnings.warn(
        "Direct VideoConfig import is deprecated. Use src.core.video.VideoConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return NewVideoConfig(*args, **kwargs)


def VideoManager(*args, **kwargs):
    """Legacy VideoManager constructor"""
    warnings.warn(
        "Direct VideoManager import is deprecated. Use src.core.video.VideoManager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return NewVideoManager(*args, **kwargs)