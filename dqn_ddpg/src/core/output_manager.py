"""
Unified Output Management System

This module provides a centralized system for managing all project outputs:
- Videos
- Charts
- Logs  
- Models
- Results

Features:
- Consistent directory structure
- Automatic cleanup
- File organization by type and date
- Configuration-driven paths
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yaml


class OutputManager:
    """Centralized output management system"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize output manager with configuration"""
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config.get("output", {}).get("base_dir", "output"))
        self._ensure_structure()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "output": {
                "base_dir": "output",
                "structure": {
                    "videos": "videos",
                    "charts": "charts",
                    "logs": "logs", 
                    "models": "models",
                    "results": "results"
                }
            }
        }
    
    def _ensure_structure(self):
        """Ensure output directory structure exists"""
        structure = self.config.get("output", {}).get("structure", {})
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for key, subdir in structure.items():
            (self.base_dir / subdir).mkdir(exist_ok=True)
            # Create common subdirectories for videos
            if key == "videos":
                for video_type in ["training", "comparison", "pipeline", "temp"]:
                    (self.base_dir / subdir / video_type).mkdir(exist_ok=True)
    
    def get_output_path(self, output_type: str, filename: str = "", 
                       subdir: str = "") -> Path:
        """Get standardized output path for given type"""
        structure = self.config.get("output", {}).get("structure", {})
        type_dir = structure.get(output_type, output_type)
        
        path = self.base_dir / type_dir
        if subdir:
            path = path / subdir
        if filename:
            path = path / filename
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_video_path(self, video_type: str = "training", 
                      filename: str = "") -> Path:
        """Get video output path"""
        return self.get_output_path("videos", filename, video_type)
    
    def get_chart_path(self, chart_type: str = "learning_curves", 
                      filename: str = "") -> Path:
        """Get chart output path"""
        return self.get_output_path("charts", filename, chart_type)
    
    def get_log_path(self, log_type: str = "experiments", 
                    filename: str = "") -> Path:
        """Get log output path"""
        return self.get_output_path("logs", filename, log_type)
    
    def get_model_path(self, algorithm: str = "", filename: str = "") -> Path:
        """Get model output path"""
        return self.get_output_path("models", filename, algorithm)
    
    def get_results_path(self, experiment: str = "", filename: str = "") -> Path:
        """Get results output path"""
        return self.get_output_path("results", filename, experiment)
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified age"""
        temp_path = self.get_video_path("temp")
        if not temp_path.exists():
            return
            
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up temporary file: {file_path}")
                    except OSError as e:
                        print(f"Failed to remove {file_path}: {e}")
    
    def migrate_legacy_outputs(self):
        """Migrate outputs from legacy directory structure"""
        legacy_paths = [
            ("videos", Path("videos")),
            ("charts", Path("output/charts")),
        ]
        
        for output_type, legacy_path in legacy_paths:
            if legacy_path.exists():
                new_path = self.get_output_path(output_type)
                print(f"Migrating {legacy_path} -> {new_path}")
                
                # Copy files if they don't exist in new location
                for file_path in legacy_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(legacy_path)
                        new_file_path = new_path / relative_path
                        
                        if not new_file_path.exists():
                            new_file_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, new_file_path)
                            print(f"  Copied: {relative_path}")
    
    def get_structure_info(self) -> Dict:
        """Get information about current output structure"""
        info = {"base_directory": str(self.base_dir)}
        structure = self.config.get("output", {}).get("structure", {})
        
        for key, subdir in structure.items():
            path = self.base_dir / subdir
            info[key] = {
                "path": str(path),
                "exists": path.exists(),
                "file_count": len(list(path.rglob("*"))) if path.exists() else 0
            }
        
        return info


# Global output manager instance
output_manager = OutputManager()


# Convenience functions
def get_video_path(video_type: str = "training", filename: str = "") -> Path:
    """Get video output path"""
    return output_manager.get_video_path(video_type, filename)


def get_chart_path(chart_type: str = "learning_curves", filename: str = "") -> Path:
    """Get chart output path"""
    return output_manager.get_chart_path(chart_type, filename)


def get_log_path(log_type: str = "experiments", filename: str = "") -> Path:
    """Get log output path"""
    return output_manager.get_log_path(log_type, filename)


def get_model_path(algorithm: str = "", filename: str = "") -> Path:
    """Get model output path"""
    return output_manager.get_model_path(algorithm, filename)


def get_results_path(experiment: str = "", filename: str = "") -> Path:
    """Get results output path"""
    return output_manager.get_results_path(experiment, filename)