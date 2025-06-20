"""
Unified Configuration Management System
Eliminates code duplication and provides centralized configuration handling
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class GPUConfig:
    """GPU configuration settings"""
    enabled: Union[bool, str] = "auto"  # auto, true, false
    device_id: Optional[int] = None
    mixed_precision: bool = True
    memory_fraction: float = 0.9
    cuda_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cuda_options is None:
            self.cuda_options = {
                'deterministic': False,
                'benchmark': True,
                'allow_tf32': True
            }


@dataclass 
class VideoConfig:
    """Video configuration settings"""
    fps: int = 30
    resolution: List[int] = None
    codec: str = "mp4v"
    quality: str = "medium"  # low, medium, high, ultra
    presets: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.resolution is None:
            self.resolution = [1280, 720]
        if self.presets is None:
            self.presets = {
                "preview": {"fps": 15, "resolution": [640, 480]},
                "medium": {"fps": 30, "resolution": [1280, 720]},
                "high": {"fps": 60, "resolution": [1920, 1080]},
                "ultra": {"fps": 60, "resolution": [2560, 1440]}
            }


@dataclass
class OutputConfig:
    """Output directory configuration"""
    base_dir: str = "output"
    models_dir: str = "models"
    videos_dir: str = "videos"
    logs_dir: str = "logs"
    results_dir: str = "results"
    
    def get_path(self, category: str, *sub_paths: str) -> Path:
        """Get full path for a category"""
        category_map = {
            'models': self.models_dir,
            'videos': self.videos_dir, 
            'logs': self.logs_dir,
            'results': self.results_dir
        }
        
        if category not in category_map:
            raise ValueError(f"Unknown output category: {category}")
        
        base_path = Path(self.base_dir) / category_map[category]
        for sub_path in sub_paths:
            base_path = base_path / sub_path
        
        return base_path


class ConfigValidator:
    """Configuration validation and type checking"""
    
    @staticmethod
    def validate_gpu_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GPU configuration"""
        validated = {}
        
        # Validate enabled field
        enabled = config.get('enabled', 'auto')
        if isinstance(enabled, str):
            enabled = enabled.lower()
            if enabled not in ['auto', 'true', 'false']:
                raise ValueError(f"Invalid GPU enabled value: {enabled}")
        elif isinstance(enabled, bool):
            enabled = str(enabled).lower()
        validated['enabled'] = enabled
        
        # Validate device_id
        device_id = config.get('device_id')
        if device_id is not None:
            if not isinstance(device_id, int) or device_id < 0:
                raise ValueError(f"Invalid GPU device_id: {device_id}")
        validated['device_id'] = device_id
        
        # Validate mixed_precision
        validated['mixed_precision'] = bool(config.get('mixed_precision', True))
        
        # Validate memory_fraction
        memory_fraction = config.get('memory_fraction', 0.9)
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError(f"Invalid memory_fraction: {memory_fraction}")
        validated['memory_fraction'] = memory_fraction
        
        # Validate cuda_options
        cuda_options = config.get('cuda_options', {})
        if not isinstance(cuda_options, dict):
            raise ValueError("cuda_options must be a dictionary")
        validated['cuda_options'] = cuda_options
        
        return validated
    
    @staticmethod
    def validate_video_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video configuration"""
        validated = {}
        
        # Validate fps
        fps = config.get('fps', 30)
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"Invalid fps: {fps}")
        validated['fps'] = fps
        
        # Validate resolution
        resolution = config.get('resolution', [1280, 720])
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"Invalid resolution: {resolution}")
        if not all(isinstance(x, int) and x > 0 for x in resolution):
            raise ValueError(f"Invalid resolution values: {resolution}")
        validated['resolution'] = resolution
        
        # Validate codec
        validated['codec'] = config.get('codec', 'mp4v')
        
        # Validate quality
        quality = config.get('quality', 'medium')
        valid_qualities = ['low', 'medium', 'high', 'ultra']
        if quality not in valid_qualities:
            raise ValueError(f"Invalid quality '{quality}'. Must be one of: {valid_qualities}")
        validated['quality'] = quality
        
        # Validate presets
        validated['presets'] = config.get('presets', {})
        
        return validated


class ConfigManager:
    """Unified configuration management system
    
    Provides centralized loading, validation, and access to configuration
    with support for hierarchical configs and environment-specific overrides.
    """
    
    def __init__(self, 
                 config_dir: str = "configs",
                 environment: str = "development",
                 auto_validate: bool = True):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, production, testing)
            auto_validate: Whether to automatically validate configs
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.auto_validate = auto_validate
        
        # Cached configurations
        self._base_config = None
        self._environment_config = None
        self._merged_config = None
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and merge configurations
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load and merge all configuration files"""
        # Load base configuration
        base_config_path = self.config_dir / "base.yaml"
        if base_config_path.exists():
            self._base_config = self._load_yaml_file(base_config_path)
        else:
            self._base_config = self._get_default_base_config()
        
        # Load environment-specific configuration
        env_config_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        if env_config_path.exists():
            self._environment_config = self._load_yaml_file(env_config_path)
        else:
            self._environment_config = {}
        
        # Load user-specific configuration (optional)
        user_config_path = self.config_dir / "user.yaml"
        user_config = {}
        if user_config_path.exists():
            user_config = self._load_yaml_file(user_config_path)
        
        # Merge configurations (user > environment > base)
        self._merged_config = self._deep_merge_configs(
            self._base_config,
            self._environment_config,
            user_config
        )
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        except FileNotFoundError:
            return {}
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")
    
    def _get_default_base_config(self) -> Dict[str, Any]:
        """Get default base configuration"""
        return {
            'project': {
                'name': 'DQN vs DDPG Comparison',
                'version': '2.0.0'
            },
            'gpu': {
                'enabled': 'auto',
                'mixed_precision': True,
                'cuda_options': {
                    'deterministic': False,
                    'benchmark': True,
                    'allow_tf32': True
                }
            },
            'video': {
                'fps': 30,
                'resolution': [1280, 720],
                'codec': 'mp4v',
                'quality': 'medium'
            },
            'output': {
                'base_dir': 'output',
                'models_dir': 'models',
                'videos_dir': 'videos',
                'logs_dir': 'logs',
                'results_dir': 'results'
            }
        }
    
    def _deep_merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple configuration dictionaries"""
        result = {}
        
        for config in configs:
            if not config:
                continue
                
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    def get_config(self, section: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """Get configuration section or entire config
        
        Args:
            section: Configuration section name (e.g., 'gpu', 'video')
                    If None, returns entire merged configuration
            
        Returns:
            Configuration dictionary or specific section
        """
        if section is None:
            return self._merged_config.copy()
        
        if '.' in section:
            # Support nested access like 'gpu.cuda_options'
            keys = section.split('.')
            result = self._merged_config
            for key in keys:
                if not isinstance(result, dict) or key not in result:
                    raise KeyError(f"Configuration section '{section}' not found")
                result = result[key]
            return result
        else:
            if section not in self._merged_config:
                raise KeyError(f"Configuration section '{section}' not found")
            return self._merged_config[section].copy()
    
    def get_gpu_config(self) -> GPUConfig:
        """Get validated GPU configuration"""
        gpu_config = self.get_config('gpu')
        
        if self.auto_validate:
            gpu_config = ConfigValidator.validate_gpu_config(gpu_config)
        
        return GPUConfig(**gpu_config)
    
    def get_video_config(self, preset: Optional[str] = None) -> VideoConfig:
        """Get validated video configuration
        
        Args:
            preset: Video quality preset to apply
            
        Returns:
            VideoConfig instance
        """
        video_config = self.get_config('video')
        
        if self.auto_validate:
            video_config = ConfigValidator.validate_video_config(video_config)
        
        # Apply preset if specified
        if preset and 'presets' in video_config and preset in video_config['presets']:
            preset_config = video_config['presets'][preset]
            video_config.update(preset_config)
        
        return VideoConfig(**video_config)
    
    def get_output_config(self) -> OutputConfig:
        """Get output directory configuration"""
        output_config = self.get_config('output')
        return OutputConfig(**output_config)
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get algorithm-specific configuration
        
        Args:
            algorithm: Algorithm name (dqn, ddpg)
            
        Returns:
            Algorithm configuration dictionary
        """
        # Try to load from merged config first
        if 'algorithms' in self._merged_config and algorithm in self._merged_config['algorithms']:
            return self._merged_config['algorithms'][algorithm].copy()
        
        # Fall back to separate algorithm config file
        algorithm_config_path = self.config_dir / "algorithms" / f"{algorithm}_config.yaml"
        if algorithm_config_path.exists():
            return self._load_yaml_file(algorithm_config_path)
        
        raise KeyError(f"Configuration for algorithm '{algorithm}' not found")
    
    def set_config(self, section: str, value: Any) -> None:
        """Set configuration value
        
        Args:
            section: Configuration section path (supports dot notation)
            value: Value to set
        """
        if '.' in section:
            keys = section.split('.')
            config = self._merged_config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
        else:
            self._merged_config[section] = value
    
    def save_config(self, 
                   file_path: Optional[str] = None, 
                   section: Optional[str] = None) -> None:
        """Save configuration to file
        
        Args:
            file_path: File path to save to (defaults to user.yaml)
            section: Specific section to save (saves entire config if None)
        """
        if file_path is None:
            file_path = self.config_dir / "user.yaml"
        else:
            file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get data to save
        if section is not None:
            data = self.get_config(section)
        else:
            data = self._merged_config
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def create_environment_configs(self) -> None:
        """Create default environment configuration files"""
        environments_dir = self.config_dir / "environments"
        environments_dir.mkdir(exist_ok=True)
        
        # Development environment
        dev_config = {
            'gpu': {
                'cuda_options': {
                    'deterministic': False  # Performance over reproducibility
                }
            },
            'video': {
                'quality': 'medium',  # Faster rendering for development
                'fps': 15
            },
            'logging': {
                'level': 'DEBUG',
                'console': True,
                'file': True
            }
        }
        
        # Production environment  
        prod_config = {
            'gpu': {
                'cuda_options': {
                    'deterministic': True  # Reproducibility for production
                }
            },
            'video': {
                'quality': 'high',  # High quality for final results
                'fps': 60
            },
            'logging': {
                'level': 'INFO',
                'console': False,
                'file': True
            }
        }
        
        # Testing environment
        test_config = {
            'gpu': {
                'enabled': False  # Use CPU for consistent testing
            },
            'video': {
                'quality': 'low',  # Fast testing
                'fps': 10
            },
            'logging': {
                'level': 'WARNING',
                'console': True,
                'file': False
            }
        }
        
        # Save environment configs
        configs = {
            'development.yaml': dev_config,
            'production.yaml': prod_config,
            'testing.yaml': test_config
        }
        
        for filename, config in configs.items():
            config_path = environments_dir / filename
            if not config_path.exists():
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def reload(self) -> None:
        """Reload all configurations"""
        self._load_configurations()
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigManager(environment={self.environment}, config_dir={self.config_dir})"


# Legacy support function for backward compatibility
def load_config(config_path: str) -> Dict[str, Any]:
    """Legacy config loading function for backward compatibility
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.isabs(config_path):
        # Try to find project root and make absolute
        current_dir = Path.cwd()
        potential_paths = [
            current_dir / config_path,
            current_dir.parent / config_path,
            current_dir.parent.parent / config_path
        ]
        
        for path in potential_paths:
            if path.exists():
                config_path = str(path)
                break
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)