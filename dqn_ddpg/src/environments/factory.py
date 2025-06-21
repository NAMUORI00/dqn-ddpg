"""
Environment Factory
Unified environment creation system eliminating code duplication
"""

import gymnasium as gym
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .continuous_cartpole import ContinuousCartPole
from .wrappers import create_dqn_env, create_ddpg_env
from .video_wrappers import VideoRecordingWrapper
from ..core.config_manager import ConfigManager, VideoConfig


class EnvironmentFactory:
    """Factory for creating environments with consistent configuration
    
    This factory eliminates the 37 instances of duplicated environment
    creation code found across the project by providing a unified interface
    for creating environments with appropriate wrappers and configurations.
    """
    
    # Environment type mappings
    DISCRETE_ENVIRONMENTS = {
        'cartpole-v1': 'CartPole-v1',
        'cartpole': 'CartPole-v1',
        'lunarlander-v2': 'LunarLander-v2',
        'lunarlander': 'LunarLander-v2',
        'acrobot-v1': 'Acrobot-v1',
        'acrobot': 'Acrobot-v1'
    }
    
    CONTINUOUS_ENVIRONMENTS = {
        'pendulum-v1': 'Pendulum-v1',
        'pendulum': 'Pendulum-v1',
        'continuous-cartpole': 'ContinuousCartPole',
        'continuous_cartpole': 'ContinuousCartPole',
        'mountaincarcontinuous-v0': 'MountainCarContinuous-v0',
        'lunarlander-continuous': 'LunarLanderContinuous-v2'
    }
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize environment factory
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
    def create_env(self, 
                   env_name: str,
                   agent_type: str,
                   video_config: Optional[Union[VideoConfig, Dict[str, Any]]] = None,
                   training: bool = True,
                   seed: Optional[int] = None,
                   **kwargs) -> gym.Env:
        """Create environment with appropriate wrappers
        
        Args:
            env_name: Environment name (supports aliases)
            agent_type: Agent type ('dqn', 'ddpg', 'auto')
            video_config: Video recording configuration
            training: Whether this is for training (affects wrappers)
            seed: Random seed
            **kwargs: Additional environment arguments
            
        Returns:
            Configured environment
            
        Raises:
            ValueError: If environment or agent type is invalid
        """
        # Normalize environment name
        env_name_normalized = self._normalize_env_name(env_name)
        
        # Auto-detect agent type if needed
        if agent_type == 'auto':
            agent_type = self._detect_agent_type(env_name_normalized)
        
        # Validate agent type
        if agent_type not in ['dqn', 'ddpg']:
            raise ValueError(f"Invalid agent type: {agent_type}. Must be 'dqn', 'ddpg', or 'auto'")
        
        # Create base environment
        env = self._create_base_env(env_name_normalized, **kwargs)
        
        # Apply agent-specific wrappers
        if agent_type == 'dqn':
            env = self._apply_dqn_wrappers(env, training)
        elif agent_type == 'ddpg':
            env = self._apply_ddpg_wrappers(env, training)
        
        # Apply video recording if requested
        if video_config is not None:
            env = self._apply_video_wrapper(env, video_config)
        
        # Set seed if provided (gymnasium v1.0+ style)
        if seed is not None:
            # Store seed for reset
            env._factory_seed = seed
        
        return env
    
    def create_training_env(self,
                           algorithm: str,
                           video_recording: bool = False,
                           video_preset: str = "medium") -> gym.Env:
        """Create training environment using algorithm configuration
        
        Args:
            algorithm: Algorithm name ('dqn' or 'ddpg')
            video_recording: Whether to enable video recording
            video_preset: Video quality preset
            
        Returns:
            Configured training environment
        """
        # Get algorithm configuration
        alg_config = self.config_manager.get_algorithm_config(algorithm)
        env_config = alg_config.get('environment', {})
        
        env_name = env_config.get('name', self._get_default_env_name(algorithm))
        
        # Get video config if recording enabled
        video_config = None
        if video_recording:
            video_config = self.config_manager.get_video_config(preset=video_preset)
        
        # Get training seed
        training_config = self.config_manager.get_config('training')
        seed = training_config.get('seed')
        
        return self.create_env(
            env_name=env_name,
            agent_type=algorithm,
            video_config=video_config,
            training=True,
            seed=seed,
            max_episode_steps=env_config.get('max_episode_steps')
        )
    
    def create_evaluation_env(self,
                             algorithm: str,
                             video_recording: bool = True,
                             video_preset: str = "high") -> gym.Env:
        """Create evaluation environment
        
        Args:
            algorithm: Algorithm name ('dqn' or 'ddpg')
            video_recording: Whether to enable video recording
            video_preset: Video quality preset
            
        Returns:
            Configured evaluation environment
        """
        # Get algorithm configuration
        alg_config = self.config_manager.get_algorithm_config(algorithm)
        env_config = alg_config.get('environment', {})
        
        env_name = env_config.get('name', self._get_default_env_name(algorithm))
        
        # Get video config
        video_config = None
        if video_recording:
            video_config = self.config_manager.get_video_config(preset=video_preset)
        
        return self.create_env(
            env_name=env_name,
            agent_type=algorithm,
            video_config=video_config,
            training=False,
            seed=42,  # Fixed seed for evaluation
            max_episode_steps=env_config.get('max_episode_steps')
        )
    
    def _normalize_env_name(self, env_name: str) -> str:
        """Normalize environment name to canonical form"""
        env_name_lower = env_name.lower().replace('_', '-')
        
        # Check discrete environments
        if env_name_lower in self.DISCRETE_ENVIRONMENTS:
            return self.DISCRETE_ENVIRONMENTS[env_name_lower]
        
        # Check continuous environments
        if env_name_lower in self.CONTINUOUS_ENVIRONMENTS:
            return self.CONTINUOUS_ENVIRONMENTS[env_name_lower]
        
        # If not found in mappings, assume it's a valid gym environment ID
        return env_name
    
    def _detect_agent_type(self, env_name: str) -> str:
        """Auto-detect appropriate agent type for environment"""
        env_name_lower = env_name.lower()
        
        # Check if it's a known discrete environment
        for discrete_env in self.DISCRETE_ENVIRONMENTS.values():
            if discrete_env.lower() in env_name_lower:
                return 'dqn'
        
        # Check if it's a known continuous environment
        for continuous_env in self.CONTINUOUS_ENVIRONMENTS.values():
            if continuous_env.lower() in env_name_lower:
                return 'ddpg'
        
        # Default fallback - try to create environment and check action space
        try:
            temp_env = gym.make(env_name)
            if hasattr(temp_env.action_space, 'n'):  # Discrete
                temp_env.close()
                return 'dqn'
            else:  # Continuous
                temp_env.close()
                return 'ddpg'
        except:
            # If all else fails, default to DQN
            return 'dqn'
    
    def _create_base_env(self, env_name: str, **kwargs) -> gym.Env:
        """Create base environment"""
        if env_name == 'ContinuousCartPole':
            # Create our custom continuous CartPole environment
            return ContinuousCartPole(**kwargs)
        else:
            # Create standard gym environment
            return gym.make(env_name, **kwargs)
    
    def _apply_dqn_wrappers(self, env: gym.Env, training: bool) -> gym.Env:
        """Apply DQN-specific wrappers"""
        # Use existing DQN wrapper function for consistency
        return create_dqn_env(env.spec.id if hasattr(env, 'spec') and env.spec else str(env))
    
    def _apply_ddpg_wrappers(self, env: gym.Env, training: bool) -> gym.Env:
        """Apply DDPG-specific wrappers"""
        # Use existing DDPG wrapper function for consistency
        return create_ddpg_env(env.spec.id if hasattr(env, 'spec') and env.spec else str(env))
    
    def _apply_video_wrapper(self, env: gym.Env, video_config: Union[VideoConfig, Dict[str, Any]]) -> gym.Env:
        """Apply video recording wrapper"""
        if isinstance(video_config, dict):
            video_config = VideoConfig(**video_config)
        
        # Get output directory
        output_config = self.config_manager.get_output_config()
        video_dir = output_config.get_path('videos', 'training')
        video_dir.mkdir(parents=True, exist_ok=True)
        
        return VideoRecordingWrapper(
            env=env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,  # Record all episodes
            video_length=0,  # Record entire episode
            name_prefix="training"
        )
    
    def _get_default_env_name(self, algorithm: str) -> str:
        """Get default environment name for algorithm"""
        defaults = {
            'dqn': 'CartPole-v1',
            'ddpg': 'Pendulum-v1'
        }
        return defaults.get(algorithm, 'CartPole-v1')
    
    def get_supported_environments(self) -> Dict[str, list]:
        """Get list of supported environments by type"""
        return {
            'discrete': list(self.DISCRETE_ENVIRONMENTS.keys()),
            'continuous': list(self.CONTINUOUS_ENVIRONMENTS.keys())
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"EnvironmentFactory(config_manager={self.config_manager})"


# Convenience functions for backward compatibility
def create_env_for_algorithm(algorithm: str, 
                           training: bool = True,
                           video_recording: bool = False,
                           config_manager: Optional[ConfigManager] = None) -> gym.Env:
    """Create environment for specific algorithm (backward compatibility)
    
    Args:
        algorithm: Algorithm name ('dqn' or 'ddpg')
        training: Whether this is for training
        video_recording: Whether to enable video recording
        config_manager: Configuration manager instance
        
    Returns:
        Configured environment
    """
    factory = EnvironmentFactory(config_manager)
    
    if training:
        return factory.create_training_env(algorithm, video_recording)
    else:
        return factory.create_evaluation_env(algorithm, video_recording)


def auto_create_env(env_name: str, 
                   training: bool = True,
                   video_recording: bool = False,
                   config_manager: Optional[ConfigManager] = None) -> gym.Env:
    """Auto-create environment with agent type detection
    
    Args:
        env_name: Environment name
        training: Whether this is for training
        video_recording: Whether to enable video recording
        config_manager: Configuration manager instance
        
    Returns:
        Configured environment
    """
    factory = EnvironmentFactory(config_manager)
    
    video_config = None
    if video_recording:
        video_config = factory.config_manager.get_video_config()
    
    return factory.create_env(
        env_name=env_name,
        agent_type='auto',  # Auto-detect
        video_config=video_config,
        training=training
    )