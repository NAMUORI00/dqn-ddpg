"""
Base reinforcement learning agent class
Eliminates 90% code duplication between DQN and DDPG agents
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path

from ..core import ReplayBuffer, get_device, enable_mixed_precision, optimize_gpu_memory


class BaseReinforcementAgent(ABC):
    """Base class for reinforcement learning agents
    
    This abstract base class provides common functionality shared between
    DQN and DDPG agents, including:
    - GPU device management and Mixed Precision Training
    - Replay buffer initialization and management  
    - Common training statistics and metrics
    - Model saving/loading infrastructure
    - Memory optimization utilities
    
    By inheriting from this class, specific agents only need to implement
    their unique algorithm logic while reusing all common infrastructure.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: Union[float, Dict[str, float]],
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 device: Optional[torch.device] = None,
                 use_gpu_buffer: bool = False,
                 use_mixed_precision: bool = False):
        """Initialize base agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension  
            learning_rate: Learning rate(s) - float for single LR, dict for multiple
            gamma: Discount factor for future rewards
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            device: Computation device (auto-detected if None)
            use_gpu_buffer: Store replay buffer on GPU for faster sampling
            use_mixed_precision: Enable Mixed Precision Training for speedup
        """
        # Store basic configuration
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Device and GPU optimization setup
        self.device = device or get_device()
        self.use_mixed_precision = use_mixed_precision and str(self.device) == 'cuda'
        
        # Initialize Mixed Precision Training
        self.scaler = enable_mixed_precision() if self.use_mixed_precision else None
        
        # Create replay buffer with GPU support
        self.buffer = ReplayBuffer(
            buffer_size, 
            use_gpu=use_gpu_buffer, 
            device=self.device
        )
        
        # Training statistics
        self.training_steps = 0
        self.total_episodes = 0
        
        # Networks and optimizers will be set by subclasses
        self.networks = {}
        self.optimizers = {}
        
        # Learning rate storage for logging
        if isinstance(learning_rate, dict):
            self.learning_rates = learning_rate
        else:
            self.learning_rates = {'main': learning_rate}
    
    def _setup_mixed_precision_training(self) -> None:
        """Set up Mixed Precision Training if enabled"""
        if self.use_mixed_precision:
            if self.scaler is None:
                self.scaler = enable_mixed_precision()
            print(f"  ✓ Mixed Precision Training enabled on {self.device}")
    
    def _register_network(self, name: str, network: nn.Module) -> None:
        """Register a network for management
        
        Args:
            name: Network identifier
            network: PyTorch network module
        """
        self.networks[name] = network
        network.to(self.device)
    
    def _register_optimizer(self, name: str, optimizer: torch.optim.Optimizer) -> None:
        """Register an optimizer for management
        
        Args:
            name: Optimizer identifier  
            optimizer: PyTorch optimizer
        """
        self.optimizers[name] = optimizer
    
    def _get_batch_data(self, batch_size: Optional[int] = None) -> Optional[Tuple]:
        """Get batch data from replay buffer with GPU optimization
        
        Args:
            batch_size: Batch size override (uses self.batch_size if None)
            
        Returns:
            Batch tensors tuple or None if buffer not ready
        """
        batch_size = batch_size or self.batch_size
        
        if not self.buffer.is_ready(batch_size):
            return None
        
        # Use GPU-optimized sampling if available
        if hasattr(self.buffer, 'sample_tensor'):
            return self.buffer.sample_tensor(batch_size, self.device)
        else:
            # Legacy buffer compatibility
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            
            # Transfer to device
            states = torch.FloatTensor(states).to(self.device, non_blocking=True)
            actions = torch.FloatTensor(actions).to(self.device, non_blocking=True)
            rewards = torch.FloatTensor(rewards).to(self.device, non_blocking=True)
            next_states = torch.FloatTensor(next_states).to(self.device, non_blocking=True)
            dones = torch.FloatTensor(dones).to(self.device, non_blocking=True)
            
            return states, actions, rewards, next_states, dones
    
    def _apply_gradient_clipping(self, max_norm: float = 1.0) -> None:
        """Apply gradient clipping to all registered networks
        
        Args:
            max_norm: Maximum gradient norm
        """
        for network in self.networks.values():
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_norm)
    
    def _update_mixed_precision(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """Perform mixed precision backward pass and optimizer step
        
        Args:
            loss: Computed loss tensor
            optimizer: Optimizer to update
        """
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            self._apply_gradient_clipping()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._apply_gradient_clipping()
            optimizer.step()
    
    def _optimize_gpu_memory(self, force: bool = False) -> None:
        """Optimize GPU memory usage periodically
        
        Args:
            force: Force optimization regardless of step count
        """
        if (force or self.training_steps % 1000 == 0) and str(self.device) == 'cuda':
            optimize_gpu_memory()
    
    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray], 
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken (int for discrete, array for continuous)
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Convert discrete actions to array format for buffer consistency
        if isinstance(action, (int, np.integer)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        
        self.buffer.push(state, action, reward, next_state, done)
    
    @abstractmethod
    def select_action(self, state: np.ndarray, **kwargs) -> Union[int, np.ndarray]:
        """Select action based on current policy
        
        Must be implemented by subclasses to define algorithm-specific action selection.
        
        Args:
            state: Current state
            **kwargs: Algorithm-specific arguments
            
        Returns:
            Selected action (int for discrete, array for continuous)
        """
        pass
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Update agent networks based on batch from replay buffer
        
        Must be implemented by subclasses to define algorithm-specific learning.
        
        Returns:
            Dictionary of training metrics (loss, q_values, etc.)
        """
        pass
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics
        
        Returns:
            Dictionary containing training metrics and statistics
        """
        metrics = {
            'training_steps': self.training_steps,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer.capacity,
            'device': str(self.device),
            'mixed_precision_enabled': self.use_mixed_precision,
            'learning_rates': self.learning_rates,
        }
        
        # Add GPU memory info if available
        if self.device.type == 'cuda':
            try:
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                metrics.update({
                    'gpu_memory_allocated_gb': memory_allocated,
                    'gpu_memory_reserved_gb': memory_reserved
                })
            except:
                pass
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save agent state to file
        
        Args:
            path: File path to save to
        """
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build save dictionary
        save_dict = {
            'training_steps': self.training_steps,
            'total_episodes': self.total_episodes,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'learning_rates': self.learning_rates,
                'use_mixed_precision': self.use_mixed_precision,
            }
        }
        
        # Save all registered networks
        for name, network in self.networks.items():
            save_dict[f'{name}_state_dict'] = network.state_dict()
        
        # Save all registered optimizers
        for name, optimizer in self.optimizers.items():
            save_dict[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
        
        # Save mixed precision scaler if used
        if self.use_mixed_precision and self.scaler is not None:
            save_dict['scaler_state_dict'] = self.scaler.state_dict()
        
        # Allow subclasses to add custom data
        custom_data = self._get_custom_save_data()
        if custom_data:
            save_dict.update(custom_data)
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load agent state from file
        
        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore basic state
        self.training_steps = checkpoint.get('training_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        
        # Load all registered networks
        for name, network in self.networks.items():
            key = f'{name}_state_dict'
            if key in checkpoint:
                network.load_state_dict(checkpoint[key])
        
        # Load all registered optimizers
        for name, optimizer in self.optimizers.items():
            key = f'{name}_optimizer_state_dict'
            if key in checkpoint:
                optimizer.load_state_dict(checkpoint[key])
        
        # Load mixed precision scaler if used
        if (self.use_mixed_precision and self.scaler is not None and 
            'scaler_state_dict' in checkpoint):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Allow subclasses to load custom data
        self._load_custom_save_data(checkpoint)
    
    def _get_custom_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data to save
        
        Override in subclasses to save additional state.
        
        Returns:
            Dictionary of custom data to save
        """
        return {}
    
    def _load_custom_save_data(self, checkpoint: Dict[str, Any]) -> None:
        """Load algorithm-specific data from checkpoint
        
        Override in subclasses to load additional state.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        pass
    
    def reset_for_new_episode(self) -> None:
        """Reset agent state for new episode
        
        Called at the start of each episode. Override in subclasses for
        algorithm-specific reset logic (e.g., noise process reset).
        """
        self.total_episodes += 1
    
    def __repr__(self) -> str:
        """String representation of agent"""
        return (f"{self.__class__.__name__}("
                f"state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"device={self.device}, "
                f"training_steps={self.training_steps})")