"""
Improved DQN Agent with optimized hyperparameters for CartPole-v1 convergence

Based on research from multiple sources including:
- ADG Efficiency DQN tuning
- PyTorch DQN tutorial  
- Saturn Cloud DQN troubleshooting
- Stack Overflow best practices

Key improvements:
1. Reduced learning rate (0.001 → 0.0001)
2. Increased target update frequency (100 → 10000)
3. Larger batch size (64 → 128)
4. Slower epsilon decay (0.995 → 0.9995)
5. Optional Double DQN implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .dqn_agent import DQNAgent
from ..networks.q_network import QNetwork


class ImprovedDQNAgent(DQNAgent):
    """
    Improved DQN Agent with optimized hyperparameters for CartPole-v1
    
    Improvements based on research:
    - Conservative learning rate for stability
    - Infrequent target updates for stability  
    - Larger batch size for better gradient quality
    - Slower exploration decay
    - Optional Double DQN for reduced overestimation
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.0001,    # Reduced from 0.001
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,    # Slower decay from 0.995
                 buffer_size: int = 100000,
                 batch_size: int = 128,             # Increased from 64
                 target_update_freq: int = 10000,  # Much higher from 100
                 device: Optional[torch.device] = None,
                 use_double_dqn: bool = False,     # Optional Double DQN
                 use_gpu_buffer: bool = False,
                 use_mixed_precision: bool = False):
        """
        Initialize Improved DQN Agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate (conservative: 0.0001)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate (slower: 0.9995)
            buffer_size: Replay buffer size
            batch_size: Training batch size (larger: 128)
            target_update_freq: Target network update frequency (higher: 10000)
            device: Computation device
            use_double_dqn: Whether to use Double DQN
            use_gpu_buffer: Whether to use GPU replay buffer
            use_mixed_precision: Whether to use mixed precision training
        """
        # Call parent constructor with improved parameters
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
            use_gpu_buffer=use_gpu_buffer,
            use_mixed_precision=use_mixed_precision
        )
        
        self.use_double_dqn = use_double_dqn
        
        # Track convergence metrics
        self.convergence_window = 100
        self.recent_rewards = []
        self.convergence_threshold = 475.0  # CartPole-v1 solving criterion
        
        print(f"🚀 Improved DQN Agent initialized:")
        print(f"   Learning Rate: {learning_rate} (conservative)")
        print(f"   Target Update: {target_update_freq} steps (stable)")
        print(f"   Batch Size: {batch_size} (larger)")
        print(f"   Epsilon Decay: {epsilon_decay} (slower)")
        print(f"   Double DQN: {use_double_dqn}")
    
    def update(self) -> Dict[str, float]:
        """
        Enhanced update method with optional Double DQN
        
        Returns:
            Dictionary containing loss and other metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch_data = self._get_batch_data()
        if batch_data is None:
            return {}
        
        states, actions, rewards, next_states, dones = batch_data
        
        # Current Q values  
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        current_q_values = self.q_network(states).gather(1, actions.long())
        
        # Target Q values calculation
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use main network for action selection, target for evaluation
                next_q_values_main = self.q_network(next_states)
                next_actions = next_q_values_main.argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions.long())
            else:
                # Standard DQN: Use target network for both action selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = F.mse_loss(current_q_values, target_q_values)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        self._update_mixed_precision(loss, self.optimizer)
        
        # Update target network
        if self.update_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at step {self.update_step}")
        
        self.update_step += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # GPU memory optimization
        self._optimize_gpu_memory()
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item(),
            'update_step': self.update_step
        }
    
    def track_episode_reward(self, reward: float) -> Dict[str, float]:
        """
        Track episode rewards for convergence analysis
        
        Args:
            reward: Episode reward
            
        Returns:
            Dictionary with convergence metrics
        """
        self.recent_rewards.append(reward)
        
        # Keep only recent rewards for convergence checking
        if len(self.recent_rewards) > self.convergence_window:
            self.recent_rewards.pop(0)
        
        # Calculate metrics
        if len(self.recent_rewards) >= self.convergence_window:
            avg_reward = np.mean(self.recent_rewards)
            std_reward = np.std(self.recent_rewards)
            converged = avg_reward >= self.convergence_threshold
            
            return {
                'avg_reward_100': avg_reward,
                'std_reward_100': std_reward,
                'converged': converged,
                'episodes_tracked': len(self.recent_rewards)
            }
        else:
            return {
                'avg_reward_100': np.mean(self.recent_rewards),
                'std_reward_100': np.std(self.recent_rewards) if len(self.recent_rewards) > 1 else 0,
                'converged': False,
                'episodes_tracked': len(self.recent_rewards)
            }
    
    def get_convergence_status(self) -> Dict[str, float]:
        """Get current convergence status"""
        if len(self.recent_rewards) >= self.convergence_window:
            avg_reward = np.mean(self.recent_rewards)
            progress = min(avg_reward / self.convergence_threshold, 1.0)
            
            return {
                'avg_reward': avg_reward,
                'threshold': self.convergence_threshold,
                'progress': progress,
                'converged': avg_reward >= self.convergence_threshold
            }
        else:
            return {
                'avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
                'threshold': self.convergence_threshold,
                'progress': 0.0,
                'converged': False
            }
    
    def _get_custom_save_data(self) -> Dict[str, any]:
        """Get improved DQN specific data to save"""
        base_data = super()._get_custom_save_data()
        base_data.update({
            'use_double_dqn': self.use_double_dqn,
            'target_update_freq': self.target_update_freq,
            'convergence_threshold': self.convergence_threshold,
            'recent_rewards': self.recent_rewards
        })
        return base_data
    
    def _load_custom_save_data(self, checkpoint: Dict[str, any]) -> None:
        """Load improved DQN specific data from checkpoint"""
        super()._load_custom_save_data(checkpoint)
        self.use_double_dqn = checkpoint.get('use_double_dqn', False)
        self.target_update_freq = checkpoint.get('target_update_freq', 10000)
        self.convergence_threshold = checkpoint.get('convergence_threshold', 475.0)
        self.recent_rewards = checkpoint.get('recent_rewards', [])