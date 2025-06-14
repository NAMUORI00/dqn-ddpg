# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational reinforcement learning project comparing DQN (Deep Q-Network) and DDPG (Deep Deterministic Policy Gradient) algorithms, focusing on their deterministic policy characteristics. The project includes sophisticated video recording capabilities and comprehensive analysis tools.

**Key Educational Focus**: Demonstrating the difference between implicit deterministic policies (DQN's Q-value argmax) and explicit deterministic policies (DDPG's direct actor output).

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n ddpg_dqn python=3.11
conda activate ddpg_dqn

# Install dependencies
pip install -r requirements.txt
```

### Quick Start (Recommended Order)
```bash
# 1. Generate video demo (fastest way to see results)
python quick_video_demo.py --duration 15

# 2. Run basic algorithm demonstration
python tests/simple_demo.py

# 3. Generate comprehensive learning visualization
python render_learning_video.py --sample-data --all

# 4. Run full experiment with training
python run_experiment.py --save-models --results-dir results
```

### Video Generation Commands
```bash
# Quick 15-second demo video
python quick_video_demo.py --duration 15

# Learning process animation with sample data
python render_learning_video.py --sample-data --learning-only --duration 30

# Complete educational video (intro + learning + comparison + outro)
python render_learning_video.py --sample-data --all

# Use real training results
python render_learning_video.py --dqn-results results/dqn_results.json --ddpg-results results/ddpg_results.json

# Algorithm comparison video
python create_comparison_video.py --auto

# Test video system compatibility
python video_test.py
python patch_video_test.py
```

### Algorithm Testing and Analysis
```bash
# Basic deterministic policy demonstration
python tests/simple_demo.py

# Detailed analysis with visualizations
python tests/detailed_test.py

# Test video recording during training
python tests/test_video_recording.py

# Test dual recording system
python tests/simple_dual_test.py
```

### Training and Experiments
```bash
# Simple training run
python simple_training.py

# Full experiment with all analysis
python run_experiment.py --save-models --results-dir results

# Training with video recording
python run_experiment.py --record-video --dual-video
```

## Architecture Overview

### Core Algorithm Components

**Agents** (`src/agents/`):
- `dqn_agent.py`: Implements DQN with implicit deterministic policy (argmax over Q-values)
- `ddpg_agent.py`: Implements DDPG with explicit deterministic policy (direct actor output)

**Networks** (`src/networks/`):
- `q_network.py`: Deep Q-Network for DQN
- `actor.py`: Actor network for DDPG (continuous action output)
- `critic.py`: Critic network for DDPG (Q-value estimation)

**Core Utilities** (`src/core/`):
- `buffer.py`: Experience replay buffer
- `noise.py`: Ornstein-Uhlenbeck noise process for exploration
- `utils.py`: Common utilities (soft updates, device management, seeding)

### Video System Architecture

**Video Pipeline** (`src/core/`):
- `video_pipeline.py`: Main video rendering pipeline for learning process visualization
- `video_manager.py`: Advanced video recording with dual-quality system
- `video_utils.py`: Common video utilities (encoding, layout, sample data generation)
- `dual_recorder.py`: Simultaneous low/high quality recording during training
- `recording_scheduler.py`: Performance-based recording scheduler

**Video Scripts** (root level):
- `quick_video_demo.py`: Fast demo video generation (15-60 seconds)
- `render_learning_video.py`: Comprehensive learning process visualization
- `create_comparison_video.py`: Side-by-side algorithm comparison videos

**Environment Integration** (`src/environments/`):
- `video_wrappers.py`: Environment wrappers that integrate video recording
- `wrappers.py`: Standard environment wrappers for CartPole-v1 (DQN) and Pendulum-v1 (DDPG)

### Key Design Patterns

1. **Deterministic Policy Comparison** (Core Educational Goal):
   - DQN: Implicit deterministic policy via `q_values.argmax()`
   - DDPG: Explicit deterministic policy via `actor_network(state)`

2. **Action Space Handling**:
   - DQN: Discrete actions (0, 1 for CartPole)
   - DDPG: Continuous actions ([-2, 2] for Pendulum)

3. **Exploration Strategies**:
   - DQN: ε-greedy exploration
   - DDPG: Gaussian noise added to deterministic policy

4. **Video System Design**:
   - **Dual Recording**: Simultaneous low-quality full recording + high-quality selective recording
   - **Pipeline Architecture**: Separate rendering pipeline for post-training visualization
   - **Fallback System**: OpenCV backend when FFmpeg unavailable
   - **Sample Data Integration**: Ability to generate demos without actual training

## Configuration System

### Algorithm Configuration
- `configs/dqn_config.yaml`: DQN hyperparameters (learning rate, epsilon decay, buffer size)
- `configs/ddpg_config.yaml`: DDPG hyperparameters (actor/critic learning rates, noise parameters)

### Video System Configuration
- `configs/video_recording.yaml`: Real-time video recording during training (dual-quality system)
- `configs/video_config.yaml`: Video rendering and encoding settings
- `configs/pipeline_config.yaml`: Learning process visualization pipeline settings

### Configuration Integration
The project uses YAML-based configuration with:
- **Hierarchical Settings**: Algorithm-specific and video-specific configurations
- **Quality Presets**: Different video quality levels for demo vs presentation use
- **Fallback Mechanisms**: Default values when specific configs are missing
- **Runtime Overrides**: Command-line arguments can override config file settings

## Video System Overview

### Two Distinct Video Systems

1. **Real-time Training Recording** (`video_manager.py`, `dual_recorder.py`):
   - Records actual agent gameplay during training
   - Dual-quality recording (low-res continuous + high-res selective)
   - Performance-based triggers for important episodes
   - Integrates with training loop via environment wrappers

2. **Post-training Visualization Pipeline** (`video_pipeline.py`):
   - Generates educational videos from training data
   - Learning curve animations, algorithm comparisons
   - Uses sample data when real training data unavailable
   - Optimized for educational/presentation content

### Key Video Features
- **FFmpeg-Independent**: Uses OpenCV fallback for broader compatibility
- **Sample Data Capability**: Can generate demos without running training
- **Multiple Output Formats**: Quick demos, detailed learning animations, comparison videos
- **Korean/English Documentation**: Supports multilingual educational content

### Video Output Structure
```
videos/
├── dqn/
│   ├── full/               # Low-quality continuous recording
│   └── highlights/         # High-quality milestone recordings
├── ddpg/
│   ├── full/
│   └── highlights/
├── comparison/             # Side-by-side comparison videos
├── pipeline/               # Learning process animations
└── temp/                   # Temporary processing files
```

## Language and Documentation

This project uses **Korean language** for documentation and comments. Key documentation files are in the `docs/` directory with comprehensive theoretical explanations and usage guides.

## Environment Specifications

### DQN Environment (CartPole-v1)
- **Action Space**: Discrete (0: left, 1: right)
- **Observation Space**: Continuous (cart position, velocity, pole angle, angular velocity)
- **Success Criteria**: Average reward ≥ 475 over 100 episodes

### DDPG Environment (Pendulum-v1)
- **Action Space**: Continuous (torque in [-2, 2])
- **Observation Space**: Continuous (cos(θ), sin(θ), angular velocity)
- **Success Criteria**: Average reward ≥ -200 (higher is better)

## Key Implementation Details

### Neural Network Architecture
- **DQN**: Simple feedforward network with ReLU activations
- **DDPG Actor**: Tanh output layer for bounded continuous actions
- **DDPG Critic**: Outputs single Q-value for state-action pairs

### Training Specifics
- **Target Networks**: Both algorithms use target networks for stable training
- **Experience Replay**: Uniform sampling from replay buffer
- **Soft Updates**: DDPG uses Polyak averaging (τ=0.005) for target network updates
- **Hard Updates**: DQN uses periodic hard updates (every 100 steps)

## Development Workflow

### Working with Configurations
- Edit YAML files in `configs/` to modify hyperparameters
- Use command-line arguments to override config values for quick testing
- Video quality presets available in `configs/video_config.yaml`

### Adding New Features
- **New Environments**: Create wrappers in `src/environments/` following existing pattern
- **Video Features**: Extend `VideoRenderingPipeline` class in `src/core/video_pipeline.py`
- **Analysis Tools**: Add functions to `experiments/visualizations.py`

### Debugging and Testing
- Use `tests/simple_demo.py` for quick algorithm behavior verification
- Use `video_test.py` and `patch_video_test.py` for video system diagnostics
- Check `videos/temp/` for intermediate processing files when debugging video issues

### File Organization
- Large generated files (videos, models, results) are gitignored
- Only source code, configs, and small JSON results are tracked
- Korean documentation in `docs/` provides comprehensive guides