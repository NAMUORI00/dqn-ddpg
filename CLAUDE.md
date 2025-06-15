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
# 1. Run basic algorithm demonstration
python tests/simple_demo.py

# 2. Run same environment comparison (core innovation)
python experiments/same_environment_comparison.py

# 3. Generate comprehensive visualization
python create_comprehensive_visualization.py

# 4. Generate presentation materials
python generate_presentation_materials.py
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

# Real-time synchronized learning + gameplay videos (Latest Feature)
python create_realtime_combined_videos.py --all --duration 20
python create_realtime_combined_videos.py --cartpole --duration 15
python create_realtime_combined_videos.py --pendulum --duration 15

# Fast synchronized training videos
python create_fast_synchronized_video.py --all --duration 20

# Success/failure contrast videos
python create_success_failure_videos.py

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

# Same environment comparison (core innovation)
python experiments/same_environment_comparison.py

# All experiments comprehensive
python run_all_experiments.py

# Deterministic policy analysis
python experiments/analyze_deterministic_policy.py
```

### Documentation and Presentation
```bash
# Generate all presentation materials
python generate_presentation_materials.py

# Check presentation materials status
python check_presentation_materials.py

# Organize project reports
python organize_reports.py
```

## Architecture Overview

### Core Algorithm Components

**Agents** (`src/agents/`):
- `dqn_agent.py`: Implements DQN with implicit deterministic policy (argmax over Q-values)
- `ddpg_agent.py`: Implements DDPG with explicit deterministic policy (direct actor output)
- `discretized_dqn_agent.py`: **Core Innovation** - DQN agent adapted for continuous action spaces via discretization

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
- `create_realtime_combined_videos.py`: **Latest** - Real-time learning graphs + gameplay in 2x2 layout
- `create_synchronized_training_video.py`: Synchronized training with episode-specific gameplay recording
- `create_success_failure_videos.py`: Success/failure contrast videos for educational demonstrations

**Environment Integration** (`src/environments/`):
- `video_wrappers.py`: Environment wrappers that integrate video recording
- `wrappers.py`: Standard environment wrappers for CartPole-v1 (DQN) and Pendulum-v1 (DDPG)
- `continuous_cartpole.py`: **Core Innovation** - Continuous action version of CartPole for fair algorithm comparison

### Key Design Patterns

1. **Deterministic Policy Comparison** (Core Educational Goal):
   - DQN: Implicit deterministic policy via `q_values.argmax()`
   - DDPG: Explicit deterministic policy via `actor_network(state)`

2. **Fair Comparison Innovation** (Project's Main Contribution):
   - **Problem**: Traditional comparison uses different environments (CartPole vs Pendulum)
   - **Solution**: ContinuousCartPole environment + DiscretizedDQN agent
   - **Result**: Same environment comparison revealing DQN 13.2x performance advantage

3. **Action Space Handling**:
   - Standard DQN: Discrete actions (0, 1 for CartPole)
   - DDPG: Continuous actions ([-2, 2] for Pendulum)
   - **DiscretizedDQN**: Continuous actions discretized into bins for DQN compatibility

4. **Video System Design**:
   - **Dual Recording**: Simultaneous low-quality full recording + high-quality selective recording
   - **Pipeline Architecture**: Separate rendering pipeline for post-training visualization
   - **Real-time Synchronized Videos**: Learning graphs + gameplay in unified 2x2 layout (Latest Innovation)
   - **FFmpeg-Independent**: OpenCV fallback for broader compatibility
   - **Sample Data Integration**: Generate demos without actual training

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
- **Real-time Synchronized Visualization**: Latest innovation combining learning curves with actual gameplay
- **Korean/English Documentation**: Supports multilingual educational content

### Latest Video Innovation: Real-time Synchronized Videos

**Core Innovation**: `create_realtime_combined_videos.py` generates 2x2 layout videos showing:
- **Top Left**: DQN learning graph (real-time)
- **Top Right**: DDPG learning graph (real-time)  
- **Bottom Left**: DQN gameplay video (synchronized)
- **Bottom Right**: DDPG gameplay video (synchronized)

**Key Features**:
- Synchronized with actual training episode counts (CartPole: 500 episodes, Pendulum: 300 episodes)
- Uses pre-recorded success/failure videos from `videos/environment_success_failure/`
- Environment-specific episode targeting for accurate representation
- Progress-based video selection (failure videos early, success videos later)
- Fallback system when specific video types are unavailable

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
├── combined/               # Latest: Real-time learning + gameplay synchronized videos
├── synchronized/           # Episode-synchronized training videos
├── environment_success_failure/  # Success/failure contrast videos
└── temp/                   # Temporary processing files
```

## Documentation Structure

The project maintains comprehensive documentation in the `docs/` directory with Korean language content:

### Core Documentation
- `docs/README.md`: Main documentation index with project overview and usage guides
- `docs/final_reports/`: Presentation-ready comprehensive project reports
- `docs/experiment_reports/`: Detailed experimental results and analysis
- `docs/analysis_reports/`: Theoretical analysis and algorithm comparisons
- `docs/documentation/`: System guides and development logs

### Key Reports
- **`FINAL_REPORT.md`**: Complete project summary suitable for presentations
- **`DQN_vs_DDPG_동일환경_비교분석.md`**: Core finding - DQN 13.2x performance advantage
- **`VISUAL_MATERIALS_REPORT.md`**: Verification that all presentation materials can be code-generated

## Environment Specifications

### DQN Environment (CartPole-v1)
- **Action Space**: Discrete (0: left, 1: right)
- **Observation Space**: Continuous (cart position, velocity, pole angle, angular velocity)
- **Success Criteria**: Average reward ≥ 475 over 100 episodes

### DDPG Environment (Pendulum-v1)
- **Action Space**: Continuous (torque in [-2, 2])
- **Observation Space**: Continuous (cos(θ), sin(θ), angular velocity)
- **Success Criteria**: Average reward ≥ -200 (higher is better)

### ContinuousCartPole (Innovation for Fair Comparison)
- **Action Space**: Continuous (force in [-1, 1] mapped to [-10, 10])
- **Observation Space**: Same as CartPole-v1 (cart position, velocity, pole angle, angular velocity)
- **Physics**: Identical to CartPole-v1 but accepts continuous force values
- **Purpose**: Enable DQN vs DDPG comparison in identical environment

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
- **New Environments**: Create wrappers in `src/environments/` following ContinuousCartPole pattern
- **Video Features**: Extend `VideoRenderingPipeline` class in `src/core/video_pipeline.py`
- **Analysis Tools**: Add functions to `experiments/visualizations.py`
- **New Algorithms**: Follow DiscretizedDQN pattern for adapting algorithms to different action spaces

### Debugging and Testing
- Use `tests/simple_demo.py` for quick algorithm behavior verification
- Use `video_test.py` and `patch_video_test.py` for video system diagnostics
- Check `videos/temp/` for intermediate processing files when debugging video issues

### File Organization
- Large generated files (videos, models, results) are gitignored
- Only source code, configs, and small JSON results are tracked
- Korean documentation in `docs/` provides comprehensive guides organized by category
- Presentation materials can be auto-generated with `generate_presentation_materials.py`

## Core Innovation Summary

This project's main contribution is enabling **fair algorithm comparison** through:

1. **ContinuousCartPole Environment**: CartPole physics with continuous action space
2. **DiscretizedDQN Agent**: DQN adapted for continuous actions via discretization
3. **Same Environment Comparison**: Both algorithms tested in identical conditions
4. **Key Finding**: DQN outperforms DDPG by 13.2x in continuous environment, contradicting conventional wisdom

The methodology eliminates environment bias and provides pure algorithmic performance comparison, revealing that environment compatibility matters more than theoretical algorithm design for specific action spaces.

## Recent Updates and Findings (June 2025)

### Experimental Validation Completed
- **CartPole Environment**: DQN achieves 13.2x better performance than DDPG
- **Pendulum Environment**: DDPG achieves 16.1x better performance than DQN  
- **Bidirectional Validation**: Both algorithms tested in both native and cross environments
- **Core Principle Established**: Environment compatibility supersedes theoretical algorithm advantages

### Latest Video Capabilities
- **Real-time Synchronized Videos**: Learning curves + gameplay in unified visualization
- **Success/Failure Contrast**: Clear visual demonstration of algorithm suitability
- **Educational Presentation Format**: All materials optimized for academic/professional presentation
- **Reproducible Results**: Complete experimental pipeline generates consistent outcomes

### Presentation Materials
All visual materials, charts, and videos can be regenerated programmatically using:
```bash
python generate_presentation_materials.py
python check_presentation_materials.py
```

This ensures reproducibility and allows for parameter adjustments while maintaining visual consistency across all deliverables.