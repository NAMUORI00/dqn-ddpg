# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational reinforcement learning project comparing DQN and DDPG algorithms, with focus on deterministic policy analysis and sophisticated video generation capabilities.

**Core Innovation**: Same-environment comparison methodology revealing DQN's 13.2x performance advantage in continuous environments.

## Development Commands

### Environment Setup
```bash
# 프로젝트 로컬 conda 환경 (권장)
cd /path/to/dqn,ddpg
conda create --prefix ./.conda python=3.11 -y
./.conda/bin/pip install -r requirements.txt

# 또는 전역 conda 환경
conda create -n ddpg_dqn python=3.11
conda activate ddpg_dqn
pip install -r requirements.txt
```

**로컬 환경 사용법**:
```bash
# 패키지 설치
./.conda/bin/pip install package_name

# Python 실행
./.conda/bin/python script.py

# 주의: .conda/ 디렉토리는 .gitignore에 포함되어 버전 관리에서 제외됨
```

### Quick Start (Recommended Order)
```bash
# 1. Basic demo
python tests/simple_demo.py

# 2. Main experiments  
python scripts/experiments/run_experiment.py
python scripts/experiments/run_same_env_experiment.py

# 3. Generate videos
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20

# 4. Create presentation materials (통합됨)  
python scripts/utilities/generate_all_materials.py
```

### Core Script Categories
```bash
# Experiments (scripts/experiments/)
python scripts/experiments/run_experiment.py              # Main comprehensive experiment
python scripts/experiments/simple_training.py            # Quick training test
python scripts/experiments/run_all_experiments.py        # All experiments

# Video Generation (scripts/video/)
python scripts/video/core/render_learning_video.py --sample-data --all
python scripts/video/core/create_realtime_combined_videos.py --all
python scripts/video/comparison/create_comparison_video.py --auto
python scripts/video/comparison/create_success_failure_videos.py

# Utilities (scripts/utilities/)
python scripts/utilities/generate_all_materials.py          # 통합된 프레젠테이션 자료 생성
python scripts/utilities/organize_reports.py               # 리포트 정리
```

## Architecture Overview

### Core Components

**Agents** (`src/agents/`):
- `dqn_agent.py`: DQN with implicit deterministic policy (argmax over Q-values)
- `ddpg_agent.py`: DDPG with explicit deterministic policy (direct actor output)  
- `discretized_dqn_agent.py`: **Core Innovation** - DQN adapted for continuous action spaces

**Environments** (`src/environments/`):
- `continuous_cartpole.py`: **Key Innovation** - Continuous CartPole for fair DQN vs DDPG comparison
- `wrappers.py`: Standard environment wrappers
- `video_wrappers.py`: Video recording integration

**Video System** (`src/core/` and `src/visualization/`):
- `video_pipeline.py`: Learning process visualization
- `video_manager.py`: Dual-quality recording system
- `dual_recorder.py`: Simultaneous low/high quality recording
- Modular visualization system with automated chart generation

### Key Design Patterns

1. **Same-Environment Comparison**: ContinuousCartPole + DiscretizedDQN enables fair algorithm comparison
2. **Dual Video System**: Real-time recording during training + post-training visualization pipeline
3. **Configuration-Driven**: YAML configs for algorithms, video settings, and pipeline parameters
4. **Modular Visualization**: 90% code duplication eliminated through systematic refactoring

## Configuration System

**YAML-based configuration** with command-line overrides:

- `configs/dqn_config.yaml`: DQN hyperparameters  
- `configs/ddpg_config.yaml`: DDPG hyperparameters
- `configs/video_config.yaml`: Video generation settings
- `configs/pipeline_config.yaml`: Visualization pipeline settings

## Key Output Directories

```
output/visualization/          # All generated charts, videos, presentations
videos/                       # Training recordings and generated videos  
results/                      # JSON experiment results
docs/                        # Comprehensive Korean/English documentation
```

## Core Innovation Summary

**Fair Algorithm Comparison** through:

1. **ContinuousCartPole Environment**: CartPole physics with continuous action space
2. **DiscretizedDQN Agent**: DQN adapted for continuous actions via discretization  
3. **Same Environment Comparison**: Both algorithms tested in identical conditions
4. **Key Finding**: DQN outperforms DDPG by 13.2x in continuous environment

**Environment compatibility matters more than theoretical algorithm design.**

## Development Patterns

### Configuration Management
- Edit YAML files in `configs/` for hyperparameters
- Use command-line arguments to override config values
- Video quality presets available in `configs/video_config.yaml`

### Testing and Debugging
- Use `tests/simple_demo.py` for quick algorithm verification
- Use `tests/detailed_test.py` for comprehensive analysis
- Check `videos/temp/` for intermediate video processing files

### Adding New Features
- **New Environments**: Follow ContinuousCartPole pattern in `src/environments/`
- **Video Features**: Extend `VideoRenderingPipeline` in `src/core/video_pipeline.py`
- **New Algorithms**: Follow DiscretizedDQN pattern for action space adaptation

### File Organization
- Generated files (videos, models, results) are gitignored
- Only source code, configs, and small JSON results are tracked
- Korean documentation in `docs/` organized by category
- All presentation materials auto-generated with `scripts/utilities/generate_all_materials.py` (통합됨)