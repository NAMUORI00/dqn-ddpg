# 비디오 생성 스크립트 (Video Generation)

DQN vs DDPG 프로젝트의 실험 결과를 시각적으로 표현하는 다양한 비디오 생성 도구들입니다.

## 📁 디렉토리 구조

```
video/
├── core/              # 핵심 비디오 생성 기능
│   ├── render_learning_video.py           # 메인 학습 과정 비디오 파이프라인
│   └── create_realtime_combined_videos.py # 🏆 최신 혁신: 2x2 실시간 비디오
│
├── comparison/        # 비교 분석 비디오
│   ├── create_comparison_video.py         # 나란히 알고리즘 비교
│   ├── create_success_failure_videos.py   # 성공/실패 대비 영상
│   └── create_synchronized_training_video.py # 학습과 동기화된 게임플레이
│
└── specialized/       # 특수 목적 비디오
    ├── create_comprehensive_visualization.py    # 통합 시각화
    ├── create_fast_synchronized_video.py        # 최적화된 동기화
    ├── create_simple_continuous_cartpole_viz.py # 간단한 시각화
    └── generate_continuous_cartpole_viz.py      # JSON 기반 시각화
```

## 🏆 핵심 비디오 생성 (core/)

### 🌟 `create_realtime_combined_videos.py` - 최신 혁신
**프로젝트의 가장 임팩트 있는 비디오 생성 도구**

```bash
# 모든 환경에 대한 2x2 실시간 비디오 생성
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20

# CartPole 환경만
python scripts/video/core/create_realtime_combined_videos.py --cartpole --duration 15

# Pendulum 환경만  
python scripts/video/core/create_realtime_combined_videos.py --pendulum --duration 15
```

**특징:**
- **2x2 레이아웃**: 학습 그래프 + 실제 게임플레이 동시 표시
- **실시간 동기화**: 학습 진행도와 게임플레이가 완벽히 동기화
- **환경별 최적화**: CartPole(500에피소드), Pendulum(300에피소드) 맞춤 설정
- **자동 비디오 선택**: 학습 진행도에 따라 성공/실패 비디오 자동 선택

**출력 레이아웃:**
```
┌─────────────┬─────────────┐
│ DQN 학습그래프 │ DDPG 학습그래프│
├─────────────┼─────────────┤  
│ DQN 게임플레이 │ DDPG 게임플레이│
└─────────────┴─────────────┘
```

---

### 📊 `render_learning_video.py` - 메인 비디오 파이프라인
종합적인 학습 과정 시각화를 위한 메인 파이프라인

```bash
# 샘플 데이터로 빠른 테스트
python scripts/video/core/render_learning_video.py --sample-data --learning-only --duration 30

# 실제 학습 결과 사용
python scripts/video/core/render_learning_video.py --dqn-results results/dqn_results.json --ddpg-results results/ddpg_results.json

# 완전한 교육용 비디오 (intro + 학습 + 비교 + outro)
python scripts/video/core/render_learning_video.py --sample-data --all
```

**기능:**
- 학습 곡선 애니메이션
- 성능 메트릭 실시간 표시
- 알고리즘 비교 섹션
- 교육적 설명 텍스트 포함

## 🔄 비교 분석 비디오 (comparison/)

### ⚔️ `create_comparison_video.py` - 나란히 알고리즘 비교
두 알고리즘의 게임플레이를 직접 비교하는 비디오

```bash
# 자동 비교 비디오 생성
python scripts/video/comparison/create_comparison_video.py --auto

# 상세 비교 (성능 지표 포함)
python scripts/video/comparison/create_comparison_video.py --detailed

# 특정 에피소드 비교
python scripts/video/comparison/create_comparison_video.py --episodes 100,200,300
```

**특징:**
- 좌우 분할 화면
- 성능 지표 오버레이
- 에피소드별 성과 비교
- 실시간 점수 표시

---

### 🎯 `create_success_failure_videos.py` - 성공/실패 대비
환경 호환성 메시지를 강조하는 대비 비디오

```bash
# 모든 대비 비디오 생성
python scripts/video/comparison/create_success_failure_videos.py

# 특정 환경만
python scripts/video/comparison/create_success_failure_videos.py --environment cartpole
```

**생성되는 비디오:**
- `cartpole_dqn_success.mp4` - CartPole에서 DQN 성공
- `cartpole_ddpg_failure.mp4` - CartPole에서 DDPG 실패  
- `pendulum_ddpg_success.mp4` - Pendulum에서 DDPG 성공
- `pendulum_dqn_failure.mp4` - Pendulum에서 DQN 실패
- `four_way_comparison.mp4` - 4개 모두 한 화면에

---

### 🔄 `create_synchronized_training_video.py` - 동기화된 학습
실제 학습 과정과 동기화된 게임플레이 비디오

```bash
# 동기화된 학습 비디오
python scripts/video/comparison/create_synchronized_training_video.py

# 빠른 버전 (압축)
python scripts/video/comparison/create_synchronized_training_video.py --fast --duration 60
```

**특징:**
- 학습 진행률과 게임플레이 동기화
- 에피소드별 성능 변화 시각화
- 학습 곡선과 실제 행동의 연관성 표시

## 🎨 특수 목적 비디오 (specialized/)

### 📈 `create_comprehensive_visualization.py` - 통합 시각화
학습 그래프와 게임플레이를 통합한 종합 시각화

```bash
# 종합 시각화 생성
python scripts/video/specialized/create_comprehensive_visualization.py

# DQN만
python scripts/video/specialized/create_comprehensive_visualization.py --algorithm dqn

# 고품질 버전
python scripts/video/specialized/create_comprehensive_visualization.py --quality high
```

**포함 요소:**
- 실시간 학습 그래프
- 게임플레이 영상
- 성능 통계
- 알고리즘 정보 패널

---

### ⚡ `create_fast_synchronized_video.py` - 최적화된 동기화
성능 최적화된 빠른 동기화 비디오

```bash
# 빠른 동기화 비디오
python scripts/video/specialized/create_fast_synchronized_video.py --all --duration 20

# 메모리 절약 모드
python scripts/video/specialized/create_fast_synchronized_video.py --low-memory
```

**최적화 특징:**
- 메모리 효율적 처리
- 빠른 렌더링
- 압축된 출력
- 배치 처리 지원

---

### 📊 `create_simple_continuous_cartpole_viz.py` - 간단한 시각화
ContinuousCartPole 환경 전용 간단한 비교 차트

```bash
# 간단한 비교 차트
python scripts/video/specialized/create_simple_continuous_cartpole_viz.py
```

**출력:**
- 성능 비교 막대 그래프
- 하드코딩된 샘플 데이터 사용
- 빠른 생성 (1-2분)

---

### 📄 `generate_continuous_cartpole_viz.py` - JSON 기반 시각화
기존 실험 결과 JSON 파일을 사용한 시각화

```bash
# JSON 결과 기반 시각화
python scripts/video/specialized/generate_continuous_cartpole_viz.py

# 특정 결과 파일 지정
python scripts/video/specialized/generate_continuous_cartpole_viz.py --results-file custom_results.json
```

## 🎯 사용 시나리오별 가이드

### 🎓 교육/발표용 (가장 임팩트 있는 순서)

```bash
# 1. 🏆 2x2 실시간 비디오 (필수)
python scripts/video/core/create_realtime_combined_videos.py --all --duration 15

# 2. 성공/실패 대비 (환경 호환성 강조)
python scripts/video/comparison/create_success_failure_videos.py

# 3. 나란히 비교 (직접적 성능 차이)
python scripts/video/comparison/create_comparison_video.py --auto
```

### 🔬 연구/분석용

```bash
# 1. 상세한 학습 과정 분석
python scripts/video/core/render_learning_video.py --dqn-results results/dqn_results.json --ddpg-results results/ddpg_results.json

# 2. 동기화된 학습 과정
python scripts/video/comparison/create_synchronized_training_video.py

# 3. 종합 통계 시각화
python scripts/video/specialized/create_comprehensive_visualization.py --quality high
```

### ⚡ 빠른 데모용

```bash
# 1. 간단한 비교 차트
python scripts/video/specialized/create_simple_continuous_cartpole_viz.py

# 2. 샘플 데이터 비디오
python scripts/video/core/render_learning_video.py --sample-data --learning-only --duration 30

# 3. 빠른 동기화
python scripts/video/specialized/create_fast_synchronized_video.py --duration 10
```

## ⚙️ 공통 설정 옵션

### 품질 설정
- `--quality low/medium/high` - 비디오 품질
- `--resolution 720p/1080p/4k` - 해상도
- `--fps 30/60` - 프레임률

### 기간 설정
- `--duration N` - 비디오 길이 (초)
- `--episodes N` - 표시할 에피소드 수
- `--sample-rate N` - 샘플링 간격

### 출력 설정
- `--output-dir DIR` - 출력 디렉토리
- `--filename NAME` - 파일명 지정
- `--format mp4/avi` - 출력 형식

## 📊 출력 구조

### 비디오 파일 위치
```
output/visualization/videos/mp4/
├── learning_process/
│   ├── dqn_learning_process.mp4
│   ├── ddpg_learning_process.mp4
│   └── combined_learning_process.mp4
│
├── comparisons/
│   ├── dqn_vs_ddpg_comparison.mp4
│   ├── success_failure_contrast.mp4
│   └── four_way_comparison.mp4
│
├── realtime_monitoring/
│   ├── cartpole_realtime_comparison.mp4
│   └── pendulum_realtime_comparison.mp4
│
└── presentations/
    ├── educational_overview.mp4
    └── research_summary.mp4
```

### 임시 파일
```
videos/temp/
├── frames/           # 개별 프레임들
├── audio/           # 오디오 트랙 (있는 경우)
└── processing/      # 처리 중인 파일들
```

## 🔧 고급 사용법

### 배치 처리
```bash
# 모든 비디오 타입 생성
for script in scripts/video/core/*.py; do
    python "$script" --quality high
done
```

### 커스텀 설정
```python
# custom_video_config.yaml
video:
  fps: 60
  resolution: [1920, 1080]
  codec: 'h264'
  quality: 'high'
  
# 사용법
python scripts/video/core/render_learning_video.py --config custom_video_config.yaml
```

### 병렬 처리
```bash
# 여러 비디오 동시 생성
python scripts/video/core/create_realtime_combined_videos.py --cartpole &
python scripts/video/core/create_realtime_combined_videos.py --pendulum &
wait  # 모든 프로세스 완료 대기
```

## 🏆 추천 워크플로

### 완전한 비디오 세트 생성
```bash
#!/bin/bash
# 1. 실시간 2x2 비디오 (핵심)
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20

# 2. 성공/실패 대비 (교육적)
python scripts/video/comparison/create_success_failure_videos.py

# 3. 상세 학습 과정 (분석용)
python scripts/video/core/render_learning_video.py --sample-data --all

# 4. 나란히 비교 (명확한 차이)
python scripts/video/comparison/create_comparison_video.py --auto

echo "모든 핵심 비디오 생성 완료!"
```

이 비디오 생성 시스템을 통해 DQN vs DDPG 프로젝트의 연구 성과를 강력하고 설득력 있게 시각화할 수 있습니다.