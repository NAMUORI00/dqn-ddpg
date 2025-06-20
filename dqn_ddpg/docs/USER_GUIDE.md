# 🎯 DQN vs DDPG 완전 사용자 가이드

> **DQN과 DDPG 알고리즘의 결정적 정책 비교 프로젝트 - 설치부터 고급 활용까지 완전 가이드**

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [빠른 시작](#2-빠른-시작)
3. [프로젝트 이해](#3-프로젝트-이해)
4. [사용법 가이드](#4-사용법-가이드)
5. [비디오 시스템](#5-비디오-시스템)
6. [실험 및 분석](#6-실험-및-분석)
7. [시각화 시스템](#7-시각화-시스템)
8. [고급 활용](#8-고급-활용)
9. [문제 해결](#9-문제-해결)

---

## 1. 📋 프로젝트 개요

### 1.1 핵심 목표

DQN(Deep Q-Network)과 DDPG(Deep Deterministic Policy Gradient) 알고리즘의 **결정적(deterministic) 정책** 특성을 비교 분석하는 완전한 교육용 강화학습 프로젝트입니다.

🎯 **핵심 목표**: 암묵적 vs 명시적 결정적 정책의 차이점을 코드와 영상으로 명확히 보여주기

### 1.2 주요 특징

- **DQN의 암묵적 결정적 정책** vs **DDPG의 명시적 결정적 정책** 비교
- 이산적 행동 공간과 연속적 행동 공간에서의 정책 구현 차이 분석
- 🎬 **학습 과정 자동 시각화**: 전문적인 교육용 비디오 생성
- 교육적 시각화를 통한 알고리즘 이해 증진

### 1.3 핵심 차이점

| 특성 | DQN | DDPG |
|------|-----|------|
| **정책 유형** | 암묵적 결정적 (Q-값 argmax) | 명시적 결정적 (액터 출력) |
| **행동 공간** | 이산적 (discrete) | 연속적 (continuous) |
| **네트워크** | Q-네트워크 | 액터-크리틱 |
| **탐험 방식** | ε-greedy | 가우시안 노이즈 |

---

## 2. 🚀 빠른 시작

### 2.1 필수 준비 사항

**시스템 요구사항:**
- Windows 10/11, macOS, Linux
- Python 3.8 이상
- Git

**권장 환경:**
- Conda 또는 Python 가상환경
- 8GB+ RAM
- GPU (선택사항, 학습 속도 향상)

### 2.2 30초 설치

```bash
# 1. 프로젝트 다운로드
git clone <repository-url>
cd dqn,ddpg

# 2. 가상환경 생성 (권장)
conda create -n dqn_ddpg python=3.11
conda activate dqn_ddpg

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 테스트 실행
python tests/simple_demo.py
```

### 2.3 첫 실행 (3가지 방법)

#### 🎬 방법 1: 비디오 데모 (가장 빠름)
```bash
python scripts/video/core/render_learning_video.py --sample-data --learning-only --duration 30
# 결과: output/visualization/videos/mp4/learning_process/ 에 생성
```

#### 🤖 방법 2: 결정적 정책 시연
```bash
python tests/simple_demo.py
# 결과: DQN과 DDPG의 정책 차이 콘솔 출력
```

#### 🏃 방법 3: 기본 학습 실행
```bash
python scripts/experiments/simple_training.py
# 결과: 간단한 학습 및 결과 분석
```

---

## 3. 🧠 프로젝트 이해

### 3.1 핵심 개념

이 프로젝트는 **결정적 정책(Deterministic Policy)**을 구현하는 두 가지 방법을 비교합니다:

#### DQN의 암묵적 결정적 정책
```python
# Q-값을 계산한 후 최댓값 선택
q_values = q_network(state)
action = q_values.argmax()  # 결정적이지만 간접적
```

#### DDPG의 명시적 결정적 정책
```python
# 액터가 직접 행동 출력
action = actor_network(state)  # 결정적이고 직접적
```

### 3.2 교육적 가치

1. **정책 표현의 차이**:
   - DQN: "어떤 행동이 가장 좋은가?" → argmax
   - DDPG: "이 상황에서 어떤 행동을 할까?" → 직접 출력

2. **행동 공간의 영향**:
   - 이산적: 모든 가능한 행동을 열거 가능
   - 연속적: 무한한 가능성, 직접 생성 필요

3. **탐험 전략의 차이**:
   - 이산적: 완전히 다른 행동 선택
   - 연속적: 기본 행동에 노이즈 추가

### 3.3 프로젝트 구조

```
dqn,ddpg/
├── src/                          # 핵심 구현
│   ├── agents/                   # DQN, DDPG 에이전트
│   ├── networks/                 # 신경망 모델들
│   ├── core/                     # 공통 컴포넌트 + 비디오 파이프라인
│   ├── environments/             # 환경 래퍼
│   └── visualization/            # 모듈화된 시각화 시스템
├── scripts/                      # 실행 스크립트들
│   ├── experiments/              # 실험 실행 스크립트 (4개)
│   ├── video/                    # 비디오 생성 스크립트 (9개)
│   └── utilities/                # 관리 도구 (4개)
├── experiments/                  # 실험 및 시각화 (기존)
├── configs/                      # 설정 파일
├── docs/                         # 문서
├── output/                       # 새로운 출력 구조
│   └── visualization/            # 확장자별 자동 분류
└── tests/                        # 테스트 스크립트
```

---

## 4. 📖 사용법 가이드

### 4.1 기본 사용법

#### 간단한 시연
```bash
python tests/simple_demo.py
```
- 각 알고리즘의 결정적 정책 특성을 간단히 확인

#### 상세 테스트
```bash
python tests/detailed_test.py
```
- 행동 일관성, Q-값 분석, 탐험 영향 등 심층 분석
- 결과: `output/visualization/images/png/charts/deterministic_policy_analysis.png`

#### 전체 실험 실행
```bash
python scripts/experiments/run_experiment.py --save-models
```
- 전체 학습 및 평가 파이프라인
- 결과: `results/` 및 `output/visualization/` 디렉토리에 저장

### 4.2 스크립트 카테고리별 사용법

#### 🧪 실험 실행 (`scripts/experiments/`)

```bash
# 메인 종합 실험 (권장)
python scripts/experiments/run_experiment.py

# 빠른 테스트
python scripts/experiments/simple_training.py

# 모든 실험 자동 실행
python scripts/experiments/run_all_experiments.py

# 핵심 발견 요약 (13.2x DQN 우위)
python scripts/experiments/run_same_env_experiment.py
```

#### 🎬 비디오 생성 (`scripts/video/`)

```bash
# 🏆 최신 2x2 실시간 비디오 (가장 임팩트 있음)
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20

# 학습 과정 비디오
python scripts/video/core/render_learning_video.py --sample-data --all

# 알고리즘 비교
python scripts/video/comparison/create_comparison_video.py --auto

# 성공/실패 대비
python scripts/video/comparison/create_success_failure_videos.py
```

#### 🔧 유틸리티 (`scripts/utilities/`)

```bash
# 모든 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py

# 생성된 자료 검증
python scripts/utilities/check_presentation_materials.py

# 리포트 정리
python scripts/utilities/organize_reports.py

# 시각화 시스템 테스트
python scripts/utilities/test_visualization_refactor.py
```

---

## 5. 🎬 비디오 시스템

### 5.1 비디오 파이프라인 특징

- ✅ **FFmpeg 불필요**: OpenCV 백업 시스템으로 안정적 동작
- ✅ **샘플 데이터 포함**: 실제 학습 없이도 즉시 데모 생성 가능
- ✅ **다양한 품질**: 미리보기부터 프레젠테이션용 고화질까지
- ✅ **교육 최적화**: 알고리즘 비교와 학습 과정 설명 포함

### 5.2 빠른 데모 비디오 생성

```bash
# 15초 데모 비디오 (추천)
python scripts/video/core/create_realtime_combined_videos.py --cartpole --duration 15

# 30초 HD 비디오
python scripts/video/core/create_realtime_combined_videos.py --pendulum --duration 30

# 모든 환경 20초 비디오
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20
```

### 5.3 전체 학습 과정 시각화

```bash
# 샘플 데이터로 학습 애니메이션 생성
python scripts/video/core/render_learning_video.py --sample-data --learning-only --duration 30

# 완전한 요약 비디오 (인트로 + 학습 + 비교 + 아웃트로)
python scripts/video/core/render_learning_video.py --sample-data --all

# 실제 학습 결과 사용
python scripts/video/core/render_learning_video.py --dqn-results results/dqn_results.json --ddpg-results results/ddpg_results.json
```

### 5.4 특수 비디오 생성

```bash
# 종합 시각화 (그래프 + 게임플레이)
python scripts/video/specialized/create_comprehensive_visualization.py

# 빠른 동기화 비디오
python scripts/video/specialized/create_fast_synchronized_video.py --all --duration 20

# 간단한 비교 차트
python scripts/video/specialized/create_simple_continuous_cartpole_viz.py
```

---

## 6. 🔬 실험 및 분석

### 6.1 핵심 실험 결과

이 프로젝트의 주요 발견:

1. **동일환경 비교**: DQN이 DDPG보다 **13.2배 우수한 성능** (ContinuousCartPole 환경)
2. **환경 호환성**: 환경 호환성 > 알고리즘 유형 원칙 확립
3. **결정적 정책**: 양 알고리즘 모두 완벽한 결정성 (1.0) 달성

### 6.2 실험 시나리오

#### 📊 연구/분석 목적
```bash
# 1. 완전한 실험 실행
python scripts/experiments/run_experiment.py --save-models --high-quality

# 2. 결과 검증
python scripts/utilities/check_presentation_materials.py

# 3. 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py
```

#### 🎓 교육/발표 목적
```bash
# 1. 핵심 발견 요약
python scripts/experiments/run_same_env_experiment.py

# 2. 빠른 데모
python scripts/experiments/simple_training.py --episodes 50 --visualize

# 3. 임팩트 있는 비디오 생성
python scripts/video/core/create_realtime_combined_videos.py --all --duration 15
```

#### 🔧 개발/테스트 목적
```bash
# 1. 빠른 기능 테스트
python scripts/experiments/simple_training.py

# 2. 전체 시스템 검증
python scripts/experiments/run_all_experiments.py

# 3. 시각화 시스템 테스트
python scripts/utilities/test_visualization_refactor.py
```

### 6.3 설정 파일

- `configs/dqn_config.yaml`: DQN 하이퍼파라미터
- `configs/ddpg_config.yaml`: DDPG 하이퍼파라미터
- `configs/video_config.yaml`: 비디오 생성 설정
- `configs/pipeline_config.yaml`: 시각화 파이프라인 설정

---

## 7. 📊 시각화 시스템

### 7.1 새로운 모듈화 시스템

프로젝트는 완전히 모듈화된 시각화 시스템을 제공합니다:

```python
# 기존: 인라인 matplotlib 코드 (50+ 줄)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dqn_rewards, label='DQN', color='blue')
# ... 30+ 줄의 스타일링 코드 ...

# 새로운 시스템: 모듈화된 클래스 (3줄)
from src.visualization.charts.comparison import ComparisonChartVisualizer
with ComparisonChartVisualizer() as viz:
    viz.create_performance_comparison(dqn_data, ddpg_data, "comparison_results.png")
```

### 7.2 자동 출력 구조

새로운 확장자 기반 출력 시스템:

```
output/visualization/
├── images/
│   ├── png/charts/        # PNG 차트 파일
│   ├── svg/diagrams/      # SVG 다이어그램
│   └── pdf/plots/         # PDF 플롯
├── videos/
│   ├── mp4/comparisons/   # MP4 비교 비디오
│   └── gif/animations/    # GIF 애니메이션
├── data/
│   ├── json/experiments/  # JSON 실험 데이터
│   └── csv/summaries/     # CSV 요약 데이터
└── documents/
    ├── md/reports/        # 마크다운 리포트
    └── html/presentations/ # HTML 프레젠테이션
```

### 7.3 시각화 모듈 구조

```
src/visualization/
├── core/              # 기본 클래스 및 공통 기능
│   ├── base.py        # BaseVisualizer 클래스
│   ├── config.py      # 설정 관리
│   └── utils.py       # 유틸리티 함수
├── charts/            # 차트 생성 모듈들
│   ├── comparison.py  # 비교 차트
│   ├── learning_curves.py # 학습 곡선
│   ├── metrics.py     # 성능 메트릭
│   └── policy_analysis.py # 정책 분석
├── video/             # 비디오 생성 시스템
├── presentation/      # 프레젠테이션 자료 생성
└── realtime/          # 실시간 모니터링
```

---

## 8. 🚀 고급 활용

### 8.1 커스텀 실험 설정

#### 하이퍼파라미터 조정
```yaml
# configs/custom_dqn.yaml
algorithm: "DQN"
learning_rate: 0.0005
batch_size: 64
episodes: 1000
epsilon_decay: 0.995
```

```bash
python scripts/experiments/run_experiment.py --config configs/custom_dqn.yaml
```

#### 환경 변수 설정
```bash
export DQN_EPISODES=500
export DDPG_EPISODES=300
export VIDEO_QUALITY=high
export RESULTS_DIR=custom_results
```

### 8.2 새로운 시각화 타입 추가

```python
from src.visualization.core.base import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    def create_visualization(self, data, **kwargs):
        fig, ax = self.create_figure(title="Custom Analysis")
        # 커스텀 시각화 로직
        return self.save_figure(fig, "custom_analysis.png", "charts")
```

### 8.3 배치 처리

```bash
# 모든 비디오 타입 생성
for script in scripts/video/core/*.py; do
    python "$script" --quality high
done

# 여러 설정으로 실험 실행
for config in configs/dqn_*.yaml; do
    python scripts/experiments/run_experiment.py --config "$config"
done
```

### 8.4 API 활용

```python
# 프로그래매틱 사용
from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.visualization.charts.comparison import ComparisonChartVisualizer

# 에이전트 생성
dqn = DQNAgent(config="configs/dqn_config.yaml")
ddpg = DDPGAgent(config="configs/ddpg_config.yaml")

# 결과 비교
with ComparisonChartVisualizer() as viz:
    viz.create_performance_comparison(dqn_results, ddpg_results)
```

---

## 9. 🔍 문제 해결

### 9.1 일반적인 문제들

#### 의존성 문제
```bash
# 패키지 확인
python -c "import torch, gymnasium, matplotlib; print('Dependencies OK')"

# 최신 버전 설치
pip install --upgrade torch gymnasium matplotlib opencv-python
```

#### CUDA 메모리 부족
```bash
# 에피소드 수 줄이기
python scripts/experiments/simple_training.py --episodes 100

# CPU 모드로 실행
export CUDA_VISIBLE_DEVICES=""
```

#### 비디오 생성 실패
```bash
# OpenCV 없이 실행
python scripts/experiments/run_experiment.py --no-video

# 낮은 품질로 테스트
python scripts/video/core/render_learning_video.py --quality low
```

#### 한글 폰트 문제
시스템에 한글 폰트가 없는 경우 DejaVu Sans로 자동 대체됩니다. 별도 설치 필요 없음.

### 9.2 성능 최적화

#### 빠른 실행
- 에피소드 수 줄이기 (`--episodes 50`)
- 비디오 비활성화 (`--no-video`)
- 낮은 품질 설정 (`--quality low`)

#### 고품질 결과
- 충분한 에피소드 (`--episodes 1000`)
- 고품질 비디오 (`--quality high`)
- 모델 저장 (`--save-models`)

### 9.3 디버깅

#### 로그 확인
```bash
# 상세 로그
python scripts/experiments/run_experiment.py --verbose

# 특정 모듈 테스트
python -c "from src.visualization.charts.comparison import ComparisonChartVisualizer; print('OK')"
```

#### 테스트 실행
```bash
# 전체 시스템 테스트
python scripts/utilities/test_visualization_refactor.py --full-test

# 개별 기능 테스트
python tests/simple_demo.py
python tests/detailed_test.py
```

### 9.4 지원 및 문의

- **문서**: `docs/` 디렉토리의 상세 가이드들
- **예제**: `tests/` 디렉토리의 테스트 스크립트들
- **설정**: `configs/` 디렉토리의 YAML 파일들

---

## 📝 결론

이 사용자 가이드를 통해 DQN vs DDPG 프로젝트의 모든 기능을 효과적으로 활용할 수 있습니다. 

**시작하기 좋은 순서:**
1. **빠른 시작**: `python tests/simple_demo.py`
2. **비디오 데모**: `python scripts/video/core/create_realtime_combined_videos.py --cartpole --duration 15`
3. **기본 실험**: `python scripts/experiments/simple_training.py`
4. **고급 활용**: 설정 커스터마이징 및 새로운 실험

🎬 **새롭게 추가된 비디오 파이프라인**과 **모듈화된 시각화 시스템**을 통해 강화학습 알고리즘의 차이점을 시각적으로 이해하고, 교육 및 연구 목적으로 활용할 수 있는 고품질 자료를 생성할 수 있습니다.

## 📚 추가 자료

- **기술 문서**: `docs/DEVELOPER_GUIDE.md`
- **최종 리포트**: `docs/final_reports/FINAL_REPORT.md`
- **실험 결과**: `docs/experiment_reports/`
- **스크립트 가이드**: `scripts/README.md`