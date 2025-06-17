# Scripts Directory

이 디렉토리는 DQN vs DDPG 프로젝트의 모든 실행 스크립트들을 기능별로 체계적으로 관리합니다.

## 📁 디렉토리 구조

```
scripts/
├── experiments/        # 실험 실행 스크립트들
├── video/             # 비디오 생성 스크립트들
│   ├── core/          # 핵심 비디오 생성 기능
│   ├── comparison/    # 비교 분석 비디오
│   └── specialized/   # 특수 목적 비디오
└── utilities/         # 유틸리티 및 관리 도구들
```

## 🎯 카테고리별 설명

### 🧪 **experiments/** - 실험 실행 스크립트
DQN과 DDPG 알고리즘의 학습 및 비교 실험을 수행하는 스크립트들

**주요 스크립트:**
- `run_experiment.py` - 메인 종합 실험 (비디오 녹화 포함)
- `run_all_experiments.py` - 모든 실험 자동 실행
- `run_same_env_experiment.py` - 동일환경 비교 (핵심 발견)
- `simple_training.py` - 빠른 테스트용 간단 학습

### 🎬 **video/** - 비디오 생성 스크립트
실험 결과를 시각적으로 표현하는 다양한 비디오 생성 도구들

#### **video/core/** - 핵심 비디오 생성
- `render_learning_video.py` - 메인 학습 과정 비디오 파이프라인
- `create_realtime_combined_videos.py` - 🏆 **최신 혁신**: 2x2 실시간 학습+게임플레이

#### **video/comparison/** - 비교 분석 비디오
- `create_comparison_video.py` - 나란히 알고리즘 비교
- `create_success_failure_videos.py` - 성공/실패 대비 영상
- `create_synchronized_training_video.py` - 학습과 동기화된 게임플레이

#### **video/specialized/** - 특수 목적 비디오
- `create_comprehensive_visualization.py` - 통합 시각화 (그래프+게임플레이)
- `create_fast_synchronized_video.py` - 최적화된 동기화 비디오
- `create_simple_continuous_cartpole_viz.py` - 간단한 ContinuousCartPole 시각화
- `generate_continuous_cartpole_viz.py` - JSON 결과 기반 시각화

### 🔧 **utilities/** - 유틸리티 및 관리 도구
프로젝트 관리, 검증, 문서화를 위한 도구들

- `generate_presentation_materials.py` - 종합 프레젠테이션 자료 생성
- `check_presentation_materials.py` - 생성된 자료 검증
- `organize_reports.py` - 리포트 정리 및 인덱싱
- `test_visualization_refactor.py` - 시각화 시스템 테스트

## 🚀 빠른 시작 가이드

### 기본 실험 실행
```bash
# 메인 종합 실험 (권장)
python scripts/experiments/run_experiment.py

# 빠른 테스트
python scripts/experiments/simple_training.py

# 모든 실험 자동 실행
python scripts/experiments/run_all_experiments.py
```

### 비디오 생성
```bash
# 최신 2x2 실시간 비디오 (권장)
python scripts/video/core/create_realtime_combined_videos.py --all --duration 20

# 학습 과정 비디오
python scripts/video/core/render_learning_video.py --sample-data --all

# 알고리즘 비교 비디오
python scripts/video/comparison/create_comparison_video.py --auto
```

### 프레젠테이션 자료 생성
```bash
# 모든 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py

# 생성된 자료 검증
python scripts/utilities/check_presentation_materials.py
```

## 📊 사용 우선순위

### 🏆 **핵심 스크립트 (필수)**
1. **`experiments/run_experiment.py`** - 메인 실험 파이프라인
2. **`video/core/create_realtime_combined_videos.py`** - 최신 비디오 혁신
3. **`utilities/generate_presentation_materials.py`** - 완전한 자료 생성
4. **`experiments/run_same_env_experiment.py`** - 핵심 발견 문서화

### 🎯 **특화 스크립트 (목적별)**
- **빠른 데모**: `experiments/simple_training.py`
- **비교 분석**: `video/comparison/create_comparison_video.py`
- **학습 과정**: `video/core/render_learning_video.py`
- **자료 검증**: `utilities/check_presentation_materials.py`

### 🔬 **고급 분석 (연구용)**
- **종합 시각화**: `video/specialized/create_comprehensive_visualization.py`
- **동기화 비디오**: `video/comparison/create_synchronized_training_video.py`
- **성공/실패 대비**: `video/comparison/create_success_failure_videos.py`

## 🎓 교육적 활용

### 강의/발표용
```bash
# 완전한 프레젠테이션 패키지 생성
python scripts/utilities/generate_presentation_materials.py

# 2x2 실시간 비교 비디오 (가장 임팩트 있음)
python scripts/video/core/create_realtime_combined_videos.py --all --duration 15
```

### 연구/분석용
```bash
# 메인 실험 + 모든 비디오 생성
python scripts/experiments/run_experiment.py --save-models --results-dir results

# 상세 비교 분석
python scripts/video/comparison/create_comparison_video.py --detailed
```

### 빠른 데모용
```bash
# 15초 빠른 데모
python scripts/experiments/simple_training.py

# 간단 시각화
python scripts/video/specialized/create_simple_continuous_cartpole_viz.py
```

## 🔗 스크립트 간 의존성

### 실험 결과 → 비디오 생성
1. `experiments/run_experiment.py` → 결과 파일 생성
2. `video/core/render_learning_video.py` → 결과 파일 사용하여 비디오 생성

### 데이터 생성 → 프레젠테이션
1. 모든 실험 스크립트 → JSON/이미지 결과 생성
2. `utilities/generate_presentation_materials.py` → 모든 결과를 프레젠테이션 자료로 통합

### 검증 워크플로
1. 실험 실행 → 결과 생성
2. `utilities/check_presentation_materials.py` → 결과 검증
3. 문제 발견 시 재실행 또는 수정

## ⚙️ 설정 및 커스터마이징

각 스크립트는 다음과 같은 설정 방법을 지원:
- **명령행 인수**: `--duration`, `--quality`, `--save-models` 등
- **설정 파일**: `configs/` 디렉토리의 YAML 파일들
- **환경 변수**: 경로 및 기본값 설정

자세한 설정 방법은 각 카테고리별 README 파일을 참조하세요.

---

이 구조화된 스크립트 시스템을 통해 DQN vs DDPG 프로젝트의 모든 기능을 체계적이고 효율적으로 활용할 수 있습니다.