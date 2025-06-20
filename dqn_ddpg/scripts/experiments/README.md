# 실험 실행 스크립트 (Experiments)

DQN과 DDPG 알고리즘의 학습, 비교, 분석을 위한 실험 실행 스크립트들입니다.

## 📁 포함된 스크립트들

### 🏆 `run_experiment.py` - 메인 종합 실험
**가장 중요한 스크립트**로, 프로젝트의 핵심 기능을 모두 포함합니다.

```bash
# 기본 실행
python scripts/experiments/run_experiment.py

# 모델 저장 포함
python scripts/experiments/run_experiment.py --save-models --results-dir results

# 고품질 비디오 포함
python scripts/experiments/run_experiment.py --video-quality high
```

**주요 기능:**
- DQN vs DDPG 완전한 비교 실험
- 실시간 비디오 녹화 (dual-quality system)
- 성능 메트릭 수집 및 분석
- 자동 결과 저장 및 시각화
- 모델 체크포인트 저장

**출력물:**
- 학습된 모델 파일 (`models/`)
- 성능 데이터 JSON (`results/`)
- 학습 과정 비디오 (`videos/`)
- 비교 분석 차트 (`output/visualization/`)

---

### 🚀 `run_all_experiments.py` - 자동화된 실험 스위트
모든 구현된 실험을 순차적으로 실행하는 자동화 도구입니다.

```bash
# 모든 실험 자동 실행
python scripts/experiments/run_all_experiments.py

# 로그 파일 지정
python scripts/experiments/run_all_experiments.py --log-file experiment_log.txt
```

**실행되는 실험들:**
1. 기본 DQN vs DDPG 비교
2. 동일환경 비교 (ContinuousCartPole)
3. 결정적 정책 분석
4. 성능 메트릭 종합 분석
5. 비디오 생성 및 검증

**사용 사례:**
- CI/CD 파이프라인에서 전체 테스트
- 새로운 환경에서 모든 기능 검증
- 성능 회귀 테스트
- 완전한 결과 세트 생성

---

### 🎯 `run_same_env_experiment.py` - 핵심 발견 문서화
**프로젝트의 가장 중요한 발견**인 동일환경 비교 결과를 요약합니다.

```bash
# 핵심 발견 요약 실행
python scripts/experiments/run_same_env_experiment.py
```

**핵심 결과:**
- **DQN이 DDPG보다 13.2배 우수한 성능** (ContinuousCartPole 환경)
- 환경 호환성 > 알고리즘 유형 원칙 입증
- 기존 통념 ("DDPG가 연속 액션에서 우수") 반박

**출력물:**
- 핵심 발견 요약 JSON
- 통계적 유의성 분석
- 실험 메타데이터
- 재현 가능성 정보

---

### ⚡ `simple_training.py` - 빠른 테스트 파이프라인
개발 및 빠른 검증을 위한 간소화된 학습 스크립트입니다.

```bash
# 빠른 테스트 실행
python scripts/experiments/simple_training.py

# 에피소드 수 조정
python scripts/experiments/simple_training.py --episodes 100

# 기본 시각화 포함
python scripts/experiments/simple_training.py --visualize
```

**특징:**
- 빠른 실행 (5-10분)
- 기본적인 성능 비교
- 간단한 시각화
- 개발 중 빠른 검증용

**사용 사례:**
- 새로운 기능 테스트
- 설정 변경 검증
- 빠른 데모 생성
- 개발 중 디버깅

## 🎯 실행 시나리오별 가이드

### 📊 연구/분석 목적
```bash
# 1. 완전한 실험 실행
python scripts/experiments/run_experiment.py --save-models --high-quality

# 2. 결과 검증
python scripts/utilities/check_presentation_materials.py

# 3. 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py
```

### 🎓 교육/발표 목적
```bash
# 1. 핵심 발견 요약
python scripts/experiments/run_same_env_experiment.py

# 2. 빠른 데모
python scripts/experiments/simple_training.py --episodes 50 --visualize

# 3. 임팩트 있는 비디오 생성
python scripts/video/core/create_realtime_combined_videos.py --all --duration 15
```

### 🔧 개발/테스트 목적
```bash
# 1. 빠른 기능 테스트
python scripts/experiments/simple_training.py

# 2. 전체 시스템 검증
python scripts/experiments/run_all_experiments.py

# 3. 시각화 시스템 테스트
python scripts/utilities/test_visualization_refactor.py
```

### 🚀 프로덕션/배포 목적
```bash
# 1. 모든 실험 자동 실행
python scripts/experiments/run_all_experiments.py --log-file production_test.txt

# 2. 결과 검증 및 정리
python scripts/utilities/organize_reports.py

# 3. 최종 프레젠테이션 패키지 생성
python scripts/utilities/generate_presentation_materials.py --high-quality
```

## ⚙️ 설정 및 커스터마이징

### 공통 매개변수
- `--episodes`: 학습 에피소드 수
- `--save-models`: 모델 저장 여부
- `--results-dir`: 결과 저장 디렉토리
- `--video-quality`: 비디오 품질 (low/medium/high)
- `--visualize`: 기본 시각화 생성 여부

### 설정 파일 사용
각 스크립트는 `configs/` 디렉토리의 YAML 설정 파일을 사용:
- `dqn_config.yaml` - DQN 하이퍼파라미터
- `ddpg_config.yaml` - DDPG 하이퍼파라미터
- `video_recording.yaml` - 비디오 녹화 설정

### 환경 변수
```bash
export DQN_EPISODES=500
export DDPG_EPISODES=300
export VIDEO_QUALITY=high
export RESULTS_DIR=custom_results
```

## 📊 출력 구조

### 실험 결과 디렉토리
```
results/
├── dqn_results.json           # DQN 성능 데이터
├── ddpg_results.json          # DDPG 성능 데이터
├── comparison_summary.json    # 비교 분석 요약
└── experiment_metadata.json   # 실험 설정 및 메타데이터
```

### 모델 파일 (선택적)
```
models/
├── dqn_final.pth             # 최종 DQN 모델
├── ddpg_actor_final.pth      # 최종 DDPG Actor
├── ddpg_critic_final.pth     # 최종 DDPG Critic
└── checkpoints/              # 중간 체크포인트들
```

### 비디오 출력
```
videos/
├── dqn/                      # DQN 학습 과정
├── ddpg/                     # DDPG 학습 과정
└── comparison/               # 비교 분석 비디오
```

## 🔍 문제 해결

### 일반적인 오류들

**CUDA 메모리 부족:**
```bash
python scripts/experiments/simple_training.py --episodes 100  # 에피소드 수 줄이기
```

**비디오 생성 실패:**
```bash
# OpenCV 없이 실행
python scripts/experiments/run_experiment.py --no-video
```

**의존성 문제:**
```bash
# 필수 패키지만으로 실행
python scripts/experiments/simple_training.py --minimal
```

### 성능 최적화

**빠른 실행:**
- 에피소드 수 줄이기 (`--episodes 50`)
- 비디오 비활성화 (`--no-video`)
- 낮은 품질 설정 (`--video-quality low`)

**고품질 결과:**
- 충분한 에피소드 (`--episodes 1000`)
- 고품질 비디오 (`--video-quality high`)
- 모델 저장 (`--save-models`)

## 🏆 모범 사례

1. **점진적 접근**: `simple_training.py` → `run_experiment.py` → `run_all_experiments.py`
2. **결과 검증**: 각 실행 후 결과 확인 및 검증
3. **설정 문서화**: 사용된 설정을 실험 메타데이터에 기록
4. **재현 가능성**: 동일한 시드 및 설정 사용
5. **자원 관리**: GPU 메모리 및 디스크 공간 모니터링

이 실험 스크립트들을 통해 DQN vs DDPG 프로젝트의 모든 연구 발견을 체계적으로 재현하고 확장할 수 있습니다.