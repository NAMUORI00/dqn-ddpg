# DQN vs DDPG 결정적 정책 비교 프로젝트 ✨

DQN(Deep Q-Network)과 DDPG(Deep Deterministic Policy Gradient) 알고리즘의 **결정적(deterministic) 정책** 특성을 비교 분석하는 완전한 교육용 강화학습 프로젝트입니다.

> 🎯 **핵심 목표**: 암묵적 vs 명시적 결정적 정책의 차이점을 코드와 영상으로 명확히 보여주기

## 프로젝트 개요

### 주요 목표
- **DQN의 암묵적 결정적 정책** vs **DDPG의 명시적 결정적 정책** 비교
- 이산적 행동 공간과 연속적 행동 공간에서의 정책 구현 차이 분석
- 🎬 **학습 과정 자동 시각화**: 전문적인 교육용 비디오 생성
- 교육적 시각화를 통한 알고리즘 이해 증진

### 핵심 차이점
| 특성 | DQN | DDPG |
|------|-----|------|
| **정책 유형** | 암묵적 결정적 (Q-값 argmax) | 명시적 결정적 (액터 출력) |
| **행동 공간** | 이산적 (discrete) | 연속적 (continuous) |
| **네트워크** | Q-네트워크 | 액터-크리틱 |
| **탐험 방식** | ε-greedy | 가우시안 노이즈 |

## 설치 및 환경 설정

### 1. 환경 생성
```bash
conda create -n ddpg_dqn python=3.11
conda activate ddpg_dqn
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 🎬 새로운 기능: 비디오 파이프라인

### 빠른 데모 비디오 생성
```bash
# 15초 데모 비디오 (추천)
python quick_video_demo.py --duration 15 --output demo.mp4

# 30초 HD 비디오
python quick_video_demo.py --duration 30 --fps 30 --output hd_demo.mp4
```

### 전체 학습 과정 시각화
```bash
# 샘플 데이터로 학습 애니메이션 생성
python render_learning_video.py --sample-data --learning-only --duration 30

# 완전한 요약 비디오 (인트로 + 학습 + 비교 + 아웃트로)
python render_learning_video.py --sample-data --all

# 실제 학습 결과 사용
python render_learning_video.py --dqn-results results/dqn.json --ddpg-results results/ddpg.json
```

### 비디오 파이프라인 특징
- ✅ **FFmpeg 불필요**: OpenCV 백업 시스템으로 안정적 동작
- ✅ **샘플 데이터 포함**: 실제 학습 없이도 즉시 데모 생성 가능
- ✅ **다양한 품질**: 미리보기부터 프레젠테이션용 고화질까지
- ✅ **교육 최적화**: 알고리즘 비교와 학습 과정 설명 포함

## 사용법

### 1. 🎬 비디오 생성 (신규 추천!)
```bash
# 첫 사용자용 빠른 데모
python quick_video_demo.py --duration 15

# 전문적인 교육용 비디오
python render_learning_video.py --sample-data --all
```

### 2. 간단한 시연
```bash
python simple_demo.py
```
- 각 알고리즘의 결정적 정책 특성을 간단히 확인

### 3. 상세 테스트
```bash
python detailed_test.py
```
- 행동 일관성, Q-값 분석, 탐험 영향 등 심층 분석
- 결과: `results/deterministic_policy_analysis.png`

### 4. 전체 실험 실행
```bash
python run_experiment.py --save-models --results-dir results
```
- 전체 학습 및 평가 파이프라인
- 결과: `results/` 디렉토리에 모든 분석 결과 저장

## 프로젝트 구조

```
dqn,ddpg/
├── src/                          # 핵심 구현
│   ├── agents/                   # DQN, DDPG 에이전트
│   ├── networks/                 # 신경망 모델들
│   ├── core/                     # 공통 컴포넌트 + 🎬 비디오 파이프라인
│   └── environments/             # 환경 래퍼
├── experiments/                  # 실험 및 시각화
├── configs/                      # 설정 파일 + 🎬 파이프라인 설정
├── docs/                         # 📖 한글/영문 문서
├── videos/                       # 🎬 생성된 비디오 출력
├── 🎬 render_learning_video.py   # 전체 비디오 파이프라인
├── 🎬 quick_video_demo.py        # 빠른 데모 생성기
├── simple_demo.py               # 간단한 시연
├── detailed_test.py             # 상세 테스트
└── run_experiment.py            # 전체 실험
```

## 핵심 구현 특징

### DQN (암묵적 결정적 정책)
```python
# Q-값 계산 후 최대값의 인덱스 선택
q_values = q_network(state)
action = q_values.argmax()  # 결정적이지만 간접적
```

### DDPG (명시적 결정적 정책)
```python
# 액터가 직접 연속 행동 출력
action = actor_network(state)  # 결정적이고 직접적
```

## 실험 결과

실험을 실행하면 다음과 같은 분석 결과를 얻을 수 있습니다:

1. **학습 곡선 비교**: 두 알고리즘의 학습 진행 과정
2. **결정적 정책 분석**: 
   - DQN: Q-값 분포 및 행동 선택 패턴
   - DDPG: 액터 출력 분포 및 행동 일관성
3. **탐험 메커니즘 비교**: ε-greedy vs 가우시안 노이즈
4. **최종 성능 비교**: 각 환경에서의 성능 지표

## 교육적 가치

이 프로젝트는 다음을 명확히 보여줍니다:

1. **정책 표현의 차이**:
   - DQN: "어떤 행동이 가장 좋은가?" → argmax
   - DDPG: "이 상황에서 어떤 행동을 할까?" → 직접 출력

2. **행동 공간의 영향**:
   - 이산적: 모든 가능한 행동을 열거 가능
   - 연속적: 무한한 가능성, 직접 생성 필요

3. **탐험 전략의 차이**:
   - 이산적: 완전히 다른 행동 선택
   - 연속적: 기본 행동에 노이즈 추가

## 주요 파일 설명

### 🎬 비디오 파이프라인 (신규)
- `src/core/video_pipeline.py`: 메인 비디오 렌더링 파이프라인
- `render_learning_video.py`: 전체 기능 비디오 생성 스크립트
- `quick_video_demo.py`: 빠른 데모 비디오 생성기
- `configs/pipeline_config.yaml`: 비디오 파이프라인 설정

### 핵심 알고리즘
- `src/agents/dqn_agent.py`: DQN 구현, argmax 기반 행동 선택
- `src/agents/ddpg_agent.py`: DDPG 구현, 액터-크리틱 구조
- `src/core/noise.py`: 탐험을 위한 노이즈 프로세스
- `experiments/visualizations.py`: 결과 시각화 도구

### 문서
- `docs/VIDEO_PIPELINE_README.md`: 📖 영문 비디오 파이프라인 가이드
- `docs/비디오_렌더링_파이프라인_가이드.md`: 📖 한글 비디오 파이프라인 가이드

## 설정 파일

- `configs/dqn_config.yaml`: DQN 하이퍼파라미터
- `configs/ddpg_config.yaml`: DDPG 하이퍼파라미터
- `configs/pipeline_config.yaml`: 🎬 비디오 파이프라인 설정

설정을 수정하여 다양한 실험을 진행할 수 있습니다.

## 🎬 생성되는 비디오 예시

비디오 파이프라인을 통해 다음과 같은 교육용 콘텐츠를 자동 생성할 수 있습니다:

1. **학습 과정 애니메이션**: DQN과 DDPG의 실시간 학습 곡선
2. **알고리즘 비교**: 성능 메트릭과 수렴 속도 비교
3. **종합 요약 비디오**: 알고리즘 소개부터 결과 분석까지 완전한 스토리

### 빠른 시작 (1분 내 완료)
```bash
python quick_video_demo.py --duration 15
# 결과: videos/quick_demo.mp4 (약 3MB)
```

## 의존성

### 핵심 요구사항
- Python 3.8+
- PyTorch 2.0+
- Gymnasium 0.29+

### 비디오 파이프라인
- matplotlib 3.7+
- opencv-python 4.7+
- numpy, pyyaml, seaborn, pandas

```bash
pip install -r requirements.txt
```

## 결론

이 프로젝트를 통해 DQN과 DDPG가 모두 **결정적 정책**을 구현하지만, 그 **방식**과 **적용 도메인**에서 중요한 차이가 있음을 확인할 수 있습니다. DQN은 이산적 행동 공간에서 Q-값을 통해 간접적으로, DDPG는 연속적 행동 공간에서 액터를 통해 직접적으로 결정적 정책을 구현합니다.

🎬 **새롭게 추가된 비디오 파이프라인**을 통해 이러한 차이점과 학습 과정을 시각적으로 이해할 수 있으며, 교육 및 프레젠테이션 목적으로 활용할 수 있는 고품질 영상을 생성할 수 있습니다.

## 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다. 비디오 파이프라인을 포함한 모든 코드는 연구 및 교육 용도로 자유롭게 사용할 수 있습니다.