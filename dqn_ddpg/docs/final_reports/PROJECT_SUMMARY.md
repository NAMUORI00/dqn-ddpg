# 🎯 DQN vs DDPG 프로젝트 요약

## 📄 프로젝트 개요

이 프로젝트는 **DQN (Deep Q-Network)**과 **DDPG (Deep Deterministic Policy Gradient)** 알고리즘의 근본적인 차이점을 시각적으로 비교 분석하는 교육용 강화학습 시스템입니다.

**핵심 교육 목표**: 두 알고리즘의 **결정적 정책(Deterministic Policy)** 특성의 차이를 이해
- **DQN**: 암묵적 결정적 정책 (Q-값의 argmax)
- **DDPG**: 명시적 결정적 정책 (Actor 네트워크의 직접 출력)

## 🌟 주요 특징

### 1. 포괄적인 시각화 시스템
- **🎬 종합 통합 영상**: 게임플레이 + 실시간 그래프 + 성능 분석을 한 화면에
- **📊 실시간 그래프**: 에피소드별 보상, Q-값, 손실 변화 추적
- **📈 비교 분석**: DQN vs DDPG 성능 지표 나란히 비교

### 2. 고급 비디오 시스템
- **이중 품질 녹화**: 전체 과정 저품질 + 하이라이트 고품질
- **학습 과정 애니메이션**: 학습 곡선과 성능 변화 시각화
- **FFmpeg 독립적**: OpenCV 기반으로 폭넓은 호환성

### 3. 교육 중심 설계
- **한국어 문서**: 완전한 한국어 가이드 제공
- **단계별 학습**: 기초부터 고급까지 체계적 구성
- **실습 중심**: 코드 이해보다 개념 학습에 집중

## 📁 프로젝트 구조

```
dqn,ddpg/
├── src/                              # 핵심 소스 코드
│   ├── agents/                       # DQN, DDPG 에이전트
│   ├── networks/                     # 신경망 구조
│   ├── environments/                 # 환경 래퍼 및 시각화
│   └── core/                         # 비디오, 유틸리티
├── configs/                          # 설정 파일 (YAML)
├── docs/                             # 📚 한국어 문서
├── videos/                           # 🎬 생성된 영상 출력
├── tests/                            # 테스트 및 데모 스크립트
└── experiments/                      # 실험 결과 분석
```

## 🎬 시각화 출력 구조

### 메인 종합 시각화
```
videos/comprehensive_visualization/
├── comprehensive_dqn_vs_ddpg.mp4     # ⭐ 완전 통합 비교 영상
├── dqn_comprehensive.mp4             # DQN 개별 종합 영상
├── ddpg_comprehensive.mp4            # DDPG 개별 종합 영상
└── final_screenshots/                # 최종 스크린샷 모음
```

### 기존 개별 시각화
```
videos/
├── comparison/                       # 게임플레이 비교 영상
├── realtime_graph_test/             # 실시간 그래프 테스트
├── dqn/                             # DQN 개별 녹화
├── ddpg/                            # DDPG 개별 녹화
└── pipeline/                        # 학습 과정 애니메이션
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 프로젝트 다운로드
git clone <repository-url>
cd dqn,ddpg

# 가상환경 및 의존성 설치
conda create -n dqn_ddpg python=3.11
conda activate dqn_ddpg
pip install -r requirements.txt
```

### 2. 종합 시각화 생성 (권장)
```bash
# 완전한 통합 비교 영상 생성
python create_comprehensive_visualization.py --mode demo

# 결과: videos/comprehensive_visualization/comprehensive_dqn_vs_ddpg.mp4
```

### 3. 개별 구성요소 테스트
```bash
# 기본 데모
python tests/simple_demo.py

# 실시간 그래프 테스트
python test_realtime_graph.py --mode basic

# 스크린샷 생성
python test_comprehensive_screenshot.py
```

## 📋 핵심 기능별 사용법

### 🎯 종합 시각화 (메인 기능)
```bash
# 모든 요소가 통합된 완전한 비교 영상
python create_comprehensive_visualization.py --mode demo
```

### 📊 실시간 그래프
```bash
# 기본 실시간 그래프
python test_realtime_graph.py --mode basic

# 고급 메트릭 포함
python test_realtime_graph.py --mode advanced
```

### 🎬 비디오 생성
```bash
# 15초 빠른 데모
python quick_video_demo.py --duration 15

# 학습 과정 애니메이션
python render_learning_video.py --sample-data --all

# 게임플레이 비교
python create_comparison_video.py --auto
```

### 🧪 실험 및 학습
```bash
# 기본 학습 실행
python simple_training.py

# 완전한 실험 (비디오 포함)
python run_experiment.py --save-models --record-video
```

## 🎓 교육적 가치

### 알고리즘 차이점 시각화

| 특성 | DQN | DDPG |
|------|-----|------|
| **정책 유형** | 암묵적 결정적 | 명시적 결정적 |
| **행동 공간** | 이산 (0, 1) | 연속 ([-2, 2]) |
| **환경** | CartPole-v1 | Pendulum-v1 |
| **탐색 방법** | ε-greedy | 가우시안 노이즈 |
| **학습 패턴** | 빠른 변동 | 부드러운 곡선 |
| **보상 범위** | 0~500 | -2000~0 |

### 실시간 관찰 가능한 지표

1. **에피소드별 보상 추이** + 이동평균
2. **Q-값 변화** (실시간 업데이트)
3. **학습 손실** (로그 스케일)
4. **현재 에피소드 진행상황**
5. **탐색률 변화** (DQN만)
6. **성능 통계** (최고/평균/표준편차)

## 🛠️ 기술 스택

### 핵심 라이브러리
- **강화학습**: Gymnasium, PyTorch
- **시각화**: Matplotlib, OpenCV
- **비디오**: imageio, PIL
- **수치 계산**: NumPy, SciPy
- **설정 관리**: PyYAML

### 시각화 시스템
- **실시간 그래프**: matplotlib + OpenCV 통합
- **비디오 녹화**: cv2.VideoWriter 기반
- **레이아웃 관리**: 계층화된 프레임 구조
- **메트릭 추적**: deque 기반 효율적 데이터 관리

## 📚 문서 구조

### 완전한 한국어 가이드
- **📚 프로젝트 문서**: `docs/` 디렉토리에 체계적으로 정리
- **🧠 알고리즘_이론_분석.md**: `docs/analysis_reports/` - DQN vs DDPG 이론 분석
- **📋 연구계획서**: `docs/documentation/` - 프로젝트 설계 및 방법론
- **🔧 개발_진행_로그**: `docs/documentation/` - 실전 개발 과정
- **📊 최종 리포트**: `docs/final_reports/` - 종합 결과 및 발견사항

## 🎯 활용 사례

### 교육용
- **강의 시연**: 실시간 알고리즘 차이 설명
- **과제 제출**: 학습 과정 영상으로 결과 검증
- **개념 이해**: 이론과 실제의 완벽한 연결

### 연구용
- **논문 보조자료**: 실험 결과의 시각적 증명
- **알고리즘 개발**: 새로운 방법론의 성능 비교
- **벤치마킹**: 표준화된 비교 환경

### 프레젠테이션용
- **학회 발표**: 포괄적인 결과 시연
- **기술 데모**: 완성도 높은 시각화
- **포트폴리오**: 고품질 시각 자료

## ⚡ 성능 최적화

### 시스템 요구사항
- **최소**: Python 3.8+, 4GB RAM
- **권장**: Python 3.11+, 8GB RAM
- **선택**: GPU (학습 가속화)

### 최적화 기법
- **메모리 관리**: deque maxlen으로 메모리 제한
- **비디오 압축**: 품질별 설정 제공
- **백엔드 최적화**: matplotlib Agg 백엔드 사용
- **병렬 처리**: 이중 품질 녹화 시스템

## 🔮 확장 가능성

### 새로운 알고리즘 추가
- 래퍼 시스템으로 쉬운 통합
- 기존 시각화 인프라 재사용
- 설정 파일 기반 커스터마이징

### 추가 환경 지원
- Gymnasium 호환 환경 자동 지원
- 커스텀 환경 래퍼 제공
- 다양한 관찰/행동 공간 대응

### 고급 시각화
- 인터랙티브 대시보드
- 웹 기반 시각화
- 실시간 스트리밍

## 🤝 기여 및 확장

이 프로젝트는 교육 목적으로 설계되어 확장과 개선이 환영됩니다:

1. **새로운 알고리즘** 추가
2. **추가 시각화** 기능 개발
3. **문서 개선** 및 번역
4. **버그 리포트** 및 성능 개선

## 📞 지원 및 문의

- **문서**: `docs/` 디렉토리의 상세 가이드 참조
- **이슈**: GitHub Issues를 통한 문제 보고
- **토론**: 교육적 활용 방안 논의 환영

---

**🎉 이 프로젝트로 DQN과 DDPG의 차이점을 직관적으로 이해하고, 강화학습의 핵심 개념을 시각적으로 탐험해보세요!**