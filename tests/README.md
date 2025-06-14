# 테스트 및 데모 코드

이 폴더에는 DQN vs DDPG 프로젝트의 다양한 테스트와 데모 스크립트들이 포함되어 있습니다.

## 📁 파일 구성

### 🎯 주요 데모 및 테스트

| 파일명 | 설명 | 실행 시간 | 용도 |
|--------|------|-----------|------|
| `simple_demo.py` | 기본 결정적 정책 시연 | ~30초 | 빠른 개념 확인 |
| `detailed_test.py` | 상세 분석 및 시각화 | ~2분 | 심화 분석 |

### 🎥 비디오 관련 테스트

| 파일명 | 설명 | 기능 |
|--------|------|------|
| `video_test.py` | 기본 비디오 녹화 테스트 | 단일 환경 녹화 |
| `test_video_recording.py` | 고급 비디오 녹화 테스트 | 이중 품질 녹화 |
| `simple_dual_test.py` | 이중 녹화 시스템 테스트 | 간단한 이중 녹화 |
| `patch_video_test.py` | 비디오 패치 테스트 | 녹화 시스템 수정 |

## 🚀 실행 방법

### 환경 활성화
```bash
conda activate ddpg_dqn
# 또는
source /home/rladb/miniconda/etc/profile.d/conda.sh && conda activate ddpg_dqn
```

### 기본 데모 실행
```bash
# 간단한 시연 (권장 시작점)
python tests/simple_demo.py

# 상세 분석
python tests/detailed_test.py
```

### 비디오 테스트 실행
```bash
# 기본 비디오 테스트
python tests/video_test.py

# 고급 비디오 녹화 테스트
python tests/test_video_recording.py

# 이중 녹화 시스템 테스트
python tests/simple_dual_test.py
```

## 📊 예상 출력

### simple_demo.py
- DQN의 Q-값 분석 및 행동 일관성 테스트
- DDPG의 액터 출력 분석 및 노이즈 영향 비교
- 두 알고리즘의 정책 특성 요약

### detailed_test.py
- 행동 일관성 정량적 분석
- Q-값/액터 출력 상세 분석
- 탐험 메커니즘 영향도 측정
- 시각화 결과: `results/deterministic_policy_analysis.png`

### 비디오 테스트들
- 환경별 에이전트 행동 녹화
- 다양한 품질 설정 테스트
- 녹화 스케줄링 검증

## 🔧 문제 해결

### 폰트 경고 (한글 표시 문제)
```
UserWarning: Glyph missing from font(s) DejaVu Sans
```
**해결방법**: 기능상 문제없음. 시각화에서 한글이 네모로 표시될 수 있으나 모든 기능은 정상 작동

### OpenCV 에러
```bash
# OpenCV 재설치
pip install opencv-python --upgrade
```

### 메모리 부족
- 비디오 테스트 시 메모리 사용량 증가
- 필요시 비디오 해상도를 `configs/video_recording.yaml`에서 조정

## 🎓 교육적 활용

### 1단계: 개념 이해
```bash
python tests/simple_demo.py
```
- DQN과 DDPG의 기본 차이점 학습
- 결정적 정책의 의미 이해

### 2단계: 심화 분석
```bash
python tests/detailed_test.py
```
- 정량적 분석 결과 검토
- 시각화 자료 해석

### 3단계: 실제 적용
```bash
python tests/test_video_recording.py
```
- 실제 환경에서의 에이전트 행동 관찰
- 비디오 분석을 통한 정책 특성 확인

## 📈 성능 벤치마크

| 테스트 | 평균 실행 시간 | 메모리 사용량 | 디스크 사용량 |
|--------|----------------|---------------|---------------|
| simple_demo | 30초 | ~200MB | 무시 가능 |
| detailed_test | 2분 | ~300MB | ~2MB (이미지) |
| video_test | 1분 | ~500MB | ~10MB (비디오) |

## 🔍 추가 정보

- **설정 파일**: `configs/` 폴더의 YAML 파일들로 모든 파라미터 조정 가능
- **결과 저장**: `results/` 폴더에 자동 저장
- **로그**: 콘솔 출력으로 실행 과정 모니터링 가능

## 🚨 주의사항

1. **환경 의존성**: WSL2 Linux 환경에서 테스트됨
2. **GPU 사용**: CUDA 사용 가능시 자동으로 GPU 활용
3. **디스크 공간**: 비디오 테스트 시 충분한 디스크 공간 확보 필요
4. **실행 순서**: simple_demo.py → detailed_test.py → video_test.py 순서 권장