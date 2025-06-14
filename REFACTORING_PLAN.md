# 리팩토링 계획

## Phase 1: 즉시 삭제 (불필요한 파일들)

### 완전히 빈 파일들
```bash
rm patch_video_test.py
rm video_test.py
rm tests/patch_video_test.py  
rm tests/video_test.py
```

### 방금 만든 미사용 파일들
```bash
rm src/core/video_editor.py
rm docs/고급_비디오_편집_기능_로드맵.md
```

## Phase 2: 비디오 스크립트 통합

### 현재 상황
```
6개의 비디오 스크립트 → 2개로 통합
```

### 통합 계획
1. **유지할 파일**:
   - `quick_video_demo.py` → 간단한 데모용
   - `src/core/video_pipeline.py` → 전체 기능

2. **제거/통합할 파일**:
   - `render_learning_video.py` → `video_pipeline.py`로 통합
   - `create_comparison_video.py` → `video_pipeline.py`로 통합  
   - `create_sample_gameplay_videos.py` → `video_pipeline.py`로 통합
   - `run_full_video_pipeline.py` → 간소화하여 `video_pipeline.py` 호출

## Phase 3: 공통 유틸리티 추출

### 새로 생성할 파일
```python
# src/core/video_utils.py
class VideoEncoder:
    @staticmethod
    def save_with_opencv(frames, output_path, fps):
        """OpenCV 비디오 저장 - 모든 곳에서 공통 사용"""
    
    @staticmethod  
    def create_side_by_side(left_frame, right_frame):
        """나란히 배치 - 비교 영상용"""
    
    @staticmethod
    def add_text_overlay(frame, text, position):
        """텍스트 오버레이 - 모든 곳에서 사용"""

class SampleDataGenerator:
    @staticmethod
    def create_learning_curves(algorithm, episodes):
        """샘플 학습 데이터 생성"""
```

## Phase 4: 설정 통합

### 현재 문제
- 하드코딩된 설정값들이 여러 파일에 분산
- 동일한 설정이 중복 정의

### 해결책
```yaml
# configs/video_config.yaml (통합 설정)
demo:
  duration: 15
  fps: 15
  resolution: [640, 480]

pipeline:
  duration: 180
  fps: 30 
  resolution: [1280, 720]

comparison:
  layout: "side_by_side"
  labels: true
```

## Phase 5: 디렉토리 구조 정리

### 현재 → 리팩토링 후
```
현재:
├── quick_video_demo.py
├── render_learning_video.py
├── create_comparison_video.py
├── create_sample_gameplay_videos.py
├── run_full_video_pipeline.py
└── src/core/video_pipeline.py

리팩토링 후:
├── scripts/
│   └── demo.py                    # quick_video_demo.py 간소화
├── src/core/
│   ├── video_pipeline.py         # 모든 기능 통합
│   └── video_utils.py            # 공통 유틸리티
└── configs/
    └── video_config.yaml         # 통합 설정
```

## 리팩토링 효과

### Before (현재)
- 6개 비디오 스크립트
- 4개 빈 파일
- 중복된 OpenCV 코드 5곳
- 분산된 설정값들

### After (리팩토링 후)
- 2개 핵심 파일
- 0개 빈 파일  
- 1개 통합 유틸리티
- 1개 통합 설정파일

**결과**: 50% 파일 감소, 유지보수성 향상, 코드 중복 제거

## 우선순위

1. **즉시 (5분)**: 빈 파일들 삭제
2. **1시간**: 비디오 스크립트 통합
3. **30분**: 공통 유틸리티 추출
4. **30분**: 설정 파일 통합
5. **30분**: 디렉토리 정리

**총 소요시간**: 약 2.5시간