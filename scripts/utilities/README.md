# 유틸리티 도구 (Utilities)

프로젝트 관리, 검증, 문서화, 테스트를 위한 유틸리티 도구들입니다.

## 📁 포함된 도구들

### 🎨 `generate_presentation_materials.py` - 종합 프레젠테이션 자료 생성
**가장 종합적인 도구**로, 모든 프레젠테이션 자료를 한 번에 생성합니다.

```bash
# 기본 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py

# 고품질 버전
python scripts/utilities/generate_presentation_materials.py --high-quality

# 특정 타입만 생성
python scripts/utilities/generate_presentation_materials.py --type charts,tables
```

**생성되는 자료들:**
- **차트**: 학습 곡선 비교, 성능 메트릭, 알고리즘 분석
- **테이블**: 성능 비교표, 통계 요약, 하이퍼파라미터 정보
- **다이어그램**: 알고리즘 구조, 환경 비교, 워크플로
- **인포그래픽**: 핵심 발견 요약, 연구 임팩트, 방법론 혁신

**출력 위치:**
```
output/visualization/
├── images/png/charts/          # 차트 이미지들
├── images/png/diagrams/        # 다이어그램들  
├── documents/md/               # 마크다운 리포트들
└── data/json/summaries/        # 요약 데이터
```

**주요 생성 차트:**
- `learning_curves_comparison.png` - 학습 곡선 비교
- `performance_comparison.png` - 성능 비교 막대그래프
- `training_metrics.png` - 상세 학습 메트릭
- `algorithm_comparison_table.png` - 알고리즘 비교표

---

### ✅ `check_presentation_materials.py` - 자료 검증 도구
생성된 프레젠테이션 자료들의 완성도와 품질을 검증합니다.

```bash
# 모든 자료 검증
python scripts/utilities/check_presentation_materials.py

# 상세 검증 (파일 크기, 해상도 등)
python scripts/utilities/check_presentation_materials.py --detailed

# 특정 카테고리만 검증
python scripts/utilities/check_presentation_materials.py --category charts
```

**검증 항목:**
- **파일 존재성**: 필수 파일들이 모두 생성되었는지 확인
- **파일 품질**: 이미지 해상도, 파일 크기, 형식 검증
- **내용 완성도**: 차트 데이터, 테이블 내용, 텍스트 품질
- **일관성**: 스타일, 색상, 폰트 일관성 확인

**검증 리포트:**
```
✅ 차트 생성 완료: 15/15 파일
✅ 테이블 생성 완료: 8/8 파일  
⚠️  비디오 일부 누락: 3/5 파일
❌ 인포그래픽 해상도 부족: 2파일
```

**출력:**
- 검증 결과 요약
- 누락된 파일 목록
- 품질 문제 리포트
- 수정 권장사항

---

### 📚 `organize_reports.py` - 리포트 정리 및 인덱싱
프로젝트의 모든 리포트와 문서를 체계적으로 정리합니다.

```bash
# 기본 리포트 정리
python scripts/utilities/organize_reports.py

# 새로운 구조로 마이그레이션
python scripts/utilities/organize_reports.py --migrate-to-new-structure

# 인덱스 재생성
python scripts/utilities/organize_reports.py --rebuild-index
```

**정리 기능:**
- **자동 분류**: 날짜, 타입, 중요도별 문서 분류
- **중복 제거**: 동일한 내용의 리포트 통합
- **인덱스 생성**: 모든 문서의 마스터 인덱스 생성
- **메타데이터**: 각 문서에 태그 및 설명 추가

**생성되는 구조:**
```
docs/
├── final_reports/           # 최종 완성 리포트들
├── experiment_reports/      # 실험별 상세 리포트
├── analysis_reports/        # 분석 및 이론 리포트
├── archive/                 # 아카이브된 실험 결과
└── INDEX.md                 # 마스터 인덱스
```

**마스터 인덱스 내용:**
- 모든 리포트 목록과 요약
- 중요도별 분류
- 날짜순 타임라인
- 태그별 검색 가능

---

### 🧪 `test_visualization_refactor.py` - 시각화 시스템 테스트
리팩토링된 새로운 시각화 시스템을 테스트합니다.

```bash
# 기본 시스템 테스트
python scripts/utilities/test_visualization_refactor.py

# 전체 모듈 테스트
python scripts/utilities/test_visualization_refactor.py --full-test

# 성능 벤치마크
python scripts/utilities/test_visualization_refactor.py --benchmark
```

**테스트 항목:**
- **모듈 import**: 모든 시각화 모듈이 정상 import되는지
- **기능 테스트**: 핵심 기능들이 올바르게 작동하는지  
- **출력 검증**: 생성된 파일들이 예상 위치에 저장되는지
- **성능 측정**: 생성 시간 및 메모리 사용량 측정

**테스트 결과:**
```
✅ BaseVisualizer 초기화 성공
✅ ComparisonVisualizer 차트 생성 성공  
✅ LearningCurveVisualizer 그래프 생성 성공
✅ 새로운 출력 구조 정상 작동
⏱️  평균 차트 생성 시간: 2.3초
```

## 🎯 워크플로별 사용 가이드

### 📊 프레젠테이션 준비 워크플로

```bash
# 1. 모든 자료 생성
python scripts/utilities/generate_presentation_materials.py --high-quality

# 2. 생성된 자료 검증
python scripts/utilities/check_presentation_materials.py --detailed

# 3. 문제가 있다면 개별 재생성
python scripts/utilities/generate_presentation_materials.py --type charts --force-regenerate

# 4. 최종 검증
python scripts/utilities/check_presentation_materials.py
```

### 🔧 개발/디버깅 워크플로

```bash
# 1. 시각화 시스템 테스트
python scripts/utilities/test_visualization_refactor.py --full-test

# 2. 문제 발견시 개별 모듈 테스트
python -c "from src.visualization.charts.comparison import ComparisonVisualizer; print('OK')"

# 3. 수정 후 재테스트
python scripts/utilities/test_visualization_refactor.py --benchmark
```

### 📚 문서 정리 워크플로

```bash
# 1. 현재 문서 상태 확인
python scripts/utilities/organize_reports.py --status

# 2. 새로운 구조로 마이그레이션 (필요시)
python scripts/utilities/organize_reports.py --migrate-to-new-structure

# 3. 인덱스 재생성
python scripts/utilities/organize_reports.py --rebuild-index

# 4. 최종 확인
python scripts/utilities/check_presentation_materials.py
```

### 🚀 배포 준비 워크플로

```bash
# 1. 전체 시스템 테스트
python scripts/utilities/test_visualization_refactor.py --full-test

# 2. 모든 프레젠테이션 자료 생성
python scripts/utilities/generate_presentation_materials.py --high-quality

# 3. 문서 정리 및 인덱싱
python scripts/utilities/organize_reports.py --rebuild-index

# 4. 최종 검증
python scripts/utilities/check_presentation_materials.py --detailed

echo "배포 준비 완료!"
```

## ⚙️ 설정 및 옵션

### 공통 옵션들
- `--high-quality` - 고품질 출력 (높은 DPI, 큰 크기)
- `--force-regenerate` - 기존 파일 있어도 강제 재생성
- `--output-dir DIR` - 출력 디렉토리 지정
- `--config FILE` - 커스텀 설정 파일 사용

### generate_presentation_materials.py 전용
- `--type charts,tables,diagrams,infographics` - 생성할 자료 타입 선택
- `--style professional,academic,simple` - 스타일 선택
- `--language korean,english` - 언어 선택

### check_presentation_materials.py 전용
- `--detailed` - 상세 검증 (파일 크기, 해상도 등)
- `--category charts,tables,videos` - 특정 카테고리만 검증
- `--fix-issues` - 발견된 문제 자동 수정 시도

### organize_reports.py 전용
- `--migrate-to-new-structure` - 새로운 디렉토리 구조로 마이그레이션
- `--rebuild-index` - 마스터 인덱스 재생성
- `--status` - 현재 문서 상태만 확인

### test_visualization_refactor.py 전용
- `--full-test` - 모든 모듈 전체 테스트
- `--benchmark` - 성능 벤치마크 측정
- `--module MODULE` - 특정 모듈만 테스트

## 📊 출력 및 결과

### 생성되는 프레젠테이션 자료
```
presentation_materials/
├── charts/
│   ├── learning_curves_comparison.png
│   ├── performance_comparison.png
│   ├── training_metrics.png
│   └── algorithm_analysis.png
│
├── tables/
│   ├── algorithm_comparison_table.png
│   ├── performance_statistics.png
│   └── hyperparameter_summary.png
│
├── diagrams/
│   ├── system_architecture.png
│   ├── experiment_workflow.png
│   └── environment_comparison.png
│
└── infographics/
    ├── key_findings_summary.png
    ├── research_impact.png
    └── methodology_innovation.png
```

### 검증 리포트
```
verification_reports/
├── material_verification_YYYYMMDD_HHMMSS.json
├── quality_check_summary.md
└── missing_files_report.txt
```

### 문서 인덱스
```
docs/INDEX.md                 # 마스터 인덱스
docs/by_date/                 # 날짜별 문서
docs/by_category/             # 카테고리별 문서  
docs/by_importance/           # 중요도별 문서
```

## 🔍 문제 해결

### 일반적인 문제들

**자료 생성 실패:**
```bash
# 의존성 확인
python -c "import matplotlib, seaborn, pandas; print('Dependencies OK')"

# 단계별 생성
python scripts/utilities/generate_presentation_materials.py --type charts
```

**검증 실패:**
```bash
# 상세 로그 확인
python scripts/utilities/check_presentation_materials.py --detailed --verbose

# 개별 파일 확인
ls -la presentation_materials/charts/
```

**문서 정리 문제:**
```bash
# 권한 확인
ls -la docs/

# 단계별 정리
python scripts/utilities/organize_reports.py --status
```

### 성능 최적화

**빠른 생성:**
- 낮은 품질 설정 사용
- 필요한 타입만 선택적 생성
- 캐시된 데이터 활용

**고품질 결과:**
- `--high-quality` 플래그 사용
- 충분한 메모리 및 디스크 공간 확보
- 병렬 처리 비활성화 (안정성 우선)

## 🏆 모범 사례

1. **정기적 검증**: 자료 생성 후 항상 검증 실행
2. **점진적 생성**: 문제 발생시 타입별로 개별 생성
3. **백업 유지**: 중요한 자료는 여러 형식으로 저장
4. **문서화**: 생성 과정과 설정을 기록
5. **버전 관리**: 중요한 변경사항은 날짜별로 관리

이 유틸리티 도구들을 통해 DQN vs DDPG 프로젝트의 모든 자료를 효율적이고 체계적으로 관리할 수 있습니다.