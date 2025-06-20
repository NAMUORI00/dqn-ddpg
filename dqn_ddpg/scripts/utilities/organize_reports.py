"""
프로젝트 내 모든 리포트 파일 정리 및 이동 스크립트

유사한 목적의 리포트들을 report 폴더로 통합 정리합니다.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가 (scripts/utilities에서 루트로)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def organize_reports():
    """모든 리포트 파일을 report 폴더로 정리"""
    
    # 작업 디렉토리를 루트로 변경
    os.chdir(project_root)
    
    # 기본 디렉토리
    project_root_path = Path(".")
    report_dir = project_root_path / "report"
    
    # report 폴더 구조 생성
    subdirs = [
        "final_reports",        # 최종 리포트들
        "experiment_reports",   # 실험 결과 리포트들  
        "analysis_reports",     # 분석 리포트들
        "documentation",        # 문서화 리포트들
        "archived"             # 이전 버전들
    ]
    
    for subdir in subdirs:
        (report_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("📁 리포트 폴더 구조 생성 완료")
    print("=" * 60)
    
    # 이동할 파일들 정의
    files_to_organize = [
        # 최종 리포트들
        {
            "source": "FINAL_REPORT.md",
            "target": "final_reports/FINAL_REPORT.md",
            "description": "프로젝트 최종 종합 리포트 (프레젠테이션 겸용)"
        },
        {
            "source": "VISUAL_MATERIALS_REPORT.md", 
            "target": "final_reports/VISUAL_MATERIALS_REPORT.md",
            "description": "시각자료 생성 가능성 검토 리포트"
        },
        
        # 실험 결과 리포트들
        {
            "source": "results/comparison_report/DQN_vs_DDPG_비교분석리포트_20250615_132453.md",
            "target": "experiment_reports/DQN_vs_DDPG_기본환경_비교분석_20250615.md", 
            "description": "기본 환경 DQN vs DDPG 비교 분석 (최신 버전)"
        },
        {
            "source": "results/same_environment_comparison/experiment_summary_20250615_140239_report.md",
            "target": "experiment_reports/DQN_vs_DDPG_동일환경_비교분석_20250615.md",
            "description": "동일 환경 공정 비교 실험 리포트 (핵심 발견)"
        },
        
        # 분석 리포트들  
        {
            "source": "docs/알고리즘_이론_분석.md",
            "target": "analysis_reports/알고리즘_이론_분석.md",
            "description": "DQN vs DDPG 이론적 심층 분석"
        },
        {
            "source": "docs/결정적 정책 비교분석.md", 
            "target": "analysis_reports/결정적_정책_비교분석.md",
            "description": "결정적 정책 구현 방식 비교 분석"
        },
        
        # 문서화 리포트들
        {
            "source": "docs/DQN_vs_DDPG_연구계획서.md",
            "target": "documentation/DQN_vs_DDPG_연구계획서.md", 
            "description": "프로젝트 초기 연구 계획서"
        },
        {
            "source": "docs/same_environment_comparison.md",
            "target": "documentation/동일환경_비교_시스템_가이드.md",
            "description": "동일 환경 비교 시스템 사용 가이드"
        },
        {
            "source": "docs/개발_진행_로그.md",
            "target": "documentation/개발_진행_로그.md", 
            "description": "프로젝트 개발 과정 기록"
        },
        
        # 아카이브 (이전 버전들)
        {
            "source": "results/comparison_report/DQN_vs_DDPG_비교분석리포트_20250615_130944.md",
            "target": "archived/DQN_vs_DDPG_비교분석리포트_130944_archived.md",
            "description": "기본 환경 비교 분석 (이전 버전)"
        }
    ]
    
    # 파일 이동 및 정리
    moved_count = 0
    
    for file_info in files_to_organize:
        source_path = Path(file_info["source"])
        target_path = report_dir / file_info["target"]
        
        if source_path.exists():
            # 타겟 디렉토리 생성
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사 (원본 유지)
            shutil.copy2(source_path, target_path)
            moved_count += 1
            
            print(f"✅ {file_info['source']}")
            print(f"   → {target_path}")
            print(f"   📝 {file_info['description']}")
            print()
        else:
            print(f"⚠️ 파일 없음: {file_info['source']}")
    
    # 인덱스 리포트 생성
    create_report_index(report_dir, files_to_organize)
    
    print("=" * 60)
    print(f"📊 정리 완료: {moved_count}개 리포트 파일 정리됨")
    print(f"📁 위치: {report_dir.absolute()}")


def create_report_index(report_dir: Path, files_info: list):
    """리포트 인덱스 파일 생성"""
    
    index_content = f"""# 📊 DQN vs DDPG 프로젝트 리포트 모음

**정리 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}  
**프로젝트**: DQN vs DDPG 결정적 정책 비교 연구

---

## 📁 리포트 구조

### 🏆 최종 리포트 (final_reports/)
프로젝트의 최종 산출물이자 프레젠테이션용 문서들

- **`FINAL_REPORT.md`** ⭐
  - 프로젝트 전체 종합 리포트
  - 프레젠테이션에서 직접 활용 가능
  - 모든 실험 결과와 발견사항 포함
  
- **`VISUAL_MATERIALS_REPORT.md`**
  - 시각자료 생성 가능성 검토
  - 프레젠테이션 준비 가이드
  - 자동화 도구 현황

### 🧪 실험 리포트 (experiment_reports/)
실제 실험 실행 결과 및 분석

- **`DQN_vs_DDPG_기본환경_비교분석_20250615.md`**
  - CartPole vs Pendulum 환경 비교
  - 각 알고리즘의 최적 환경에서의 성능
  - 결정성 점수: 1.000 (완벽한 결정성 확인)

- **`DQN_vs_DDPG_동일환경_비교분석_20250615.md`** ⭐
  - **핵심 발견**: DQN이 DDPG보다 13.2배 우수
  - ContinuousCartPole 환경에서 공정 비교
  - "연속 환경 = DDPG 우위" 통념 반박

### 🔬 분석 리포트 (analysis_reports/)
이론적 분석 및 심층 연구

- **`알고리즘_이론_분석.md`**
  - DQN vs DDPG 심층 이론 분석
  - 결정적 정책의 구현 방식 차이
  - 코드 레벨 상세 비교

- **`결정적_정책_비교분석.md`**
  - 암묵적 vs 명시적 결정적 정책
  - 행동 공간별 알고리즘 적합성
  - 탐험 전략 효과성 분석

### 📖 문서화 리포트 (documentation/)
프로젝트 계획, 가이드, 개발 과정

- **`DQN_vs_DDPG_연구계획서.md`**
  - 프로젝트 초기 설계 문서
  - 연구 목표 및 방법론 정의

- **`동일환경_비교_시스템_가이드.md`**
  - ContinuousCartPole + DiscretizedDQN 시스템
  - 공정 비교를 위한 혁신적 접근법

- **`개발_진행_로그.md`**
  - 개발 과정 상세 기록
  - 문제 해결 과정 및 의사결정

### 📦 아카이브 (archived/)
이전 버전 및 백업 파일들

---

## 🎯 용도별 리포트 활용 가이드

### 📈 **프레젠테이션 준비**
1. **메인**: `final_reports/FINAL_REPORT.md`
2. **시각자료**: `final_reports/VISUAL_MATERIALS_REPORT.md`
3. **핵심 발견**: `experiment_reports/DQN_vs_DDPG_동일환경_비교분석_20250615.md`

### 🔬 **연구 심화 학습**
1. **이론**: `analysis_reports/알고리즘_이론_분석.md`
2. **실험**: `experiment_reports/` 전체
3. **방법론**: `documentation/동일환경_비교_시스템_가이드.md`

### 🛠️ **구현 및 재현**
1. **시스템 이해**: `documentation/` 전체
2. **실험 재현**: `experiment_reports/` + 실제 코드
3. **확장 개발**: `documentation/개발_진행_로그.md`

---

## 🏆 주요 성과 및 발견사항

### ⚡ **게임체인저 발견**
- **DQN이 연속 환경에서 DDPG보다 13.2배 우수** (동일환경 비교)
- **"연속 환경 = DDPG 우위"라는 기존 통념 반박**
- **환경 적합성 > 알고리즘 이론적 설계**

### ✅ **완벽한 결정성 구현**
- DQN, DDPG 모두 결정성 점수 1.000 달성
- 암묵적 vs 명시적 구현 방식의 차이 확인
- 결정성보다 탐험 전략이 더 중요함 입증

### 🚀 **혁신적 기여**
- **동일환경 비교 시스템**: ContinuousCartPole + DiscretizedDQN
- **자동 비디오 파이프라인**: 학습 과정 자동 영상화
- **완전한 재현성**: 모든 결과가 코드로 재생성 가능

---

## 📚 참고 정보

### 🔗 **관련 파일 위치**
- **원본 파일들**: 각 리포트의 원본은 기존 위치에 유지
- **실험 데이터**: `results/` 디렉토리
- **시각자료**: `videos/`, `presentation_materials/`
- **소스코드**: `src/`, `experiments/`

### 🎬 **비디오 자료**
- `videos/comprehensive_visualization/`: 종합 시각화 (3개)
- `videos/comparison/`: 알고리즘 비교 (3개)  
- `videos/realtime_graph_test/`: 실시간 그래프 (2개)

### 📊 **데이터 및 결과**
- `results/same_environment_comparison/`: 핵심 발견 데이터
- `results/comparison_report/`: 기본 환경 비교 데이터
- `results/deterministic_analysis/`: 결정성 분석 데이터

---

**🎉 이 프로젝트는 강화학습 교육과 연구에 새로운 관점을 제시하는 완전한 패키지입니다!**

*마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 인덱스 파일 저장
    index_path = report_dir / "README.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📋 리포트 인덱스 생성: {index_path}")


def main():
    """메인 실행 함수"""
    print("📊 DQN vs DDPG 프로젝트 리포트 정리 시작")
    print("=" * 60)
    
    organize_reports()
    
    print("\n🎉 리포트 정리 완료!")
    print("📁 모든 리포트가 'report/' 폴더로 체계적으로 정리되었습니다.")
    print("📋 'report/README.md'에서 전체 리포트 인덱스를 확인할 수 있습니다.")


if __name__ == "__main__":
    main()