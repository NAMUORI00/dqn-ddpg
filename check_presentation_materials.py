"""
프레젠테이션 자료 생성 결과 확인 스크립트
"""

import os
from pathlib import Path

def check_presentation_materials():
    """생성된 프레젠테이션 자료 확인"""
    
    output_dir = Path("presentation_materials")
    
    print("📊 프레젠테이션 자료 생성 결과 확인")
    print("=" * 50)
    
    # 각 카테고리별 확인
    categories = {
        "charts": "차트",
        "tables": "테이블", 
        "diagrams": "다이어그램",
        "infographics": "인포그래픽"
    }
    
    total_files = 0
    
    for category, korean_name in categories.items():
        cat_dir = output_dir / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*.png"))
            print(f"\n📁 {korean_name} ({category}/)")
            if files:
                for file in files:
                    file_size = file.stat().st_size / 1024  # KB
                    print(f"  ✅ {file.name} ({file_size:.1f} KB)")
                    total_files += 1
            else:
                print(f"  ⚠️ 파일 없음")
        else:
            print(f"\n📁 {korean_name} ({category}/)")
            print(f"  ⚠️ 디렉토리 없음")
    
    # 기존 파일 확인
    print(f"\n📁 기존 시각자료")
    existing_files = [
        "results/comparison_report/comprehensive_comparison.png",
        "results/comparison_report/learning_curves_comparison.png", 
        "results/deterministic_analysis/deterministic_policy_analysis.png",
        "videos/realtime_graph_test/screenshots/dqn_vs_ddpg_comparison.png"
    ]
    
    existing_count = 0
    for file_path in existing_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / 1024
            print(f"  ✅ {file_path} ({file_size:.1f} KB)")
            existing_count += 1
        else:
            print(f"  ❌ {file_path} - 없음")
    
    # 비디오 자료 확인
    print(f"\n🎬 비디오 자료")
    video_dirs = [
        "videos/comprehensive_visualization/",
        "videos/comparison/",
        "videos/realtime_graph_test/"
    ]
    
    video_count = 0
    for video_dir in video_dirs:
        if Path(video_dir).exists():
            mp4_files = list(Path(video_dir).glob("*.mp4"))
            if mp4_files:
                print(f"  📂 {video_dir}: {len(mp4_files)}개 영상")
                video_count += len(mp4_files)
            else:
                print(f"  📂 {video_dir}: 영상 없음")
        else:
            print(f"  📂 {video_dir}: 디렉토리 없음")
    
    # 요약
    print(f"\n" + "=" * 50)
    print(f"📊 요약")
    print(f"=" * 50)
    print(f"🆕 새로 생성된 자료: {total_files}개")
    print(f"📁 기존 이미지 자료: {existing_count}개") 
    print(f"🎬 비디오 자료: {video_count}개")
    print(f"📈 총 시각자료: {total_files + existing_count + video_count}개")
    
    # 프레젠테이션 준비도 평가
    print(f"\n🎯 프레젠테이션 준비도")
    print(f"=" * 50)
    
    if total_files >= 3:
        print("✅ 기본 프레젠테이션 자료 준비 완료")
    else:
        print("⚠️ 기본 자료가 부족합니다")
    
    if existing_count >= 3:
        print("✅ 상세 분석 자료 준비 완료")
    else:
        print("⚠️ 상세 분석 자료가 부족합니다")
    
    if video_count >= 3:
        print("✅ 비디오 시연 자료 준비 완료")
    else:
        print("⚠️ 비디오 자료가 부족합니다")
    
    # 권장사항
    print(f"\n💡 권장 프레젠테이션 구성")
    print(f"=" * 50)
    print("15분 발표: 기본 자료 3-4개 + 핵심 비디오 1개")
    print("30분 발표: 기본 자료 + 상세 분석 + 비디오 2-3개")
    print("45분 발표: 모든 자료 활용 + 라이브 데모")
    
    return total_files, existing_count, video_count

if __name__ == "__main__":
    check_presentation_materials()