#!/usr/bin/env python3
"""
Video Test - 비디오 기능 테스트 스크립트

이 스크립트는 프로젝트의 모든 비디오 관련 기능을 테스트합니다:
- 기본 비디오 녹화
- 학습 과정 렌더링
- 비교 영상 생성
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_video_demo():
    """기본 비디오 데모 테스트"""
    print("🎬 기본 비디오 데모 테스트 중...")
    
    try:
        from quick_video_demo import main as demo_main
        result = demo_main(duration=10, fps=15, output="videos/test_demo.mp4")
        print(f"✅ 기본 데모 완료: {result}")
        return True
    except Exception as e:
        print(f"❌ 기본 데모 실패: {e}")
        return False

def test_learning_video():
    """학습 과정 비디오 테스트"""
    print("📊 학습 과정 비디오 테스트 중...")
    
    try:
        # 간단한 샘플 데이터로 테스트
        import subprocess
        result = subprocess.run([
            sys.executable, "render_learning_video.py", 
            "--sample-data", "--learning-only", "--duration", "15"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 학습 과정 비디오 완료")
            return True
        else:
            print(f"❌ 학습 과정 비디오 실패: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 학습 과정 비디오 실패: {e}")
        return False

def test_comparison_video():
    """비교 영상 테스트"""
    print("🤖 비교 영상 테스트 중...")
    
    try:
        # 샘플 게임플레이 영상으로 테스트
        from src.core.video_utils import SampleDataGenerator, VideoEncoder
        import numpy as np
        
        # 샘플 프레임 생성
        frames_dqn = []
        frames_ddpg = []
        
        for i in range(30):  # 1초 (30 FPS)
            frame_dqn = SampleDataGenerator.create_sample_frame("dqn", episode=1, step=i)
            frame_ddpg = SampleDataGenerator.create_sample_frame("ddpg", episode=1, step=i)
            frames_dqn.append(frame_dqn)
            frames_ddpg.append(frame_ddpg)
        
        # 비디오 저장
        os.makedirs("videos/test", exist_ok=True)
        VideoEncoder.save_with_opencv(frames_dqn, "videos/test/sample_dqn.mp4")
        VideoEncoder.save_with_opencv(frames_ddpg, "videos/test/sample_ddpg.mp4")
        
        print("✅ 비교 영상 테스트 완료")
        return True
    except Exception as e:
        print(f"❌ 비교 영상 테스트 실패: {e}")
        return False

def test_video_utilities():
    """비디오 유틸리티 테스트"""
    print("🔧 비디오 유틸리티 테스트 중...")
    
    try:
        from src.core.video_utils import check_video_dependencies
        deps = check_video_dependencies()
        
        print("비디오 의존성 확인:")
        for name, version in deps.items():
            status = "✅" if version else "❌"
            print(f"  {status} {name}: {version}")
        
        return True
    except Exception as e:
        print(f"❌ 비디오 유틸리티 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🎬 비디오 시스템 종합 테스트 시작")
    print("=" * 50)
    
    results = []
    
    # 의존성 확인
    results.append(test_video_utilities())
    
    # 기본 데모 테스트
    results.append(test_basic_video_demo())
    
    # 학습 과정 비디오 테스트
    results.append(test_learning_video())
    
    # 비교 영상 테스트
    results.append(test_comparison_video())
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("🎯 테스트 결과 요약:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"성공: {passed}/{total}")
    
    if passed == total:
        print("🎉 모든 비디오 기능이 정상 작동합니다!")
        return 0
    else:
        print("⚠️  일부 기능에 문제가 있습니다. 로그를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)