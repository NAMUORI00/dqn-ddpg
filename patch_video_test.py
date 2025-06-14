#!/usr/bin/env python3
"""
Patch Video Test - 비디오 패치 및 호환성 테스트

이 스크립트는 다양한 시스템 환경에서의 비디오 기능 호환성을 테스트하고
필요시 대체 방법을 제공합니다.
"""

import sys
import os
import subprocess
from pathlib import Path

def test_ffmpeg_availability():
    """FFmpeg 사용 가능성 테스트"""
    print("🎬 FFmpeg 사용 가능성 확인 중...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg 사용 가능")
            return True
        else:
            print("❌ FFmpeg 사용 불가")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg 설치되지 않음")
        return False

def test_opencv_video_codec():
    """OpenCV 비디오 코덱 테스트"""
    print("📹 OpenCV 비디오 코덱 테스트 중...")
    
    try:
        import cv2
        import numpy as np
        
        # 테스트 비디오 생성
        test_dir = Path("videos/test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 다양한 코덱 테스트
        codecs_to_test = [
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
        ]
        
        successful_codecs = []
        
        for codec, extension in codecs_to_test:
            try:
                filename = test_dir / f"test_codec_{codec}{extension}"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(filename), fourcc, 20.0, (640, 480))
                
                # 테스트 프레임 작성
                for i in range(10):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    out.write(frame)
                
                out.release()
                
                # 파일이 실제로 생성되었는지 확인
                if filename.exists() and filename.stat().st_size > 1000:
                    successful_codecs.append(codec)
                    print(f"  ✅ {codec} 코덱 사용 가능")
                else:
                    print(f"  ❌ {codec} 코덱 사용 불가")
                    
            except Exception as e:
                print(f"  ❌ {codec} 코덱 테스트 실패: {e}")
        
        if successful_codecs:
            print(f"✅ OpenCV 비디오 코덱 사용 가능: {successful_codecs}")
            return True, successful_codecs
        else:
            print("❌ 사용 가능한 OpenCV 비디오 코덱 없음")
            return False, []
            
    except ImportError:
        print("❌ OpenCV 설치되지 않음")
        return False, []
    except Exception as e:
        print(f"❌ OpenCV 비디오 코덱 테스트 실패: {e}")
        return False, []

def test_matplotlib_animation():
    """Matplotlib 애니메이션 기능 테스트"""
    print("📊 Matplotlib 애니메이션 테스트 중...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI 없는 백엔드 사용
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        
        # 간단한 애니메이션 생성 테스트
        fig, ax = plt.subplots()
        x = np.linspace(0, 2*np.pi, 100)
        line, = ax.plot(x, np.sin(x))
        
        def animate(frame):
            line.set_ydata(np.sin(x + frame * 0.1))
            return line,
        
        # 애니메이션 객체 생성만 테스트 (실제 저장은 하지 않음)
        anim = animation.FuncAnimation(fig, animate, frames=10, interval=100, blit=True)
        
        plt.close(fig)
        print("✅ Matplotlib 애니메이션 사용 가능")
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib 애니메이션 테스트 실패: {e}")
        return False

def test_video_pipeline_dependencies():
    """비디오 파이프라인 의존성 전체 테스트"""
    print("🔧 비디오 파이프라인 의존성 테스트 중...")
    
    try:
        # 프로젝트 모듈 import 테스트
        sys.path.insert(0, str(Path(__file__).parent / "src" / "core"))
        
        from video_utils import check_video_dependencies
        from video_pipeline import VideoRenderingPipeline, PipelineConfig
        
        # 의존성 확인
        deps = check_video_dependencies()
        print("의존성 상태:")
        for name, version in deps.items():
            status = "✅" if version else "❌"
            print(f"  {status} {name}: {version}")
        
        # 파이프라인 구성 테스트
        config = PipelineConfig(
            fps=10,
            duration_seconds=5,
            resolution=(320, 240)
        )
        
        pipeline = VideoRenderingPipeline(config)
        print("✅ 비디오 파이프라인 초기화 성공")
        return True
        
    except Exception as e:
        print(f"❌ 비디오 파이프라인 의존성 테스트 실패: {e}")
        return False

def apply_video_patches():
    """비디오 관련 패치 적용"""
    print("🔨 비디오 관련 패치 적용 중...")
    
    patches_applied = []
    
    # 1. 임시 디렉토리 생성
    try:
        os.makedirs("videos/temp", exist_ok=True)
        patches_applied.append("임시 디렉토리 생성")
    except Exception as e:
        print(f"❌ 임시 디렉토리 생성 실패: {e}")
    
    # 2. 환경 변수 설정 (matplotlib 백엔드)
    try:
        os.environ['MPLBACKEND'] = 'Agg'
        patches_applied.append("Matplotlib 백엔드 설정")
    except Exception as e:
        print(f"❌ 환경 변수 설정 실패: {e}")
    
    # 3. OpenCV 경고 억제
    try:
        import cv2
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        patches_applied.append("OpenCV 경고 억제")
    except Exception:
        pass
    
    if patches_applied:
        print(f"✅ 적용된 패치: {patches_applied}")
        return True
    else:
        print("❌ 적용된 패치 없음")
        return False

def generate_compatibility_report():
    """호환성 리포트 생성"""
    print("📝 호환성 리포트 생성 중...")
    
    report = []
    report.append("# 비디오 시스템 호환성 리포트\n")
    report.append(f"생성 시간: {sys.version}\n")
    
    # FFmpeg 테스트
    ffmpeg_available = test_ffmpeg_availability()
    report.append(f"FFmpeg 사용 가능: {'예' if ffmpeg_available else '아니오'}\n")
    
    # OpenCV 테스트
    opencv_result, codecs = test_opencv_video_codec()
    report.append(f"OpenCV 비디오 코덱: {'사용 가능' if opencv_result else '사용 불가'}\n")
    if codecs:
        report.append(f"사용 가능한 코덱: {', '.join(codecs)}\n")
    
    # Matplotlib 테스트
    matplotlib_available = test_matplotlib_animation()
    report.append(f"Matplotlib 애니메이션: {'사용 가능' if matplotlib_available else '사용 불가'}\n")
    
    # 의존성 테스트
    deps_available = test_video_pipeline_dependencies()
    report.append(f"비디오 파이프라인: {'사용 가능' if deps_available else '사용 불가'}\n")
    
    # 권장사항
    report.append("\n## 권장사항\n")
    if not ffmpeg_available:
        report.append("- FFmpeg 설치 권장 (더 나은 비디오 품질)\n")
    if not opencv_result:
        report.append("- OpenCV 재설치 필요\n")
    if not matplotlib_available:
        report.append("- Matplotlib 설치 확인 필요\n")
    
    # 리포트 저장
    try:
        with open("videos/compatibility_report.md", "w", encoding="utf-8") as f:
            f.writelines(report)
        print("✅ 호환성 리포트 저장됨: videos/compatibility_report.md")
        return True
    except Exception as e:
        print(f"❌ 리포트 저장 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🔧 비디오 패치 및 호환성 테스트")
    print("=" * 50)
    
    results = []
    
    # 1. 패치 적용
    results.append(apply_video_patches())
    
    # 2. FFmpeg 테스트
    results.append(test_ffmpeg_availability())
    
    # 3. OpenCV 테스트
    opencv_result, _ = test_opencv_video_codec()
    results.append(opencv_result)
    
    # 4. Matplotlib 테스트
    results.append(test_matplotlib_animation())
    
    # 5. 파이프라인 테스트
    results.append(test_video_pipeline_dependencies())
    
    # 6. 리포트 생성
    results.append(generate_compatibility_report())
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("🎯 테스트 결과:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"성공: {passed}/{total}")
    
    if passed >= total * 0.8:  # 80% 이상 성공시 OK
        print("✅ 비디오 시스템 호환성 양호")
        return 0
    else:
        print("⚠️  비디오 시스템 호환성 문제 있음. 리포트를 확인하세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)