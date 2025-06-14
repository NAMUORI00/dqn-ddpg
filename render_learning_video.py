#!/usr/bin/env python3
"""
전체 학습 과정 시각화 비디오 생성 스크립트
DQN과 DDPG의 학습 과정을 종합적으로 시각화하여 비디오로 출력합니다.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import yaml
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

# 비디오 파이프라인 직접 임포트 (torch 의존성 회피)
sys.path.insert(0, str(Path(__file__).parent / "src" / "core"))
from video_pipeline import VideoRenderingPipeline, PipelineConfig


def check_dependencies():
    """필요한 의존성 확인"""
    missing_deps = []
    
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import cv2
        print(f"✅ opencv-python: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    # ffmpeg 확인 (선택사항으로 변경)
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ffmpeg: 설치됨")
        else:
            print("⚠️  ffmpeg: 설치되지 않음 (비디오 연결 기능 제한)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  ffmpeg: 설치되지 않음 (비디오 연결 기능 제한)")
    
    if missing_deps:
        print(f"\n❌ 누락된 의존성: {', '.join(missing_deps)}")
        print("\n설치 방법:")
        print("pip install matplotlib opencv-python numpy")
        print("ffmpeg 설치: https://ffmpeg.org/download.html")
        return False
    
    print("\n✅ 모든 의존성이 확인되었습니다.")
    return True


def find_result_files(results_dir: str = "results"):
    """결과 파일들 자동 검색"""
    results_path = Path(results_dir)
    
    dqn_results = None
    ddpg_results = None
    
    # JSON 결과 파일 검색
    if results_path.exists():
        json_files = list(results_path.glob("*.json"))
        
        for json_file in json_files:
            if "dqn" in json_file.name.lower():
                dqn_results = str(json_file)
            elif "ddpg" in json_file.name.lower():
                ddpg_results = str(json_file)
    
    return dqn_results, ddpg_results


def create_sample_data(output_dir: str = "results"):
    """샘플 데이터 생성 (결과 파일이 없을 때)"""
    print("[INFO] 결과 파일이 없어 샘플 데이터를 생성합니다.")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    import numpy as np
    np.random.seed(42)
    
    # DQN 샘플 데이터
    dqn_episodes = 500
    dqn_rewards = []
    for i in range(dqn_episodes):
        base_reward = min(450, 50 + i * 0.8)
        noise = np.random.normal(0, 25)
        reward = max(0, base_reward + noise)
        dqn_rewards.append(reward)
    
    dqn_data = {
        'config': {
            'algorithm': 'DQN',
            'environment': 'CartPole-v1',
            'episodes': dqn_episodes,
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'metrics': {
            'episode_rewards': dqn_rewards,
            'episode_lengths': [np.random.randint(50, 500) for _ in range(dqn_episodes)],
            'training_losses': [np.random.exponential(0.1) for _ in range(dqn_episodes * 5)],
            'q_values': [np.random.normal(100, 20) for _ in range(dqn_episodes * 5)]
        },
        'final_evaluation': {
            'mean_reward': np.mean(dqn_rewards[-100:]),
            'std_reward': np.std(dqn_rewards[-100:]),
            'success_rate': 0.95
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # DDPG 샘플 데이터
    ddpg_episodes = 400
    ddpg_rewards = []
    for i in range(ddpg_episodes):
        base_reward = max(-200, -800 + i * 1.5)
        noise = np.random.normal(0, 50)
        reward = min(0, base_reward + noise)
        ddpg_rewards.append(reward)
    
    ddpg_data = {
        'config': {
            'algorithm': 'DDPG',
            'environment': 'Pendulum-v1',
            'episodes': ddpg_episodes,
            'actor_lr': 0.001,
            'critic_lr': 0.002,
            'batch_size': 64
        },
        'metrics': {
            'episode_rewards': ddpg_rewards,
            'episode_lengths': [200 for _ in range(ddpg_episodes)],  # Pendulum은 고정 길이
            'training_losses': [np.random.exponential(0.05) for _ in range(ddpg_episodes * 5)],
            'q_values': [np.random.normal(-300, 100) for _ in range(ddpg_episodes * 5)]
        },
        'final_evaluation': {
            'mean_reward': np.mean(ddpg_rewards[-100:]),
            'std_reward': np.std(ddpg_rewards[-100:]),
            'success_rate': 0.80
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 파일 저장
    dqn_path = output_path / "dqn_results.json"
    ddpg_path = output_path / "ddpg_results.json"
    
    with open(dqn_path, 'w') as f:
        json.dump(dqn_data, f, indent=2)
    
    with open(ddpg_path, 'w') as f:
        json.dump(ddpg_data, f, indent=2)
    
    print(f"[INFO] 샘플 데이터 생성 완료:")
    print(f"  - DQN: {dqn_path}")
    print(f"  - DDPG: {ddpg_path}")
    
    return str(dqn_path), str(ddpg_path)


def main():
    parser = argparse.ArgumentParser(description="학습 과정 시각화 비디오 생성")
    
    parser.add_argument("--dqn-results", type=str, 
                       help="DQN 결과 JSON 파일 경로")
    parser.add_argument("--ddpg-results", type=str,
                       help="DDPG 결과 JSON 파일 경로")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="결과 파일 디렉토리 (자동 검색용)")
    parser.add_argument("--output-dir", type=str, default="videos/pipeline",
                       help="비디오 출력 디렉토리")
    parser.add_argument("--config", type=str,
                       help="파이프라인 설정 YAML 파일")
    
    # 비디오 설정
    parser.add_argument("--duration", type=int, default=180,
                       help="비디오 길이 (초)")
    parser.add_argument("--fps", type=int, default=30,
                       help="비디오 FPS")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       help="비디오 해상도 (예: 1920x1080)")
    
    # 생성할 비디오 타입
    parser.add_argument("--learning-only", action="store_true",
                       help="학습 과정 애니메이션만 생성")
    parser.add_argument("--comparison-only", action="store_true",
                       help="비교 비디오만 생성")
    parser.add_argument("--summary-only", action="store_true",
                       help="요약 비디오만 생성")
    parser.add_argument("--all", action="store_true", default=True,
                       help="모든 비디오 생성 (기본값)")
    
    parser.add_argument("--sample-data", action="store_true",
                       help="샘플 데이터로 테스트")
    parser.add_argument("--check-deps", action="store_true",
                       help="의존성 확인만 수행")
    
    args = parser.parse_args()
    
    # 의존성 확인
    if args.check_deps or not check_dependencies():
        return
    
    print("=" * 60)
    print("🎬 DQN vs DDPG 학습 과정 시각화 비디오 생성기")
    print("=" * 60)
    
    # 결과 파일 찾기 또는 생성
    if args.sample_data:
        dqn_results, ddpg_results = create_sample_data(args.results_dir)
    else:
        dqn_results = args.dqn_results
        ddpg_results = args.ddpg_results
        
        if not dqn_results or not ddpg_results:
            print("[INFO] 결과 파일 자동 검색 중...")
            auto_dqn, auto_ddpg = find_result_files(args.results_dir)
            
            dqn_results = dqn_results or auto_dqn
            ddpg_results = ddpg_results or auto_ddpg
        
        # 여전히 파일이 없으면 샘플 데이터 생성
        if not dqn_results or not ddpg_results or \
           not os.path.exists(dqn_results) or not os.path.exists(ddpg_results):
            print("[WARNING] 결과 파일을 찾을 수 없습니다. 샘플 데이터를 생성합니다.")
            dqn_results, ddpg_results = create_sample_data(args.results_dir)
    
    print(f"📊 DQN 결과: {dqn_results}")
    print(f"📊 DDPG 결과: {ddpg_results}")
    
    # 파이프라인 설정
    if args.config and os.path.exists(args.config):
        config = PipelineConfig.from_yaml(args.config)
    else:
        # 명령행 인자로 설정 생성
        width, height = map(int, args.resolution.split('x'))
        config = PipelineConfig(
            output_dir=args.output_dir,
            fps=args.fps,
            duration_seconds=args.duration,
            resolution=(width, height)
        )
    
    print(f"⚙️  설정: {config.fps}fps, {config.resolution[0]}x{config.resolution[1]}, {config.duration_seconds}초")
    
    # 파이프라인 생성
    pipeline = VideoRenderingPipeline(config)
    
    try:
        # 데이터 로드
        print("\n📥 학습 데이터 로드 중...")
        pipeline.load_training_data(dqn_results, ddpg_results)
        
        # 비디오 생성
        print("\n🎬 비디오 생성 시작...")
        
        if args.learning_only:
            print("📹 학습 과정 애니메이션 생성 중...")
            result = pipeline.create_learning_animation()
            print(f"✅ 완료: {result}")
            
        elif args.comparison_only:
            print("📹 알고리즘 비교 비디오 생성 중...")
            result = pipeline.create_comparison_video()
            print(f"✅ 완료: {result}")
            
        elif args.summary_only:
            print("📹 요약 비디오 생성 중...")
            result = pipeline.create_summary_video()
            print(f"✅ 완료: {result}")
            
        else:  # 전체 파이프라인
            print("📹 전체 파이프라인 실행 중...")
            result = pipeline.run_full_pipeline(dqn_results, ddpg_results)
            print(f"✅ 메인 비디오 완료: {result}")
        
        print("\n🎉 비디오 생성이 완료되었습니다!")
        print(f"📁 출력 디렉토리: {config.output_dir}")
        
        # 생성된 파일 목록 표시
        output_path = Path(config.output_dir)
        if output_path.exists():
            video_files = list(output_path.glob("*.mp4"))
            if video_files:
                print("\n📹 생성된 비디오 파일:")
                for video_file in video_files:
                    file_size = video_file.stat().st_size / (1024 * 1024)  # MB
                    print(f"  - {video_file.name} ({file_size:.1f} MB)")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)