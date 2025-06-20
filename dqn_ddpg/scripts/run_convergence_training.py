#!/usr/bin/env python3
"""
수렴까지 완전학습 실행 스크립트
최대한 많은 에피소드를 저장하고 수렴할 때까지 학습을 진행합니다.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.complete_training import CompleteTrainingPipeline
from src.core.gpu_config import load_gpu_config
from src.core.utils import set_seed


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="수렴까지 완전학습 (모든 에피소드 저장)")
    
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ddpg'], required=True,
                       help='학습할 알고리즘')
    parser.add_argument('--max-episodes', type=int, default=20000,
                       help='최대 학습 에피소드 수 (default: 20000)')
    parser.add_argument('--resume', type=int, default=None,
                       help='재시작할 에피소드 ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (default: 42)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='사용할 GPU ID')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 사용 비활성화')
    
    # 수렴 조건 커스터마이징
    parser.add_argument('--cv-threshold', type=float, default=0.05,
                       help='변동계수 임계값 (default: 0.05)')
    parser.add_argument('--improvement-threshold', type=float, default=0.01,
                       help='성능 개선 임계값 (default: 0.01)')
    parser.add_argument('--patience', type=int, default=150,
                       help='정체 허용 기간 (default: 150)')
    parser.add_argument('--confidence', type=float, default=0.85,
                       help='수렴 신뢰도 임계값 (default: 0.85)')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    print("🚀 수렴까지 완전학습 시작")
    print(f"🤖 알고리즘: {args.algorithm.upper()}")
    print(f"📊 최대 에피소드: {args.max_episodes:,}")
    print(f"🎯 수렴 조건:")
    print(f"   - CV 임계값: {args.cv_threshold}")
    print(f"   - 개선 임계값: {args.improvement_threshold}")
    print(f"   - 정체 허용: {args.patience}에피소드")
    print(f"   - 신뢰도: {args.confidence}")
    print(f"💾 저장 정책: 모든 에피소드 저장")
    print("=" * 80)
    
    # GPU 설정 로드
    config_path = f"configs/{args.algorithm}_config.yaml"
    gpu_config = load_gpu_config(
        config_path=config_path,
        device_id=args.gpu_id,
        force_cpu=args.no_gpu
    )
    print(f"🔧 GPU 설정: {gpu_config}")
    
    # 학습 파이프라인 생성
    pipeline = CompleteTrainingPipeline(
        config_path=config_path,
        algorithm=args.algorithm,
        gpu_config=gpu_config
    )
    
    # 수렴 조건 커스터마이징
    pipeline.convergence_detector.cv_threshold = args.cv_threshold
    pipeline.convergence_detector.improvement_threshold = args.improvement_threshold
    pipeline.convergence_detector.plateau_patience = args.patience
    pipeline.convergence_detector.confidence_threshold = args.confidence
    
    print(f"📊 수렴 감지기 설정:")
    print(f"   - 최소 에피소드: {pipeline.convergence_detector.min_episodes}")
    print(f"   - 안정성 윈도우: {pipeline.convergence_detector.stability_window}")
    print(f"   - CV 임계값: {pipeline.convergence_detector.cv_threshold}")
    print(f"   - 개선 임계값: {pipeline.convergence_detector.improvement_threshold}")
    print(f"   - 정체 허용: {pipeline.convergence_detector.plateau_patience}")
    print(f"   - 신뢰도 임계값: {pipeline.convergence_detector.confidence_threshold}")
    
    try:
        # 학습 시작
        start_time = time.time()
        
        pipeline.train(
            num_episodes=args.max_episodes,
            resume_from=args.resume
        )
        
        # 완료 메시지
        total_time = time.time() - start_time
        convergence_summary = pipeline.convergence_detector.get_convergence_summary()
        
        print("\n" + "="*80)
        print("🎉 학습 완료!")
        print("="*80)
        print(f"⏰ 총 학습 시간: {total_time/3600:.2f}시간")
        print(f"📊 총 에피소드: {convergence_summary['total_episodes']:,}")
        print(f"🎯 수렴 상태: {convergence_summary['status']}")
        print(f"📈 수렴 확률: {convergence_summary['convergence_probability']:.3f}")
        
        # 저장된 파일 정보
        # 환경 이름 가져오기
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        env_name = config['environment']['name'].lower()
        
        print(f"\n📁 결과 파일:")
        print(f"   모델: models/{args.algorithm}/{env_name}/")
        print(f"   로그: logs/")
        print(f"   결과: results/{args.algorithm}/")
        
        if convergence_summary['status'] == 'converged':
            print(f"\n✅ 성공적으로 수렴 완료!")
        else:
            print(f"\n⚠️ 최대 에피소드에 도달했지만 완전히 수렴하지 않았을 수 있습니다.")
            
        # 저장 공간 정보
        model_dir = Path(f"models/{args.algorithm}/{env_name}")
        if model_dir.exists():
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*.pth'))
            total_size_gb = total_size / (1024**3)
            print(f"💾 저장된 모델 크기: {total_size_gb:.2f}GB")
            
            model_count = len(list(model_dir.glob('episode_*.pth')))
            print(f"📦 저장된 모델 수: {model_count:,}개")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다")
        print("💾 현재까지의 모든 체크포인트가 저장되었습니다")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()