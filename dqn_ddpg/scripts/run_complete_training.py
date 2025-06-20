#!/usr/bin/env python3
"""
완전학습 실행 스크립트
DQN과 DDPG의 완전학습을 순차적으로 실행하고 종합 분석을 수행합니다.
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
from src.analysis.training_analyzer import TrainingAnalyzer
from src.analysis.model_benchmarks import ModelBenchmarkSystem
from src.core.episode_checkpoint import EpisodeCheckpointManager


def run_dqn_training(episodes: int, gpu_config, resume_from: int = None):
    """DQN 완전학습 실행"""
    print("\n" + "="*80)
    print("🎯 DQN 완전학습 시작")
    print("="*80)
    
    pipeline = CompleteTrainingPipeline(
        config_path="configs/dqn_config.yaml",
        algorithm="dqn",
        gpu_config=gpu_config
    )
    
    pipeline.train(num_episodes=episodes, resume_from=resume_from)
    return pipeline


def run_ddpg_training(episodes: int, gpu_config, resume_from: int = None):
    """DDPG 완전학습 실행"""
    print("\n" + "="*80)
    print("🎯 DDPG 완전학습 시작")
    print("="*80)
    
    pipeline = CompleteTrainingPipeline(
        config_path="configs/ddpg_config.yaml",
        algorithm="ddpg",
        gpu_config=gpu_config
    )
    
    pipeline.train(num_episodes=episodes, resume_from=resume_from)
    return pipeline


def run_benchmarks(algorithms: list):
    """모델 벤치마킹 실행"""
    print("\n" + "="*80)
    print("🔬 모델 성능 벤치마킹 시작")
    print("="*80)
    
    benchmark_results = {}
    
    for algorithm in algorithms:
        print(f"\n📊 {algorithm.upper()} 모델 벤치마킹...")
        
        # 환경 이름 결정 (알고리즘별 기본값)
        default_envs = {
            'dqn': 'cartpole-v1',
            'ddpg': 'pendulum-v1'
        }
        env_name = default_envs.get(algorithm, 'unknown')
        
        # 체크포인트 매니저 로드
        checkpoint_manager = EpisodeCheckpointManager(
            base_path="models",
            algorithm=algorithm,
            environment=env_name
        )
        
        if not checkpoint_manager.checkpoints:
            print(f"⚠️ {algorithm} 체크포인트를 찾을 수 없습니다")
            continue
        
        # 벤치마크 시스템 생성
        benchmark_system = ModelBenchmarkSystem(
            checkpoint_manager=checkpoint_manager,
            num_eval_episodes=30,  # 빠른 평가를 위해 30에피소드
            eval_deterministic=True
        )
        
        # 설정 로드
        import yaml
        config_path = f"configs/{algorithm}_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 최근 10개 모델만 평가 (시간 절약)
        results = benchmark_system.evaluate_all_checkpoints(config, max_models=10)
        benchmark_results[algorithm] = results
        
        # 벤치마크 리포트 생성
        if results:
            benchmark_system.generate_benchmark_report()
    
    return benchmark_results


def run_comprehensive_analysis():
    """종합 분석 수행"""
    print("\n" + "="*80)
    print("📊 종합 분석 시작")
    print("="*80)
    
    analyzer = TrainingAnalyzer()
    
    # 종합 리포트 생성
    report_path = analyzer.generate_comprehensive_report()
    
    print(f"\n✅ 종합 리포트 생성 완료: {report_path}")
    return report_path


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="DQN vs DDPG 완전학습 및 분석 파이프라인")
    
    parser.add_argument('--episodes', type=int, default=10000,
                       help='각 알고리즘의 최대 학습 에피소드 수 (수렴 시 조기 종료, default: 10000)')
    parser.add_argument('--algorithms', nargs='+', choices=['dqn', 'ddpg'], default=['dqn', 'ddpg'],
                       help='학습할 알고리즘 (default: dqn ddpg)')
    parser.add_argument('--skip-training', action='store_true',
                       help='학습을 건너뛰고 분석만 수행')
    parser.add_argument('--skip-benchmarks', action='store_true',
                       help='벤치마킹을 건너뛰기')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='종합 분석을 건너뛰기')
    parser.add_argument('--resume-dqn', type=int, default=None,
                       help='DQN 재시작 에피소드 ID')
    parser.add_argument('--resume-ddpg', type=int, default=None,
                       help='DDPG 재시작 에피소드 ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (default: 42)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='사용할 GPU ID')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 사용 비활성화')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 시작 시간 기록
    total_start_time = time.time()
    
    print("🚀 DQN vs DDPG 완전학습 및 분석 파이프라인 시작")
    print(f"📊 학습 에피소드: {args.episodes}")
    print(f"🤖 알고리즘: {', '.join(args.algorithms)}")
    print(f"🎲 시드: {args.seed}")
    
    # GPU 설정 로드
    gpu_config = None
    if not args.skip_training:
        # 첫 번째 알고리즘의 설정으로 GPU 설정 로드
        first_algorithm = args.algorithms[0] if args.algorithms else 'dqn'
        config_path = f"configs/{first_algorithm}_config.yaml"
        
        gpu_config = load_gpu_config(
            config_path=config_path,
            device_id=args.gpu_id,
            force_cpu=args.no_gpu
        )
        print(f"🔧 GPU 설정: {gpu_config}")
    
    try:
        # 1. 학습 단계
        if not args.skip_training:
            for algorithm in args.algorithms:
                if algorithm == 'dqn':
                    run_dqn_training(args.episodes, gpu_config, args.resume_dqn)
                elif algorithm == 'ddpg':
                    run_ddpg_training(args.episodes, gpu_config, args.resume_ddpg)
                
                # 알고리즘 간 짧은 휴식
                time.sleep(2)
        else:
            print("⏭️ 학습 단계 건너뛰기")
        
        # 2. 벤치마킹 단계
        if not args.skip_benchmarks:
            benchmark_results = run_benchmarks(args.algorithms)
        else:
            print("⏭️ 벤치마킹 단계 건너뛰기")
        
        # 3. 종합 분석 단계
        if not args.skip_analysis:
            report_path = run_comprehensive_analysis()
        else:
            print("⏭️ 분석 단계 건너뛰기")
        
        # 완료 요약
        total_time = time.time() - total_start_time
        print("\n" + "="*80)
        print("🎉 모든 작업 완료!")
        print("="*80)
        print(f"⏰ 총 소요 시간: {total_time/3600:.2f}시간")
        
        # 결과 파일 위치 안내
        print("\n📁 생성된 파일들:")
        print("  📊 모델 체크포인트: models/")
        print("  📋 학습 로그: logs/")
        print("  📈 벤치마크 결과: results/benchmarks/")
        print("  📄 분석 리포트: output/analysis/")
        
        if not args.skip_analysis:
            print(f"  🎯 종합 리포트: {report_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()