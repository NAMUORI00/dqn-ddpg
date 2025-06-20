#!/usr/bin/env python3
"""
모델 저장 구조 확인 스크립트
실제 저장된 모델들의 위치와 구조를 확인합니다.
"""

import os
import json
from pathlib import Path


def check_model_directories():
    """모델 디렉토리 구조 확인"""
    print("📁 모델 저장 구조 확인")
    print("=" * 50)
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ models/ 디렉토리가 존재하지 않습니다")
        print("💡 아직 학습을 실행하지 않았습니다")
        return False
    
    print("✅ models/ 디렉토리 존재")
    
    # 알고리즘별 확인
    algorithms = ['dqn', 'ddpg']
    environments = {
        'dqn': 'cartpole-v1',
        'ddpg': 'pendulum-v1'
    }
    
    found_models = False
    
    for algorithm in algorithms:
        env_name = environments[algorithm]
        alg_dir = models_dir / algorithm
        env_dir = models_dir / algorithm / env_name
        
        print(f"\n🤖 {algorithm.upper()} 모델 확인:")
        
        if env_dir.exists():
            print(f"  ✅ {env_dir} 존재")
            
            # 체크포인트 파일들 확인
            checkpoint_files = list(env_dir.glob('episode_*.pth'))
            if checkpoint_files:
                print(f"  📦 저장된 모델: {len(checkpoint_files)}개")
                
                # 크기 계산
                total_size = sum(f.stat().st_size for f in checkpoint_files)
                total_size_mb = total_size / (1024 * 1024)
                print(f"  💾 총 크기: {total_size_mb:.2f}MB")
                
                # 최신/오래된 모델
                if len(checkpoint_files) > 0:
                    episode_nums = []
                    for f in checkpoint_files:
                        try:
                            episode_num = int(f.stem.split('_')[1])
                            episode_nums.append(episode_num)
                        except:
                            pass
                    
                    if episode_nums:
                        episode_nums.sort()
                        print(f"  📊 에피소드 범위: {min(episode_nums)} ~ {max(episode_nums)}")
                
                found_models = True
            else:
                print(f"  ⚠️ 체크포인트 파일이 없습니다")
            
            # 메타데이터 확인
            metadata_file = env_dir / "checkpoint_metadata.json"
            if metadata_file.exists():
                print(f"  ✅ 메타데이터 파일 존재")
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"     - 총 저장된 모델: {metadata.get('total_saved', 0)}")
                    print(f"     - 환경: {metadata.get('environment', 'unknown')}")
                except:
                    print(f"  ⚠️ 메타데이터 파일 읽기 실패")
            else:
                print(f"  ❌ 메타데이터 파일 없음")
                
        elif alg_dir.exists():
            print(f"  ⚠️ {alg_dir} 존재하지만 환경별 디렉토리가 없습니다")
            print(f"     예상 위치: {env_dir}")
        else:
            print(f"  ❌ {alg_dir} 디렉토리가 없습니다")
    
    return found_models


def show_expected_structure():
    """예상되는 저장 구조 표시"""
    print("\n📋 예상되는 모델 저장 구조:")
    print("=" * 50)
    
    structure = """
models/
├── dqn/
│   └── cartpole-v1/
│       ├── episode_0001.pth
│       ├── episode_0002.pth
│       ├── episode_0003.pth
│       ├── ...
│       ├── checkpoint_metadata.json
│       └── training_log.json
└── ddpg/
    └── pendulum-v1/
        ├── episode_0001.pth
        ├── episode_0002.pth
        ├── episode_0003.pth
        ├── ...
        ├── checkpoint_metadata.json
        └── training_log.json
    """
    print(structure)


def show_usage_instructions():
    """사용 방법 안내"""
    print("\n🚀 학습 실행 방법:")
    print("=" * 50)
    
    print("1. DQN 학습 (CartPole-v1 환경):")
    print("   python scripts/run_convergence_training.py --algorithm dqn")
    
    print("\n2. DDPG 학습 (Pendulum-v1 환경):")
    print("   python scripts/run_convergence_training.py --algorithm ddpg")
    
    print("\n3. 두 알고리즘 연속 학습:")
    print("   python scripts/run_complete_training.py --episodes 1000")
    
    print("\n💡 각 학습은 수렴할 때까지 모든 에피소드의 모델을 저장합니다!")


def main():
    """메인 실행"""
    print("🔍 DQN vs DDPG 모델 저장 구조 확인")
    print("=" * 60)
    
    # 현재 모델 상태 확인
    has_models = check_model_directories()
    
    # 예상 구조 표시
    show_expected_structure()
    
    if not has_models:
        # 사용법 안내
        show_usage_instructions()
    else:
        print("\n✅ 저장된 모델을 발견했습니다!")
        print("📊 분석 및 벤치마킹을 실행할 수 있습니다.")


if __name__ == "__main__":
    main()