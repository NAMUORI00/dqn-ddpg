#!/usr/bin/env python3
"""
4가지 알고리즘-환경 조합을 모두 자동으로 실행하는 스크립트
기존 완료된 조합은 건너뛰고 나머지만 실행
"""

import subprocess
import time
import os
import sys
import json
from pathlib import Path

def check_combination_completed(algorithm, environment):
    """조합이 이미 완료되었는지 확인"""
    # results/cross_environment 디렉토리에서 완료된 조합 찾기
    results_dir = Path("results/cross_environment")
    if not results_dir.exists():
        return False
    
    combination_name = f"{algorithm}_{environment}"
    
    # 각 세션 디렉토리 확인
    for session_dir in results_dir.iterdir():
        if session_dir.is_dir():
            combination_dir = session_dir / combination_name
            if combination_dir.exists():
                # training_results.json 파일이 있는지 확인
                result_file = combination_dir / "training_results.json"
                if result_file.exists():
                    print(f"✅ {algorithm} + {environment} 이미 완료됨: {combination_dir}")
                    return True
    
    return False

def run_combination(algorithm, environment, log_name):
    """특정 조합을 실행하고 완료까지 대기"""
    cmd = [
        "./.conda/bin/python",
        "scripts/train_cross_environment.py",
        "--algorithm", algorithm,
        "--environment", environment
    ]
    
    print(f"\n🚀 {algorithm} + {environment} 학습 시작...")
    print(f"📝 로그 파일: {log_name}")
    
    # 로그 파일로 출력 리다이렉트
    with open(log_name, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    # 프로세스 완료까지 대기
    print(f"⏳ {algorithm} + {environment} 학습 중 (PID: {process.pid})...")
    
    # 주기적으로 진행상황 체크
    while process.poll() is None:
        time.sleep(60)  # 1분마다 체크
        print(f"⏱️  {algorithm} + {environment} 학습 계속 진행 중...")
    
    return_code = process.returncode
    
    if return_code == 0:
        print(f"✅ {algorithm} + {environment} 학습 완료!")
    else:
        print(f"❌ {algorithm} + {environment} 학습 실패 (exit code: {return_code})")
    
    return return_code == 0

def main():
    """4가지 조합을 모두 실행"""
    
    all_combinations = [
        ("DQN", "CartPole-v1", "training_dqn_cartpole.log"),
        ("DDPG", "CartPole-v1", "training_ddpg_cartpole.log"),
        ("DQN", "Pendulum-v1", "training_dqn_pendulum.log"),
        ("DDPG", "Pendulum-v1", "training_ddpg_pendulum.log")
    ]
    
    print("🎯 4가지 알고리즘-환경 조합 전체 실행")
    print(f"📊 총 {len(all_combinations)}개 조합")
    print("="*60)
    
    # 완료된 조합과 미완료 조합 구분
    pending_combinations = []
    for algorithm, environment, log_name in all_combinations:
        if check_combination_completed(algorithm, environment):
            continue
        else:
            pending_combinations.append((algorithm, environment, log_name))
            print(f"⏳ {algorithm} + {environment} 실행 예정")
    
    if not pending_combinations:
        print("\n🎉 모든 조합이 이미 완료되었습니다!")
        return
    
    print(f"\n📋 실행할 조합: {len(pending_combinations)}개")
    print("="*60)
    
    total_start_time = time.time()
    completed_count = 0
    
    for i, (algorithm, environment, log_name) in enumerate(pending_combinations, 1):
        print(f"\n{'='*60}")
        print(f"진행: {i}/{len(pending_combinations)} - {algorithm} + {environment}")
        print(f"완료된 조합: {len(all_combinations) - len(pending_combinations) + completed_count}")
        print(f"{'='*60}")
        
        success = run_combination(algorithm, environment, log_name)
        
        if success:
            completed_count += 1
            print(f"✅ {algorithm} + {environment} 성공!")
        else:
            print(f"❌ {algorithm} + {environment} 실패!")
            # 실패해도 다음 조합 계속 진행
        
        if i < len(pending_combinations):
            print("⏱️  다음 조합까지 10초 대기...")
            time.sleep(10)
    
    total_time = time.time() - total_start_time
    total_completed = len(all_combinations) - len(pending_combinations) + completed_count
    
    print(f"\n\n🎉 전체 실행 완료!")
    print(f"✅ 완료된 조합: {total_completed}/{len(all_combinations)}")
    print(f"⏰ 총 소요 시간: {total_time/60:.1f}분")
    print("="*60)
    
    # 최종 결과 요약
    print("\n📊 최종 결과 요약:")
    for algorithm, environment, _ in all_combinations:
        if check_combination_completed(algorithm, environment):
            print(f"✅ {algorithm} + {environment}")
        else:
            print(f"❌ {algorithm} + {environment}")

if __name__ == "__main__":
    main()