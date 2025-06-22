#!/usr/bin/env python3
"""
나머지 조합들을 순차적으로 실행하는 스크립트
"""

import subprocess
import time
import os
import sys

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
    print(f"⏳ {algorithm} + {environment} 학습 중...")
    process.wait()
    
    if process.returncode == 0:
        print(f"✅ {algorithm} + {environment} 학습 완료!")
    else:
        print(f"❌ {algorithm} + {environment} 학습 실패 (exit code: {process.returncode})")
    
    return process.returncode == 0

def main():
    """나머지 조합들을 순차 실행"""
    
    remaining_combinations = [
        ("DQN", "Pendulum-v1", "training_dqn_pendulum.log"),
        ("DDPG", "Pendulum-v1", "training_ddpg_pendulum.log")
    ]
    
    print("🎯 나머지 조합들 순차 실행")
    print(f"📊 총 {len(remaining_combinations)}개 조합")
    
    for i, (algorithm, environment, log_name) in enumerate(remaining_combinations, 1):
        print(f"\n{'='*60}")
        print(f"진행: {i}/{len(remaining_combinations)} - {algorithm} + {environment}")
        print(f"{'='*60}")
        
        success = run_combination(algorithm, environment, log_name)
        
        if not success:
            print(f"❌ {algorithm} + {environment} 실패로 인해 중단")
            break
        
        if i < len(remaining_combinations):
            print("⏱️  다음 조합까지 5초 대기...")
            time.sleep(5)
    
    print(f"\n🎉 모든 나머지 조합 실행 완료!")

if __name__ == "__main__":
    main()