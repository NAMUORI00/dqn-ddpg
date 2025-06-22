#!/usr/bin/env python3
"""
백그라운드 학습 진행상황 체크 스크립트
"""

import subprocess
import time
import os
from pathlib import Path

def check_processes():
    """실행 중인 프로세스 확인"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.split('\n')
        training_processes = [
            line for line in lines 
            if 'unified_training_system.py' in line and 'grep' not in line
        ]
        
        return training_processes
    except subprocess.CalledProcessError:
        return []

def check_results_structure():
    """결과 디렉토리 구조 확인"""
    results_dir = Path("results")
    if not results_dir.exists():
        return "results/ 디렉토리가 아직 생성되지 않았습니다."
    
    structure = []
    for algorithm in ["DQN", "DDPG"]:
        alg_dir = results_dir / algorithm
        if alg_dir.exists():
            structure.append(f"  {algorithm}/")
            for env_dir in alg_dir.iterdir():
                if env_dir.is_dir():
                    structure.append(f"    {env_dir.name}/")
                    # 최신 타임스탬프 디렉토리 찾기
                    timestamp_dirs = [d for d in env_dir.iterdir() if d.is_dir()]
                    if timestamp_dirs:
                        latest_dir = max(timestamp_dirs, key=lambda x: x.name)
                        structure.append(f"      {latest_dir.name}/")
                        
                        # 체크포인트 수 확인
                        checkpoints_dir = latest_dir / "checkpoints"
                        if checkpoints_dir.exists():
                            checkpoint_count = len(list(checkpoints_dir.glob("*.pth")))
                            structure.append(f"        checkpoints/ ({checkpoint_count}개)")
                        
                        # 결과 파일 확인
                        result_file = latest_dir / "training_results.json"
                        if result_file.exists():
                            structure.append(f"        training_results.json ✅")
    
    return "\n".join(structure) if structure else "아직 결과가 생성되지 않았습니다."

def main():
    print("🔍 공정한 비교 학습 진행상황 체크")
    print("=" * 60)
    
    # 실행 중인 프로세스 확인
    processes = check_processes()
    print(f"📊 실행 중인 프로세스: {len(processes)}개")
    
    if processes:
        print("\n🚀 실행 중인 학습:")
        for i, process in enumerate(processes, 1):
            parts = process.split()
            if len(parts) >= 11:
                # PID와 명령어 추출
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                
                # 알고리즘과 환경 추출
                cmd_parts = process.split()
                algorithm = None
                environment = None
                
                for j, part in enumerate(cmd_parts):
                    if part == "--algorithm" and j + 1 < len(cmd_parts):
                        algorithm = cmd_parts[j + 1]
                    elif part == "--environment" and j + 1 < len(cmd_parts):
                        environment = cmd_parts[j + 1]
                
                print(f"  {i}. {algorithm} + {environment}")
                print(f"     PID: {pid}, CPU: {cpu}%, MEM: {mem}%")
    else:
        print("❌ 실행 중인 학습 프로세스가 없습니다.")
    
    # 로그 파일 확인
    print(f"\n📝 로그 파일:")
    log_files = [
        "dqn_cartpole_fair.log",
        "ddpg_cartpole_fair.log", 
        "dqn_pendulum_fair.log",
        "ddpg_pendulum_fair.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"  {log_file}: {size} bytes")
        else:
            print(f"  {log_file}: 없음")
    
    # 결과 디렉토리 구조 확인
    print(f"\n📁 결과 디렉토리 구조:")
    structure = check_results_structure()
    print(structure)
    
    print(f"\n⏰ 체크 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()