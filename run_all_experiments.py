#!/usr/bin/env python3
"""
모든 구현된 실험을 순차적으로 실행하는 스크립트
빠른 테스트를 위해 축소된 에피소드로 실행
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# 프로젝트 루트 디렉토리
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

def print_header(message):
    """헤더 출력"""
    print("\n" + "="*60)
    print(f" {message}")
    print("="*60 + "\n")

def run_command(command, description, timeout=600):
    """명령 실행 및 결과 출력"""
    print(f"🚀 {description}")
    print(f"   명령: {command}")
    print(f"   시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if result.stdout:
                print(f"   출력:\n{result.stdout[:500]}...")  # 처음 500자만 출력
        else:
            print(f"   ❌ 실패: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if result.stderr:
                print(f"   에러:\n{result.stderr[:500]}...")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ 타임아웃 ({timeout}초)")
        return False
    except Exception as e:
        print(f"   ❌ 예외 발생: {str(e)}")
        return False

def main():
    print_header("DQN vs DDPG 전체 실험 실행 시작")
    print(f"프로젝트 디렉토리: {project_root}")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/experiments_log", exist_ok=True)
    
    # 실험 목록
    experiments = [
        {
            "name": "1. 간단한 데모 실행",
            "command": "python tests/simple_demo.py",
            "timeout": 120
        },
        {
            "name": "2. 상세 테스트 실행",
            "command": "python tests/detailed_test.py",
            "timeout": 180
        },
        {
            "name": "3. 빠른 학습 실험 (축소 버전)",
            "command": "python simple_training.py",
            "timeout": 300
        },
        {
            "name": "4. 결정적 정책 분석",
            "command": "python experiments/analyze_deterministic_policy.py --results-dir results",
            "timeout": 180
        },
        {
            "name": "5. 종합 비교 리포트 생성",
            "command": "python experiments/generate_comparison_report.py --results-dir results",
            "timeout": 120
        },
        {
            "name": "6. 학습 비디오 렌더링 (샘플 데이터)",
            "command": "python render_learning_video.py --sample-data --learning-only --duration 10",
            "timeout": 180
        },
        {
            "name": "7. 비교 비디오 생성",
            "command": "python create_comparison_video.py --auto --episodes 50",
            "timeout": 300
        }
    ]
    
    # 실험별 결과 저장
    results = []
    successful = 0
    failed = 0
    
    # 각 실험 실행
    for i, exp in enumerate(experiments):
        print_header(f"실험 {i+1}/{len(experiments)}: {exp['name']}")
        
        success = run_command(
            exp['command'], 
            exp['name'], 
            exp.get('timeout', 300)
        )
        
        results.append({
            'name': exp['name'],
            'command': exp['command'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # 실험 간 대기
        time.sleep(2)
    
    # 결과 요약
    print_header("실험 실행 결과 요약")
    print(f"총 실험 수: {len(experiments)}")
    print(f"성공: {successful}")
    print(f"실패: {failed}")
    print(f"성공률: {successful/len(experiments)*100:.1f}%")
    
    print("\n실험별 결과:")
    for i, result in enumerate(results):
        status = "✅" if result['success'] else "❌"
        print(f"  {i+1}. {status} {result['name']}")
    
    # 결과 로그 저장
    log_file = f"results/experiments_log/run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("DQN vs DDPG 전체 실험 실행 로그\n")
        f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"프로젝트 디렉토리: {project_root}\n\n")
        
        f.write("실험 결과 요약\n")
        f.write(f"총 실험 수: {len(experiments)}\n")
        f.write(f"성공: {successful}\n")
        f.write(f"실패: {failed}\n")
        f.write(f"성공률: {successful/len(experiments)*100:.1f}%\n\n")
        
        f.write("실험별 상세 결과\n")
        for i, result in enumerate(results):
            f.write(f"\n{i+1}. {result['name']}\n")
            f.write(f"   명령: {result['command']}\n")
            f.write(f"   결과: {'성공' if result['success'] else '실패'}\n")
            f.write(f"   시간: {result['timestamp']}\n")
    
    print(f"\n📝 실행 로그 저장: {log_file}")
    
    # 최종 안내
    print_header("실험 완료!")
    print("생성된 결과물 확인:")
    print("  • 학습 결과: results/dqn_results.json, results/ddpg_results.json")
    print("  • 결정적 정책 분석: results/deterministic_analysis/")
    print("  • 비교 분석 리포트: results/comparison_report/")
    print("  • 비디오 파일: videos/")
    print("  • 실행 로그: results/experiments_log/")

if __name__ == "__main__":
    main()