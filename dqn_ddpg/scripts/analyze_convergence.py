#!/usr/bin/env python3
"""
수렴 상태 상세 분석 스크립트
"""

import json
import numpy as np
from pathlib import Path

def analyze_convergence(result_file, algorithm, environment):
    """수렴 상태 상세 분석"""
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data['episode_rewards']
    final_avg_reward = data['final_avg_reward']
    best_avg_reward = data['best_avg_reward']
    
    # 수렴 기준 설정
    if environment == "CartPole-v1":
        convergence_threshold = 475.0
        target_performance = 500.0
    elif environment == "Pendulum-v1":
        convergence_threshold = -200.0
        target_performance = -50.0  # 더 엄격한 기준
    
    # 최근 100 에피소드 평균
    last_100_avg = np.mean(episode_rewards[-100:])
    
    # 최근 200 에피소드 평균
    last_200_avg = np.mean(episode_rewards[-200:])
    
    # 최근 500 에피소드 평균
    last_500_avg = np.mean(episode_rewards[-500:])
    
    # 안정성 분석 (최근 100 에피소드의 표준편차)
    last_100_std = np.std(episode_rewards[-100:])
    
    # 학습 곡선 기울기 (최근 200 에피소드)
    recent_episodes = np.arange(len(episode_rewards[-200:]))
    recent_rewards = episode_rewards[-200:]
    slope = np.polyfit(recent_episodes, recent_rewards, 1)[0]
    
    # 수렴 판정 (여러 기준)
    converged_basic = last_100_avg >= convergence_threshold
    converged_strict = last_100_avg >= target_performance * 0.95
    stable = last_100_std < (abs(target_performance) * 0.1)
    plateau = abs(slope) < 0.1
    
    print(f"\n🔍 {algorithm} + {environment} 상세 분석:")
    print(f"   📈 최종 평균 보상: {final_avg_reward:.2f}")
    print(f"   🏆 최고 평균 보상: {best_avg_reward:.2f}")
    print(f"   📊 최근 100 에피소드 평균: {last_100_avg:.2f}")
    print(f"   📊 최근 200 에피소드 평균: {last_200_avg:.2f}")
    print(f"   📊 최근 500 에피소드 평균: {last_500_avg:.2f}")
    print(f"   📉 최근 100 에피소드 표준편차: {last_100_std:.2f}")
    print(f"   📈 학습 곡선 기울기: {slope:.4f}")
    print(f"   🎯 수렴 기준: {convergence_threshold}")
    print(f"   🌟 목표 성능: {target_performance}")
    
    print(f"\n   ✅ 수렴 판정:")
    print(f"      기본 수렴: {'✅' if converged_basic else '❌'} ({last_100_avg:.2f} >= {convergence_threshold})")
    print(f"      엄격 수렴: {'✅' if converged_strict else '❌'} ({last_100_avg:.2f} >= {target_performance * 0.95:.1f})")
    print(f"      성능 안정: {'✅' if stable else '❌'} (표준편차 < {abs(target_performance) * 0.1:.1f})")
    print(f"      학습 정체: {'✅' if plateau else '❌'} (기울기 < 0.1)")
    
    overall_converged = converged_basic and stable
    print(f"   🏆 종합 수렴: {'✅' if overall_converged else '❌'}")
    
    # 성능 등급 평가
    if environment == "CartPole-v1":
        if last_100_avg >= 490:
            grade = "S (완벽)"
        elif last_100_avg >= 475:
            grade = "A (수렴)"
        elif last_100_avg >= 400:
            grade = "B (양호)"
        elif last_100_avg >= 200:
            grade = "C (보통)"
        else:
            grade = "D (부족)"
    else:  # Pendulum-v1
        if last_100_avg >= -50:
            grade = "S (완벽)"
        elif last_100_avg >= -100:
            grade = "A (우수)"
        elif last_100_avg >= -150:
            grade = "B (양호)"
        elif last_100_avg >= -200:
            grade = "C (보통)"
        else:
            grade = "D (부족)"
    
    print(f"   🎖️  성능 등급: {grade}")
    
    return {
        'algorithm': algorithm,
        'environment': environment,
        'final_avg_reward': final_avg_reward,
        'last_100_avg': last_100_avg,
        'last_200_avg': last_200_avg,
        'last_500_avg': last_500_avg,
        'std_last_100': last_100_std,
        'slope': slope,
        'converged_basic': converged_basic,
        'converged_strict': converged_strict,
        'stable': stable,
        'plateau': plateau,
        'overall_converged': overall_converged,
        'grade': grade
    }

def main():
    results_dir = Path("results/cross_environment")
    
    combinations = [
        ("20250621_231516", "DQN", "CartPole-v1"),
        ("20250621_235008", "DDPG", "CartPole-v1"),
        ("20250622_003219", "DQN", "Pendulum-v1"),
        ("20250622_004126", "DDPG", "Pendulum-v1")
    ]
    
    analysis_results = []
    
    print("🔬 수렴 상태 상세 분석 시작")
    print("="*80)
    
    for session, algorithm, environment in combinations:
        result_file = results_dir / session / f"{algorithm}_{environment}" / "training_results.json"
        
        if result_file.exists():
            result = analyze_convergence(result_file, algorithm, environment)
            analysis_results.append(result)
        else:
            print(f"❌ {algorithm} + {environment}: 결과 파일 없음")
    
    # 종합 비교
    print(f"\n\n{'='*80}")
    print("🏆 종합 수렴 상태 비교")
    print("="*80)
    
    # CartPole 비교
    cartpole_results = [r for r in analysis_results if r['environment'] == 'CartPole-v1']
    if len(cartpole_results) == 2:
        dqn_cartpole = next(r for r in cartpole_results if r['algorithm'] == 'DQN')
        ddpg_cartpole = next(r for r in cartpole_results if r['algorithm'] == 'DDPG')
        
        print(f"\n🎮 CartPole-v1 비교:")
        print(f"   DQN:  {dqn_cartpole['last_100_avg']:7.2f} ({dqn_cartpole['grade']}) {'✅' if dqn_cartpole['overall_converged'] else '❌'}")
        print(f"   DDPG: {ddpg_cartpole['last_100_avg']:7.2f} ({ddpg_cartpole['grade']}) {'✅' if ddpg_cartpole['overall_converged'] else '❌'}")
        
        diff = ddpg_cartpole['last_100_avg'] - dqn_cartpole['last_100_avg']
        winner = "DDPG" if diff > 0 else "DQN"
        print(f"   🏆 우승: {winner} (+{abs(diff):.2f}점)")
    
    # Pendulum 비교
    pendulum_results = [r for r in analysis_results if r['environment'] == 'Pendulum-v1']
    if len(pendulum_results) == 2:
        dqn_pendulum = next(r for r in pendulum_results if r['algorithm'] == 'DQN')
        ddpg_pendulum = next(r for r in pendulum_results if r['algorithm'] == 'DDPG')
        
        print(f"\n🎯 Pendulum-v1 비교:")
        print(f"   DQN:  {dqn_pendulum['last_100_avg']:7.2f} ({dqn_pendulum['grade']}) {'✅' if dqn_pendulum['overall_converged'] else '❌'}")
        print(f"   DDPG: {ddpg_pendulum['last_100_avg']:7.2f} ({ddpg_pendulum['grade']}) {'✅' if ddpg_pendulum['overall_converged'] else '❌'}")
        
        diff = dqn_pendulum['last_100_avg'] - ddpg_pendulum['last_100_avg']
        winner = "DQN" if diff > 0 else "DDPG"
        print(f"   🏆 우승: {winner} (+{abs(diff):.2f}점)")
    
    # 전체 수렴 통계
    total_converged = sum(1 for r in analysis_results if r['overall_converged'])
    print(f"\n📊 전체 수렴 통계:")
    print(f"   전체 조합: {len(analysis_results)}")
    print(f"   수렴 완료: {total_converged}")
    print(f"   수렴 비율: {total_converged/len(analysis_results)*100:.1f}%")

if __name__ == "__main__":
    main()