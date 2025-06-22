#!/usr/bin/env python3
"""
간단한 수렴 상태 확인 스크립트
"""

import json
from pathlib import Path

def check_convergence(session, algorithm, environment):
    """수렴 상태 확인"""
    
    result_file = Path(f"results/cross_environment/{session}/{algorithm}_{environment}/training_results.json")
    
    if not result_file.exists():
        return None
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    final_avg_reward = data['final_avg_reward']
    best_avg_reward = data['best_avg_reward']
    converged = data.get('converged', False)
    
    # 수렴 기준
    if environment == "CartPole-v1":
        threshold = 475.0
        target = 500.0
    else:  # Pendulum-v1
        threshold = -200.0
        target = -50.0
    
    # 실제 수렴 여부 재확인
    actual_converged = final_avg_reward >= threshold
    excellent = final_avg_reward >= (target * 0.9 if environment == "CartPole-v1" else target * 1.1)
    
    return {
        'algorithm': algorithm,
        'environment': environment,
        'final_avg_reward': final_avg_reward,
        'best_avg_reward': best_avg_reward,
        'threshold': threshold,
        'stored_converged': converged,
        'actual_converged': actual_converged,
        'excellent': excellent
    }

def main():
    combinations = [
        ("20250621_231516", "DQN", "CartPole-v1"),
        ("20250621_235008", "DDPG", "CartPole-v1"),
        ("20250622_003219", "DQN", "Pendulum-v1"),
        ("20250622_004126", "DDPG", "Pendulum-v1")
    ]
    
    print("🔍 수렴 상태 확인")
    print("="*80)
    
    results = []
    for session, algorithm, environment in combinations:
        result = check_convergence(session, algorithm, environment)
        if result:
            results.append(result)
            
            print(f"\n🎯 {algorithm} + {environment}:")
            print(f"   최종 평균: {result['final_avg_reward']:8.2f}")
            print(f"   최고 평균: {result['best_avg_reward']:8.2f}")
            print(f"   수렴 기준: {result['threshold']:8.2f}")
            print(f"   수렴 여부: {'✅' if result['actual_converged'] else '❌'} ({result['final_avg_reward']:.2f} >= {result['threshold']})")
            print(f"   우수 성능: {'✅' if result['excellent'] else '❌'}")
            
            if result['stored_converged'] != result['actual_converged']:
                print(f"   ⚠️  저장된 수렴 상태와 다름!")
    
    print(f"\n\n{'='*80}")
    print("📊 종합 수렴 분석")
    print("="*80)
    
    # CartPole 비교
    cartpole_results = [r for r in results if r['environment'] == 'CartPole-v1']
    if len(cartpole_results) == 2:
        dqn = next(r for r in cartpole_results if r['algorithm'] == 'DQN')
        ddpg = next(r for r in cartpole_results if r['algorithm'] == 'DDPG')
        
        print(f"\n🎮 CartPole-v1 (수렴 기준: 475.0):")
        print(f"   DQN:  {dqn['final_avg_reward']:7.2f} {'✅ 수렴' if dqn['actual_converged'] else '❌ 미수렴'}")
        print(f"   DDPG: {ddpg['final_avg_reward']:7.2f} {'✅ 수렴' if ddpg['actual_converged'] else '❌ 미수렴'}")
        
        diff = ddpg['final_avg_reward'] - dqn['final_avg_reward']
        print(f"   차이: DDPG가 {diff:+.2f}점 {'우수' if diff > 0 else '부족'}")
    
    # Pendulum 비교
    pendulum_results = [r for r in results if r['environment'] == 'Pendulum-v1']
    if len(pendulum_results) == 2:
        dqn = next(r for r in pendulum_results if r['algorithm'] == 'DQN')
        ddpg = next(r for r in pendulum_results if r['algorithm'] == 'DDPG')
        
        print(f"\n🎯 Pendulum-v1 (수렴 기준: -200.0):")
        print(f"   DQN:  {dqn['final_avg_reward']:7.2f} {'✅ 수렴' if dqn['actual_converged'] else '❌ 미수렴'}")
        print(f"   DDPG: {ddpg['final_avg_reward']:7.2f} {'✅ 수렴' if ddpg['actual_converged'] else '❌ 미수렴'}")
        
        diff = dqn['final_avg_reward'] - ddpg['final_avg_reward']
        print(f"   차이: DQN이 {diff:+.2f}점 {'우수' if diff > 0 else '부족'}")
    
    # 전체 통계
    total_converged = sum(1 for r in results if r['actual_converged'])
    total_excellent = sum(1 for r in results if r['excellent'])
    
    print(f"\n📈 전체 통계:")
    print(f"   총 조합: {len(results)}")
    print(f"   수렴 완료: {total_converged}/{len(results)} ({total_converged/len(results)*100:.1f}%)")
    print(f"   우수 성능: {total_excellent}/{len(results)} ({total_excellent/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()