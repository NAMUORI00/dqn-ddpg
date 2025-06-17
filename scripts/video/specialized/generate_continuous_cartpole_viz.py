"""
ContinuousCartPole 실험 결과 시각화 생성

기존 JSON 결과 파일을 읽어서 시각화 이미지를 생성합니다.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가 (scripts/video/specialized에서 루트로)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
os.chdir(project_root)

def load_results(json_path):
    """JSON 결과 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualization(results):
    """시각화 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Same Environment DQN vs DDPG Comparison Results', fontsize=16, fontweight='bold')
    
    # 학습 곡선
    dqn_scores = results['training_results']['dqn']['scores']
    ddpg_scores = results['training_results']['ddpg']['scores']
    
    axes[0, 0].plot(dqn_scores, label='DQN', alpha=0.7, color='blue', linewidth=1)
    axes[0, 0].plot(ddpg_scores, label='DDPG', alpha=0.7, color='red', linewidth=1)
    axes[0, 0].set_title('Learning Performance Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 평가 점수
    dqn_metrics = results['training_results']['dqn']['metrics']
    ddpg_metrics = results['training_results']['ddpg']['metrics']
    
    if dqn_metrics and ddpg_metrics:
        dqn_eval_episodes = [m['episode'] for m in dqn_metrics]
        dqn_eval_scores = [m['eval_score'] for m in dqn_metrics]
        ddpg_eval_episodes = [m['episode'] for m in ddpg_metrics]
        ddpg_eval_scores = [m['eval_score'] for m in ddpg_metrics]
        
        axes[0, 1].plot(dqn_eval_episodes, dqn_eval_scores, 'o-', label='DQN', color='blue', markersize=4)
        axes[0, 1].plot(ddpg_eval_episodes, ddpg_eval_scores, 'o-', label='DDPG', color='red', markersize=4)
    
    axes[0, 1].set_title('Evaluation Performance Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Evaluation Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 행동 비교
    if 'determinism_analysis' in results and 'action_comparison' in results['determinism_analysis']:
        comp = results['determinism_analysis']['action_comparison']
        dqn_actions = comp['dqn_actions']
        ddpg_actions = comp['ddpg_actions']
        
        axes[1, 0].scatter(dqn_actions, ddpg_actions, alpha=0.6, s=30)
        axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5, linewidth=2)
        axes[1, 0].set_title('Action Selection Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('DQN Action')
        axes[1, 0].set_ylabel('DDPG Action')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-1.1, 1.1)
        axes[1, 0].set_ylim(-1.1, 1.1)
    
    # 최종 성능 비교
    final_dqn = results['final_scores']['dqn']
    final_ddpg = results['final_scores']['ddpg']
    
    bars = axes[1, 1].bar(['DQN', 'DDPG'], [final_dqn, final_ddpg], 
                         color=['blue', 'red'], alpha=0.7, width=0.6)
    axes[1, 1].set_title('Final Performance Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Final Score')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, value in zip(bars, [final_dqn, final_ddpg]):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """메인 함수"""
    # JSON 파일 경로
    json_path = "results/same_environment_comparison/comparison_results_20250615_135451.json"
    
    if not os.path.exists(json_path):
        print(f"JSON 파일을 찾을 수 없습니다: {json_path}")
        return
    
    print("ContinuousCartPole 실험 결과 시각화 생성 중...")
    
    # 결과 로드
    results = load_results(json_path)
    
    # 시각화 생성
    fig = create_visualization(results)
    
    # 저장
    save_dir = "results/same_environment_comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"comparison_plots_{timestamp}.png")
    
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"시각화 저장 완료: {fig_path}")
    
    # 요약 정보 출력
    print("\n=== 실험 결과 요약 ===")
    print(f"환경: {results['environment_info']['name']}")
    print(f"DQN 최종 점수: {results['final_scores']['dqn']:.2f}")
    print(f"DDPG 최종 점수: {results['final_scores']['ddpg']:.2f}")
    
    if 'determinism_analysis' in results:
        dqn_det = results['determinism_analysis']['dqn_determinism']['determinism_score']
        ddpg_det = results['determinism_analysis']['ddpg_determinism']['determinism_score']
        print(f"DQN 결정성: {dqn_det:.3f}")
        print(f"DDPG 결정성: {ddpg_det:.3f}")

if __name__ == "__main__":
    main()