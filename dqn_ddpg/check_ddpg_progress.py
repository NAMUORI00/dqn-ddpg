#!/usr/bin/env python3
"""DDPG 학습 진행상황 체크"""

import torch
import json
from pathlib import Path
import os

def check_progress():
    models_dir = Path("models/ddpg_manual")
    
    if not models_dir.exists():
        print("❌ 모델 디렉토리가 없습니다.")
        return
        
    # 체크포인트 파일들 확인
    checkpoints = list(models_dir.glob("checkpoint_episode_*.pth"))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    print(f"📊 저장된 체크포인트: {len(checkpoints)}개")
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        episode_num = int(latest_checkpoint.stem.split('_')[-1])
        print(f"📈 최신 에피소드: {episode_num}")
        
        # 최신 체크포인트 로드
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
            avg_reward = checkpoint.get('avg_reward', 'Unknown')
            print(f"📊 최신 평균 보상: {avg_reward}")
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 오류: {e}")
    
    # best_model 확인
    best_model_path = models_dir / "best_model.pth"
    if best_model_path.exists():
        try:
            best_model = torch.load(best_model_path, map_location='cpu', weights_only=False)
            best_episode = best_model.get('episode', 'Unknown')
            best_reward = best_model.get('avg_reward', 'Unknown')
            print(f"🏆 최고 성능: Episode {best_episode}, 평균 보상 {best_reward}")
        except Exception as e:
            print(f"⚠️ 최고 모델 로드 오류: {e}")
    
    # 프로세스 확인
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_ddpg_manual.py' in result.stdout:
            print("🔄 DDPG 학습이 계속 진행 중입니다.")
        else:
            print("⏹️ DDPG 학습이 완료되었습니다.")
            
            # 결과 파일 확인
            results_dir = Path("results/ddpg_manual")
            if results_dir.exists():
                result_files = list(results_dir.glob("*.json"))
                if result_files:
                    latest_result = max(result_files, key=os.path.getmtime)
                    print(f"📋 결과 파일: {latest_result}")
                    
                    with open(latest_result, 'r') as f:
                        results = json.load(f)
                    
                    print(f"✅ 최종 결과:")
                    print(f"   총 에피소드: {results.get('total_episodes', 'Unknown')}")
                    print(f"   최종 평균 보상: {results.get('final_avg_reward', 'Unknown'):.1f}")
                    print(f"   수렴 여부: {results.get('converged', 'Unknown')}")
                    print(f"   학습 시간: {results.get('training_time', 0)/60:.1f}분")
                else:
                    print("📋 결과 파일이 아직 생성되지 않았습니다.")
            
    except Exception as e:
        print(f"⚠️ 프로세스 확인 오류: {e}")

if __name__ == "__main__":
    check_progress()