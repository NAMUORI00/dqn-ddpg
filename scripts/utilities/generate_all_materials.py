#!/usr/bin/env python3
"""
모든 프레젠테이션 자료 생성 스크립트

기존 generate_presentation_materials.py의 기능을 
모듈화된 시각화 시스템으로 대체한 통합 스크립트입니다.

사용법:
    python scripts/utilities/generate_all_materials.py
    python scripts/utilities/generate_all_materials.py --high-quality
    python scripts/utilities/generate_all_materials.py --output-dir custom_output
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.visualization import (
    quick_comparison,
    generate_presentation_materials,
    create_comparison_video,
    get_global_config,
    set_global_config,
    VisualizationConfig
)

def load_sample_data():
    """샘플 데이터 생성 (실제 결과가 없을 때 사용)"""
    import numpy as np
    
    # DQN 샘플 데이터 (CartPole 성능 특성)
    episodes = 500
    dqn_rewards = []
    
    # 초기 학습: 낮은 성능에서 시작
    for i in range(50):
        reward = np.random.normal(20, 10)
        dqn_rewards.append(max(0, reward))
    
    # 중간 학습: 점진적 향상
    for i in range(50, 200):
        base_reward = 20 + (i - 50) * 2.5
        reward = np.random.normal(base_reward, 15)
        dqn_rewards.append(max(0, min(500, reward)))
    
    # 후반 학습: 안정적 성능
    for i in range(200, episodes):
        reward = np.random.normal(450, 30)
        dqn_rewards.append(max(350, min(500, reward)))
    
    # DDPG 샘플 데이터 (Pendulum 성능 특성)
    ddpg_rewards = []
    
    # 초기 학습: 매우 낮은 성능
    for i in range(50):
        reward = np.random.normal(-1500, 200)
        ddpg_rewards.append(min(-800, reward))
    
    # 중간 학습: 점진적 향상
    for i in range(50, 200):
        base_reward = -1500 + (i - 50) * 8
        reward = np.random.normal(base_reward, 100)
        ddpg_rewards.append(max(-1800, min(-200, reward)))
    
    # 후반 학습: 안정적 성능
    for i in range(200, episodes):
        reward = np.random.normal(-250, 50)
        ddpg_rewards.append(max(-400, min(-150, reward)))
    
    dqn_data = {
        'episode_rewards': dqn_rewards,
        'episode_lengths': [np.random.randint(150, 500) for _ in range(len(dqn_rewards))],
        'episodes': list(range(len(dqn_rewards))),
        'environment': 'CartPole-v1',
        'algorithm': 'DQN',
        'policy_type': 'implicit_deterministic',
        'rewards': dqn_rewards  # 호환성을 위해 유지
    }
    
    ddpg_data = {
        'episode_rewards': ddpg_rewards,
        'episode_lengths': [np.random.randint(180, 200) for _ in range(len(ddpg_rewards))],
        'episodes': list(range(len(ddpg_rewards))),
        'environment': 'Pendulum-v1', 
        'algorithm': 'DDPG',
        'policy_type': 'explicit_deterministic',
        'rewards': ddpg_rewards  # 호환성을 위해 유지
    }
    
    return dqn_data, ddpg_data

def load_experiment_data(results_dir: str = "output/logs") -> tuple:
    """실험 결과 데이터 로드"""
    results_path = Path(results_dir)
    
    dqn_file = results_path / "dqn_results.json"
    ddpg_file = results_path / "ddpg_results.json"
    
    dqn_data = None
    ddpg_data = None
    
    # DQN 결과 로드
    if dqn_file.exists():
        try:
            with open(dqn_file, 'r', encoding='utf-8') as f:
                dqn_data = json.load(f)
            print(f"✓ DQN 결과 로드: {dqn_file}")
        except Exception as e:
            print(f"⚠ DQN 결과 로드 실패: {e}")
    
    # DDPG 결과 로드
    if ddpg_file.exists():
        try:
            with open(ddpg_file, 'r', encoding='utf-8') as f:
                ddpg_data = json.load(f)
            print(f"✓ DDPG 결과 로드: {ddpg_file}")
        except Exception as e:
            print(f"⚠ DDPG 결과 로드 실패: {e}")
    
    # 샘플 데이터 사용 여부 결정
    if dqn_data is None or ddpg_data is None:
        print("실제 실험 결과를 찾을 수 없어 샘플 데이터를 사용합니다.")
        sample_dqn, sample_ddpg = load_sample_data()
        
        if dqn_data is None:
            dqn_data = sample_dqn
        if ddpg_data is None:
            ddpg_data = sample_ddpg
    
    return dqn_data, ddpg_data

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="모든 프레젠테이션 자료 생성")
    parser.add_argument("--output-dir", default="output", 
                       help="출력 디렉토리 (기본값: output)")
    parser.add_argument("--results-dir", default="output/logs",
                       help="실험 결과 디렉토리 (기본값: output/logs)")
    parser.add_argument("--high-quality", action="store_true",
                       help="고품질 모드 활성화")
    parser.add_argument("--skip-videos", action="store_true",
                       help="비디오 생성 건너뛰기")
    parser.add_argument("--quick-only", action="store_true",
                       help="빠른 비교만 생성 (전체 자료 생성 안함)")
    
    args = parser.parse_args()
    
    print("🎯 DQN vs DDPG 프레젠테이션 자료 생성 시작")
    print("=" * 60)
    
    # 설정 준비
    config = get_global_config()
    if args.high_quality:
        config.high_quality = True
        config.dpi = 300
        config.figure_size = (12, 8)
        print("📈 고품질 모드 활성화")
    
    set_global_config(config)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print(f"📊 실험 데이터 로드 중... ({args.results_dir})")
    dqn_data, ddpg_data = load_experiment_data(args.results_dir)
    
    generated_materials = {}
    
    try:
        if args.quick_only:
            # 빠른 비교만 생성
            print("⚡ 빠른 비교 차트 생성 중...")
            results = quick_comparison(dqn_data, ddpg_data, 
                                     output_dir=str(output_dir / "charts"))
            generated_materials.update(results)
            
        else:
            # 전체 프레젠테이션 자료 생성
            print("🎨 전체 프레젠테이션 자료 생성 중...")
            results = generate_presentation_materials(dqn_data, ddpg_data, 
                                                    output_dir=str(output_dir))
            generated_materials.update(results)
            
            # 비디오 생성 (선택적)
            if not args.skip_videos:
                try:
                    print("🎬 비교 비디오 생성 중...")
                    video_path = create_comparison_video(
                        dqn_data, ddpg_data, 
                        output_path=str(output_dir / "videos" / "comparison.mp4")
                    )
                    generated_materials['comparison_video'] = video_path
                    print(f"✓ 비교 비디오 생성 완료: {video_path}")
                    
                except Exception as e:
                    print(f"⚠ 비디오 생성 실패: {e}")
                    print("  OpenCV 설치 및 비디오 모듈 상태를 확인하세요.")
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("🎉 프레젠테이션 자료 생성 완료!")
        print("=" * 60)
        
        for material_type, path in generated_materials.items():
            if path:
                print(f"📄 {material_type}: {path}")
        
        print(f"\n📁 모든 자료가 {output_dir}에 저장되었습니다.")
        
        # 요약 파일 생성
        summary_file = output_dir / "generation_summary.json"
        summary_data = {
            'timestamp': str(Path().cwd()),
            'materials_generated': generated_materials,
            'settings': {
                'high_quality': args.high_quality,
                'skip_videos': args.skip_videos,
                'quick_only': args.quick_only,
                'output_dir': str(output_dir),
                'results_dir': args.results_dir
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 생성 요약: {summary_file}")
        
    except Exception as e:
        print(f"❌ 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())