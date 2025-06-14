#!/usr/bin/env python3
"""
비디오 녹화 기능 테스트 스크립트
기본 녹화 기능이 정상 작동하는지 확인합니다.
"""

import sys
import os
# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.environments.video_wrappers import create_video_env
from src.core.video_manager import VideoConfig, VideoManager


def test_basic_recording():
    """기본 녹화 기능 테스트"""
    print("=" * 50)
    print("기본 비디오 녹화 테스트")
    print("=" * 50)
    
    # 비디오 매니저 생성
    config = VideoConfig.get_preset('demo')
    video_manager = VideoManager(config)
    
    # 테스트용 비디오 설정
    video_config = video_manager.get_video_config_for_episode(
        algorithm='dqn',
        episode_id=1,
        is_highlight=True
    )
    
    print(f"비디오 설정: {video_config}")
    
    # 비디오 녹화 환경 생성
    env = create_video_env("CartPole-v1", video_config)
    
    print("환경 생성 완료. 짧은 에피소드를 실행합니다...")
    
    # 짧은 에피소드 실행
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(50):  # 최대 50 스텝
        # 무작위 행동 (테스트용)
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"에피소드 완료: {steps} 스텝, 총 보상: {total_reward}")
    
    # 환경 정리
    env.close()
    
    # 결과 확인
    storage_summary = video_manager.get_storage_summary()
    print(f"저장 공간 사용량: {storage_summary}")
    
    video_list = video_manager.export_video_list('dqn')
    if video_list:
        print(f"생성된 비디오: {video_list[-1]['path']}")
        print("✅ 비디오 녹화 테스트 성공!")
    else:
        print("❌ 비디오 생성 실패")


def test_dual_environments():
    """DQN과 DDPG 환경 모두 테스트"""
    print("\n" + "=" * 50)
    print("DQN & DDPG 환경 녹화 테스트")
    print("=" * 50)
    
    config = VideoConfig.get_preset('demo')
    video_manager = VideoManager(config)
    
    # DQN 환경 테스트
    print("\n[DQN 테스트]")
    dqn_config = video_manager.get_video_config_for_episode('dqn', 2, True)
    dqn_env = create_video_env("CartPole-v1", dqn_config)
    
    state, _ = dqn_env.reset()
    for _ in range(30):
        action = dqn_env.action_space.sample()
        state, reward, terminated, truncated, _ = dqn_env.step(action)
        if terminated or truncated:
            break
    dqn_env.close()
    print("DQN 환경 테스트 완료")
    
    # DDPG 환경 테스트  
    print("\n[DDPG 테스트]")
    ddpg_config = video_manager.get_video_config_for_episode('ddpg', 2, True)
    ddpg_env = create_video_env("Pendulum-v1", ddpg_config)
    
    state, _ = ddpg_env.reset()
    for _ in range(30):
        action = ddpg_env.action_space.sample()
        state, reward, terminated, truncated, _ = ddpg_env.step(action)
        if terminated or truncated:
            break
    ddpg_env.close()
    print("DDPG 환경 테스트 완료")
    
    # 결과 확인
    video_list = video_manager.export_video_list()
    print(f"\n생성된 비디오 수: {len(video_list)}")
    for video in video_list:
        print(f"  - {video['experiment']}: {video['path']}")


def test_config_loading():
    """설정 파일 로딩 테스트"""
    print("\n" + "=" * 50)
    print("설정 파일 로딩 테스트")
    print("=" * 50)
    
    try:
        # YAML 설정 로드
        config = VideoConfig.from_yaml('configs/video_recording.yaml')
        print(f"✅ 설정 로드 성공: {config}")
        
        # 프리셋 테스트
        presets = ['low', 'medium', 'high', 'demo']
        for preset in presets:
            preset_config = VideoConfig.get_preset(preset)
            print(f"✅ {preset} 프리셋: {preset_config.quality} 품질")
            
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")


def main():
    """메인 테스트 함수"""
    parser = argparse.ArgumentParser(description='비디오 녹화 기능 테스트')
    parser.add_argument('--test', choices=['basic', 'dual', 'config', 'all'], 
                       default='all', help='실행할 테스트')
    
    args = parser.parse_args()
    
    print("📹 비디오 녹화 시스템 테스트 시작")
    
    try:
        if args.test in ['basic', 'all']:
            test_basic_recording()
        
        if args.test in ['dual', 'all']:
            test_dual_environments()
        
        if args.test in ['config', 'all']:
            test_config_loading()
        
        print("\n" + "🎉 모든 테스트 완료!")
        print("\n다음 단계:")
        print("1. videos/ 폴더에서 생성된 비디오 확인")
        print("2. 비디오 품질 및 오버레이 확인")
        print("3. 메타데이터 파일 확인")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        print("OpenCV 설치 확인: pip install opencv-python")


if __name__ == "__main__":
    main()