#!/usr/bin/env python3
"""
Simple dual recording test - ASCII only for Windows compatibility
"""

import sys
import os
# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
import gymnasium as gym
import numpy as np

try:
    from src.core.video_manager import VideoConfig, VideoManager
    from src.core.dual_recorder import DualVideoRecorder, DualRecordingConfig
    from src.core.recording_scheduler import create_default_recording_scheduler
    print("SUCCESS: All imports successful")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic dual recording functionality"""
    print("=" * 50)
    print("Testing basic dual recording functionality")
    print("=" * 50)
    
    try:
        # 1. Create video config
        print("1. Creating video configuration...")
        video_config = VideoConfig.get_preset('demo')
        print(f"   Video config: {video_config.resolution} @ {video_config.fps}fps")
        
        # 2. Create video manager
        print("2. Creating video manager...")
        video_manager = VideoManager(video_config)
        print("   Video manager created successfully")
        
        # 3. Create dual recording config
        print("3. Creating dual recording config...")
        dual_config = DualRecordingConfig()
        print(f"   Full recording: {dual_config.full_resolution} @ {dual_config.full_fps}fps")
        print(f"   Selective recording: {dual_config.selective_resolution} @ {dual_config.selective_fps}fps")
        
        # 4. Create dual recorder
        print("4. Creating dual recorder...")
        dual_recorder = DualVideoRecorder(video_manager, dual_config)
        print("   Dual recorder created successfully")
        
        # 5. Create scheduler
        print("5. Creating scheduler...")
        scheduler = create_default_recording_scheduler()
        print("   Scheduler created successfully")
        
        # 6. Test scheduler logic
        print("6. Testing scheduler logic...")
        should_record_ep1 = scheduler.should_record_episode('dqn', 1)
        should_record_ep4 = scheduler.should_record_episode('dqn', 4)
        should_record_ep50 = scheduler.should_record_episode('dqn', 50)
        
        print(f"   Episode 1 should record: {should_record_ep1}")
        print(f"   Episode 4 should record: {should_record_ep4}")
        print(f"   Episode 50 should record: {should_record_ep50}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False


def test_environment_creation():
    """Test environment creation with render mode"""
    print("\n" + "=" * 50)
    print("Testing environment creation")
    print("=" * 50)
    
    try:
        # Test basic environment creation
        print("1. Creating CartPole environment...")
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        print("   Environment created successfully")
        
        # Test reset and step
        print("2. Testing environment functionality...")
        state, info = env.reset()
        print(f"   Initial state shape: {state.shape}")
        
        # Test rendering
        print("3. Testing rendering...")
        frame = env.render()
        if frame is not None:
            print(f"   Render frame shape: {frame.shape}")
            print("   Rendering successful")
        else:
            print("   WARNING: Rendering returned None")
        
        # Test a few steps
        print("4. Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            print(f"   Step {i+1}: reward={reward:.2f}, frame_shape={frame.shape if frame is not None else None}")
            
            if terminated or truncated:
                print("   Episode terminated")
                break
        
        env.close()
        print("   Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Environment test failed: {e}")
        return False


def test_config_loading():
    """Test YAML config loading"""
    print("\n" + "=" * 50)
    print("Testing configuration loading")
    print("=" * 50)
    
    try:
        # Test video recording config
        print("1. Loading video recording config...")
        config_path = "configs/video_recording.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("   Config loaded successfully")
            print(f"   Video enabled: {config.get('video', {}).get('enabled', False)}")
            print(f"   Dual recording enabled: {config.get('dual_recording', {}).get('enabled', False)}")
            
            # Test dual config creation
            dual_config = DualRecordingConfig.from_yaml_config(config)
            print(f"   Dual config created: {dual_config.full_resolution} / {dual_config.selective_resolution}")
            
        else:
            print(f"   WARNING: Config file not found: {config_path}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Config test failed: {e}")
        return False


def simulate_short_recording():
    """Simulate a short recording session"""
    print("\n" + "=" * 50)
    print("Simulating short recording session")
    print("=" * 50)
    
    try:
        # Setup
        print("1. Setting up recording system...")
        video_config = VideoConfig.get_preset('demo')
        video_manager = VideoManager(video_config)
        dual_config = DualRecordingConfig()
        dual_recorder = DualVideoRecorder(video_manager, dual_config)
        
        # Create environment
        print("2. Creating environment...")
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        
        # Start recording
        print("3. Starting recording for episode 1...")
        success = dual_recorder.start_episode_recording('dqn', 1, record_selective=True)
        if not success:
            print("   ERROR: Failed to start recording")
            return False
        
        print("   Recording started successfully")
        
        # Simulate episode
        print("4. Simulating episode...")
        state, info = env.reset()
        total_reward = 0
        
        for step in range(20):  # Short episode
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Capture frame
            frame = env.render()
            if frame is not None:
                dual_recorder.add_frame(frame)
            
            if step % 5 == 0:
                print(f"   Step {step}: reward={total_reward:.1f}")
            
            if terminated or truncated:
                break
        
        # Stop recording
        print("5. Stopping recording...")
        episode_metadata = {
            'episode_id': 1,
            'total_reward': total_reward,
            'episode_length': step + 1,
            'algorithm': 'dqn'
        }
        
        dual_recorder.stop_episode_recording(episode_metadata)
        print(f"   Recording stopped. Total reward: {total_reward:.2f}")
        
        # Check stats
        stats = dual_recorder.get_recording_stats()
        print("6. Recording statistics:")
        print(f"   Total episodes: {stats['total_episodes']}")
        print(f"   Frames processed: {stats['total_frames_processed']}")
        print(f"   Avg processing time: {stats['average_processing_time']*1000:.2f}ms")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"ERROR: Recording simulation failed: {e}")
        return False


def main():
    """Main test execution"""
    print("Dual Recording System Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Environment Creation", test_environment_creation),
        ("Configuration Loading", test_config_loading),
        ("Recording Simulation", simulate_short_recording)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"PASSED: {test_name}")
                passed += 1
            else:
                print(f"FAILED: {test_name}")
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        print("The dual recording system is ready for use.")
        print("\nNext steps:")
        print("1. Run: python run_experiment.py --help")
        print("2. Test: python run_experiment.py --dual-video --video-preset demo")
    else:
        print(f"WARNING: {total - passed} tests failed")
        print("Please check the error messages above")
    
    return passed == total


if __name__ == "__main__":
    main()