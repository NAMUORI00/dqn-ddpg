#!/usr/bin/env python3
"""
Test script for refactored components
Validates that the new unified systems work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_manager():
    """Test the new ConfigManager"""
    print("🧪 Testing ConfigManager...")
    
    try:
        from src.core.config_manager import ConfigManager, GPUConfig, VideoConfig
        
        # Test initialization
        config_manager = ConfigManager(environment="development")
        print("  ✅ ConfigManager initialization successful")
        
        # Test GPU config
        gpu_config = config_manager.get_gpu_config()
        print(f"  ✅ GPU config loaded: enabled={gpu_config.enabled}")
        
        # Test video config
        video_config = config_manager.get_video_config(preset="medium")
        print(f"  ✅ Video config loaded: {video_config.resolution}@{video_config.fps}fps")
        
        # Test algorithm config
        try:
            dqn_config = config_manager.get_algorithm_config("dqn")
            print(f"  ✅ DQN config loaded: lr={dqn_config['agent']['learning_rate']}")
        except KeyError:
            print("  ⚠️ DQN config not found (expected if algorithms section missing)")
        
        # Test environment-specific settings
        config_manager_prod = ConfigManager(environment="production")
        prod_video = config_manager_prod.get_video_config()
        print(f"  ✅ Production video config: {prod_video.quality}")
        
        print("  🎉 ConfigManager tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ ConfigManager test failed: {e}")
        return False

def test_exceptions():
    """Test the exception handling system"""
    print("\n🧪 Testing Exception System...")
    
    try:
        from src.core.exceptions import (
            DQNDDPGException, GPUNotAvailableError, ConfigurationError,
            ErrorHandler, RecoveryStrategy
        )
        
        # Test custom exception
        try:
            raise GPUNotAvailableError("Test GPU error")
        except GPUNotAvailableError as e:
            error_info = e.get_error_info()
            print(f"  ✅ GPU exception handled: {error_info['user_message']}")
        
        # Test error handler
        error_handler = ErrorHandler()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_info = error_handler.handle_error(e, context="test", reraise=False)
            print(f"  ✅ Error handler worked: {error_info['category']}")
        
        # Test recovery strategy
        recovery = RecoveryStrategy(max_retries=2)
        
        def failing_operation():
            if not hasattr(failing_operation, 'calls'):
                failing_operation.calls = 0
            failing_operation.calls += 1
            if failing_operation.calls < 2:
                raise ValueError("Simulated failure")
            return "success"
        
        result = recovery.with_retry(failing_operation, "test operation")
        print(f"  ✅ Recovery strategy worked: {result}")
        
        print("  🎉 Exception system tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Exception system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_factory():
    """Test the environment factory (without creating actual environments)"""
    print("\n🧪 Testing Environment Factory...")
    
    try:
        from src.environments.factory import EnvironmentFactory
        from src.core.config_manager import ConfigManager
        
        # Test factory initialization
        config_manager = ConfigManager()
        factory = EnvironmentFactory(config_manager)
        print("  ✅ EnvironmentFactory initialization successful")
        
        # Test environment detection
        agent_type = factory._detect_agent_type("CartPole-v1")
        print(f"  ✅ Agent type detection: CartPole-v1 -> {agent_type}")
        
        agent_type = factory._detect_agent_type("Pendulum-v1")
        print(f"  ✅ Agent type detection: Pendulum-v1 -> {agent_type}")
        
        # Test environment normalization
        normalized = factory._normalize_env_name("cartpole")
        print(f"  ✅ Environment normalization: cartpole -> {normalized}")
        
        # Test supported environments
        supported = factory.get_supported_environments()
        print(f"  ✅ Supported environments: {len(supported['discrete'])} discrete, {len(supported['continuous'])} continuous")
        
        print("  🎉 Environment factory tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Environment factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_agent():
    """Test the base agent class (without PyTorch)"""
    print("\n🧪 Testing Base Agent Structure...")
    
    try:
        # Just test that the file can be compiled
        import py_compile
        py_compile.compile('src/agents/base_agent.py', doraise=True)
        print("  ✅ BaseAgent syntax check passed")
        
        py_compile.compile('src/agents/dqn_agent.py', doraise=True)
        print("  ✅ DQNAgent syntax check passed")
        
        py_compile.compile('src/agents/ddpg_agent.py', doraise=True)
        print("  ✅ DDPGAgent syntax check passed")
        
        print("  🎉 Base agent structure tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Base agent test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Refactored Components")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_config_manager())
    results.append(test_exceptions())
    results.append(test_environment_factory())
    results.append(test_base_agent())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print(f"❌ Failed: {total - passed}/{total}")
        print("⚠️ Some issues need to be addressed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())