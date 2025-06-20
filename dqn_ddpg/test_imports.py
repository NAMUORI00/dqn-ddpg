#!/usr/bin/env python3
"""
GPU 최적화 코드의 import 및 기본 구조 테스트
실제 PyTorch 없이도 코드 구조를 검증할 수 있습니다.
"""

import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """모든 주요 모듈 import 테스트"""
    print("🔍 GPU 최적화 모듈 import 테스트 시작...")
    
    try:
        # 핵심 유틸리티 import 테스트
        print("  ✓ src.core.utils 모듈 체크...")
        from src.core.utils import get_device, set_cuda_options, enable_mixed_precision
        print("    - get_device, set_cuda_options, enable_mixed_precision 함수 확인")
        
        # GPU 버퍼 import 테스트  
        print("  ✓ src.core.buffer 모듈 체크...")
        from src.core.buffer import ReplayBuffer, PrioritizedReplayBuffer
        print("    - ReplayBuffer, PrioritizedReplayBuffer 클래스 확인")
        
        # GPU 설정 모듈 import 테스트
        print("  ✓ src.core.gpu_config 모듈 체크...")
        from src.core.gpu_config import GPUConfig, load_gpu_config
        print("    - GPUConfig, load_gpu_config 클래스/함수 확인")
        
        print("\n✅ 모든 모듈 import 성공!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import 오류: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        return False

def test_config_structure():
    """설정 파일 구조 테스트"""
    print("\n🔍 설정 파일 구조 테스트...")
    
    try:
        import yaml
        
        # DQN 설정 파일 테스트
        print("  ✓ DQN 설정 파일 체크...")
        with open('configs/dqn_config.yaml', 'r', encoding='utf-8') as f:
            dqn_config = yaml.safe_load(f)
        
        # GPU 설정 존재 확인
        assert 'gpu' in dqn_config, "DQN 설정에 GPU 섹션이 없습니다"
        assert 'network' in dqn_config, "DQN 설정에 network 섹션이 없습니다"
        print("    - GPU 및 network 설정 섹션 확인")
        
        # DDPG 설정 파일 테스트
        print("  ✓ DDPG 설정 파일 체크...")
        with open('configs/ddpg_config.yaml', 'r', encoding='utf-8') as f:
            ddpg_config = yaml.safe_load(f)
            
        assert 'gpu' in ddpg_config, "DDPG 설정에 GPU 섹션이 없습니다"
        assert 'network' in ddpg_config, "DDPG 설정에 network 섹션이 없습니다"
        print("    - GPU 및 network 설정 섹션 확인")
        
        print("\n✅ 모든 설정 파일 구조 정상!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ 설정 파일을 찾을 수 없습니다: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"\n❌ YAML 파싱 오류: {e}")
        return False
    except AssertionError as e:
        print(f"\n❌ 설정 구조 오류: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        return False

def test_mock_gpu_config():
    """Mock 환경에서 GPU 설정 테스트"""
    print("\n🔍 GPU 설정 로직 테스트...")
    
    try:
        # Mock 설정 생성
        mock_config = {
            'gpu': {
                'enabled': 'auto',
                'device_id': None,
                'use_gpu_buffer': True,
                'use_mixed_precision': True,
                'cuda_options': {
                    'allow_tf32': True,
                    'benchmark': True,
                    'deterministic': False
                },
                'memory': {
                    'cleanup_interval': 1000,
                    'monitor_usage': True
                }
            },
            'network': {
                'hidden_dims': [256, 256],
                'use_layer_norm': False,
                'dropout_rate': 0.0
            }
        }
        
        from src.core.gpu_config import GPUConfig
        
        # GPU 설정 객체 생성 테스트
        print("  ✓ GPUConfig 객체 생성 테스트...")
        gpu_config = GPUConfig(mock_config)
        print(f"    - GPU enabled: {gpu_config.enabled}")
        print(f"    - Device: {gpu_config.device}")
        print(f"    - GPU buffer: {gpu_config.use_gpu_buffer}")
        print(f"    - Mixed precision: {gpu_config.use_mixed_precision}")
        
        # 메모리 설정 테스트
        memory_config = gpu_config.get_memory_config()
        print(f"    - Memory cleanup interval: {memory_config['cleanup_interval']}")
        
        # 네트워크 설정 테스트
        network_config = gpu_config.get_network_config()
        print(f"    - Hidden dims: {network_config['hidden_dims']}")
        
        print("\n✅ GPU 설정 로직 정상 작동!")
        return True
        
    except Exception as e:
        print(f"\n❌ GPU 설정 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 DQN/DDPG GPU 최적화 코드 검증 시작\n")
    
    tests = [
        ("모듈 Import", test_imports),
        ("설정 파일 구조", test_config_structure), 
        ("GPU 설정 로직", test_mock_gpu_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"테스트: {test_name}")
        print('='*50)
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("최종 결과")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {passed}/{len(tests)} 테스트 통과")
    
    if passed == len(tests):
        print("\n🎉 모든 테스트 통과! GPU 최적화 코드가 정상적으로 구현되었습니다.")
    else:
        print(f"\n⚠️ {len(tests) - passed}개 테스트 실패. 코드를 점검해주세요.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)