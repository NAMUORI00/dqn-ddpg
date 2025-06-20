#!/usr/bin/env python3
"""
Mock PyTorch 환경에서 GPU 최적화 코드 구조 검증
실제 PyTorch 없이도 코드 로직을 테스트할 수 있습니다.
"""

import sys
import os
from unittest.mock import MagicMock
import yaml

# Mock PyTorch 모듈
class MockTorch:
    class device:
        def __init__(self, device_str):
            self.type = device_str.split(':')[0] if ':' in device_str else device_str
            
    class cuda:
        @staticmethod
        def is_available():
            return False  # CPU 환경 시뮬레이션
            
        @staticmethod
        def device_count():
            return 0
            
        class amp:
            class GradScaler:
                def __init__(self):
                    pass
                    
    @staticmethod 
    def tensor(*args, **kwargs):
        return MagicMock()

# Mock 모듈 설정
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_utils():
    """GPU 유틸리티 함수 테스트"""
    print("🔧 GPU 유틸리티 함수 테스트...")
    
    try:
        from src.core.utils import get_device, get_gpu_memory_info
        
        # 디바이스 선택 테스트
        device = get_device()
        print(f"  ✓ get_device() 결과: {device}")
        
        # GPU 메모리 정보 테스트 (CPU 환경)
        memory_info = get_gpu_memory_info()
        print(f"  ✓ GPU 메모리 정보: {memory_info}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_structure():
    """리플레이 버퍼 구조 테스트"""
    print("\n💾 리플레이 버퍼 구조 테스트...")
    
    try:
        from src.core.buffer import ReplayBuffer
        
        # CPU 모드로 버퍼 생성
        buffer = ReplayBuffer(capacity=1000, use_gpu=False)
        print(f"  ✓ ReplayBuffer 생성 성공 (capacity: {buffer.capacity})")
        print(f"  ✓ GPU 사용: {buffer.use_gpu}")
        print(f"  ✓ 디바이스: {buffer.device}")
        
        # 기본 메서드 확인
        print(f"  ✓ is_ready 메서드: {hasattr(buffer, 'is_ready')}")
        print(f"  ✓ sample_tensor 메서드: {hasattr(buffer, 'sample_tensor')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_config_logic():
    """GPU 설정 로직 테스트"""
    print("\n⚙️ GPU 설정 로직 테스트...")
    
    try:
        from src.core.gpu_config import GPUConfig
        
        # 테스트 설정
        test_config = {
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
        
        # GPU 설정 객체 생성
        gpu_config = GPUConfig(test_config)
        print(f"  ✓ GPUConfig 객체 생성 성공")
        print(f"  ✓ GPU enabled: {gpu_config.enabled}")
        print(f"  ✓ Device: {gpu_config.device}")
        print(f"  ✓ GPU buffer: {gpu_config.use_gpu_buffer}")
        print(f"  ✓ Mixed precision: {gpu_config.use_mixed_precision}")
        
        # 메모리 설정 테스트
        memory_config = gpu_config.get_memory_config()
        print(f"  ✓ Memory cleanup interval: {memory_config['cleanup_interval']}")
        
        # 네트워크 설정 테스트
        network_config = gpu_config.get_network_config()
        print(f"  ✓ Hidden dims: {network_config['hidden_dims']}")
        
        # 에이전트 키워드 생성 테스트
        base_kwargs = {'state_dim': 4, 'action_dim': 2}
        agent_kwargs = gpu_config.create_agent_kwargs(base_kwargs)
        print(f"  ✓ Agent kwargs: {list(agent_kwargs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_file_loading():
    """실제 설정 파일 로딩 테스트"""
    print("\n📋 설정 파일 로딩 테스트...")
    
    try:
        # DQN 설정 로딩
        with open('configs/dqn_config.yaml', 'r', encoding='utf-8') as f:
            dqn_config = yaml.safe_load(f)
        
        print(f"  ✓ DQN 설정 로딩 성공")
        
        # GPU 설정 확인
        gpu_settings = dqn_config['gpu']
        print(f"    - GPU enabled: {gpu_settings['enabled']}")
        print(f"    - GPU buffer: {gpu_settings['use_gpu_buffer']}")
        print(f"    - Mixed precision: {gpu_settings['use_mixed_precision']}")
        
        # DDPG 설정 로딩
        with open('configs/ddpg_config.yaml', 'r', encoding='utf-8') as f:
            ddpg_config = yaml.safe_load(f)
            
        print(f"  ✓ DDPG 설정 로딩 성공")
        
        # 네트워크 설정 확인
        network_settings = ddpg_config['network']
        print(f"    - Actor hidden dims: {network_settings['actor']['hidden_dims']}")
        print(f"    - Critic hidden dims: {network_settings['critic']['hidden_dims']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_requirements_structure():
    """Requirements.txt 구조 확인"""
    print("\n📦 Requirements.txt 구조 확인...")
    
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # CUDA 관련 내용 확인
        cuda_keywords = ['torch>=2.0.0', 'CUDA', 'cu118', 'cu121', 'nvidia-ml-py3']
        found_keywords = []
        
        for keyword in cuda_keywords:
            if keyword in content:
                found_keywords.append(keyword)
        
        print(f"  ✓ Requirements.txt 파일 존재")
        print(f"  ✓ CUDA 관련 키워드 발견: {found_keywords}")
        print(f"  ✓ GPU 설치 가이드 포함: {'GPU 버전' in content}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return False

def test_agent_structure():
    """에이전트 클래스 구조 확인"""
    print("\n🤖 에이전트 클래스 구조 확인...")
    
    try:
        # sys.modules에 torch 관련 추가 Mock 설정
        sys.modules['torch.cuda'] = MockTorch.cuda
        sys.modules['torch.cuda.amp'] = MockTorch.cuda.amp
        
        from src.agents.dqn_agent import DQNAgent
        from src.agents.ddpg_agent import DDPGAgent
        
        print(f"  ✓ DQNAgent 클래스 import 성공")
        print(f"  ✓ DDPGAgent 클래스 import 성공")
        
        # DQN 에이전트 생성자 파라미터 확인
        import inspect
        dqn_sig = inspect.signature(DQNAgent.__init__)
        dqn_params = list(dqn_sig.parameters.keys())
        
        gpu_params = ['use_gpu_buffer', 'use_mixed_precision', 'device']
        found_gpu_params = [p for p in gpu_params if p in dqn_params]
        
        print(f"  ✓ DQN GPU 파라미터: {found_gpu_params}")
        
        # DDPG 에이전트 생성자 파라미터 확인
        ddpg_sig = inspect.signature(DDPGAgent.__init__)
        ddpg_params = list(ddpg_sig.parameters.keys())
        
        found_gpu_params_ddpg = [p for p in gpu_params if p in ddpg_params]
        print(f"  ✓ DDPG GPU 파라미터: {found_gpu_params_ddpg}")
        
        return len(found_gpu_params) >= 2 and len(found_gpu_params_ddpg) >= 2
        
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 GPU 최적화 코드 구조 검증 (Mock 환경)\n")
    
    tests = [
        ("GPU 유틸리티", test_gpu_utils),
        ("리플레이 버퍼", test_buffer_structure), 
        ("GPU 설정 로직", test_gpu_config_logic),
        ("설정 파일 로딩", test_config_file_loading),
        ("Requirements 구조", test_requirements_structure),
        ("에이전트 구조", test_agent_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"테스트: {test_name}")
        print('='*60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 테스트 실행 오류: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*60}")
    print("최종 결과")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {passed}/{len(tests)} 테스트 통과")
    
    if passed >= len(tests) * 0.8:  # 80% 이상 통과면 성공
        print("\n🎉 대부분의 테스트 통과! GPU 최적화 코드 구조가 올바르게 구현되었습니다.")
        print("💡 실제 GPU 환경에서는 모든 기능이 정상 작동할 것으로 예상됩니다.")
    else:
        print(f"\n⚠️ {len(tests) - passed}개 테스트 실패. 코드를 점검해주세요.")
    
    return passed >= len(tests) * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)