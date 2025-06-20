"""GPU 설정 및 관리를 위한 유틸리티 모듈

이 모듈은 YAML 설정 파일의 GPU 설정을 파싱하고 
적절한 GPU 최적화 옵션을 적용하는 기능을 제공합니다.
"""

import torch
import yaml
from typing import Dict, Any, Optional, Union
from .utils import get_device, set_cuda_options, enable_mixed_precision


class GPUConfig:
    """GPU 설정 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: YAML 설정 딕셔너리
        """
        self.gpu_config = config.get('gpu', {})
        self.network_config = config.get('network', {})
        
        # GPU 설정 파싱
        self.enabled = self._parse_gpu_enabled()
        self.device = self._get_device()
        self.use_gpu_buffer = self._get_gpu_buffer_setting()
        self.use_mixed_precision = self._get_mixed_precision_setting()
        
        # CUDA 옵션 적용
        if self.device.type == 'cuda':
            self._apply_cuda_options()
    
    def _parse_gpu_enabled(self) -> bool:
        """GPU 사용 여부 파싱"""
        enabled = self.gpu_config.get('enabled', 'auto')
        
        if enabled == 'auto':
            return torch.cuda.is_available()
        elif enabled is True or enabled == 'true':
            if not torch.cuda.is_available():
                raise RuntimeError("GPU가 요청되었지만 CUDA를 사용할 수 없습니다.")
            return True
        else:
            return False
    
    def _get_device(self) -> torch.device:
        """디바이스 객체 반환"""
        if not self.enabled:
            return torch.device('cpu')
        
        device_id = self.gpu_config.get('device_id')
        return get_device(device_id)
    
    def _get_gpu_buffer_setting(self) -> bool:
        """GPU 버퍼 사용 설정"""
        if not self.enabled:
            return False
        return self.gpu_config.get('use_gpu_buffer', True)
    
    def _get_mixed_precision_setting(self) -> bool:
        """Mixed Precision 사용 설정"""
        if not self.enabled:
            return False
        return self.gpu_config.get('use_mixed_precision', True)
    
    def _apply_cuda_options(self) -> None:
        """CUDA 최적화 옵션 적용"""
        cuda_options = self.gpu_config.get('cuda_options', {})
        
        set_cuda_options(
            allow_tf32=cuda_options.get('allow_tf32', True),
            benchmark=cuda_options.get('benchmark', True),
            deterministic=cuda_options.get('deterministic', False)
        )
    
    def get_memory_config(self) -> Dict[str, Any]:
        """메모리 관리 설정 반환"""
        return self.gpu_config.get('memory', {
            'cleanup_interval': 1000,
            'monitor_usage': True
        })
    
    def get_network_config(self, network_type: str = 'default') -> Dict[str, Any]:
        """네트워크 아키텍처 설정 반환
        
        Args:
            network_type: 네트워크 타입 ('actor', 'critic', 'default')
        """
        if network_type in self.network_config:
            return self.network_config[network_type]
        else:
            return self.network_config
    
    def create_agent_kwargs(self, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 생성을 위한 키워드 인수 생성
        
        Args:
            base_kwargs: 기본 에이전트 파라미터
            
        Returns:
            GPU 설정이 추가된 키워드 인수
        """
        kwargs = base_kwargs.copy()
        kwargs['device'] = self.device
        kwargs['use_gpu_buffer'] = self.use_gpu_buffer
        kwargs['use_mixed_precision'] = self.use_mixed_precision
        
        return kwargs
    
    def print_info(self) -> None:
        """GPU 설정 정보 출력"""
        print("🔧 GPU 설정 정보:")
        print(f"  - 디바이스: {self.device}")
        print(f"  - GPU 버퍼 사용: {self.use_gpu_buffer}")
        print(f"  - Mixed Precision: {self.use_mixed_precision}")
        
        if self.device.type == 'cuda':
            from .utils import get_gpu_memory_info
            gpu_info = get_gpu_memory_info(self.device)
            if gpu_info.get('available', False):
                print(f"  - GPU 메모리: {gpu_info['allocated']:.1f}GB / {gpu_info['total']:.1f}GB")
                print(f"  - 메모리 사용률: {gpu_info['utilization']:.1f}%")


def load_gpu_config(config_path: str) -> GPUConfig:
    """YAML 파일에서 GPU 설정 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        GPU 설정 객체
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return GPUConfig(config)


def get_optimal_batch_size(base_batch_size: int, device: torch.device, 
                          model_size: str = 'medium') -> int:
    """GPU에 최적화된 배치 크기 추천
    
    Args:
        base_batch_size: 기본 배치 크기
        device: 사용할 디바이스
        model_size: 모델 크기 ('small', 'medium', 'large')
        
    Returns:
        최적화된 배치 크기
    """
    if device.type != 'cuda':
        return base_batch_size
    
    try:
        from .utils import get_gpu_memory_info
        gpu_info = get_gpu_memory_info(device)
        
        if not gpu_info.get('available', False):
            return base_batch_size
        
        # GPU 메모리 크기에 따른 배치 크기 조정
        total_memory_gb = gpu_info['total']
        
        # 모델 크기에 따른 메모리 사용량 추정
        memory_multipliers = {
            'small': 1.0,
            'medium': 1.5,
            'large': 2.0
        }
        
        multiplier = memory_multipliers.get(model_size, 1.5)
        
        # GPU 메모리에 따른 배치 크기 조정
        if total_memory_gb >= 24:  # RTX 4090, A100 등
            recommended_batch_size = int(base_batch_size * 4 / multiplier)
        elif total_memory_gb >= 16:  # RTX 4080, RTX 3080 Ti 등
            recommended_batch_size = int(base_batch_size * 3 / multiplier)
        elif total_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080 등
            recommended_batch_size = int(base_batch_size * 2 / multiplier)
        elif total_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070 등
            recommended_batch_size = int(base_batch_size * 1.5 / multiplier)
        else:  # 8GB 미만
            recommended_batch_size = base_batch_size
        
        # 32의 배수로 조정 (GPU 효율성을 위해)
        recommended_batch_size = max(32, (recommended_batch_size // 32) * 32)
        
        return min(recommended_batch_size, base_batch_size * 4)  # 최대 4배까지만
        
    except Exception:
        # 오류 발생 시 기본값 반환
        return base_batch_size