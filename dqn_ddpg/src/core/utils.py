import torch
import torch.nn as nn
from typing import Iterator, Tuple, Optional, Union, List
import warnings
import os


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """타겟 네트워크의 소프트 업데이트 (Polyak averaging)
    
    DDPG에서 사용되는 점진적 타겟 네트워크 업데이트 방식입니다.
    target = tau * source + (1 - tau) * target
    
    Args:
        target_net: 업데이트할 타겟 네트워크
        source_net: 소스 네트워크
        tau: 업데이트 비율 (일반적으로 0.001 ~ 0.005)
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    """타겟 네트워크의 하드 업데이트
    
    DQN에서 사용되는 전체 복사 방식의 타겟 네트워크 업데이트입니다.
    
    Args:
        target_net: 업데이트할 타겟 네트워크
        source_net: 소스 네트워크
    """
    target_net.load_state_dict(source_net.state_dict())


def calculate_huber_loss(td_errors: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber 손실 계산
    
    큰 오류에 대해 덜 민감한 손실 함수입니다.
    
    Args:
        td_errors: TD 오류들
        delta: Huber 손실의 임계값
        
    Returns:
        Huber 손실
    """
    return torch.where(
        td_errors.abs() <= delta,
        0.5 * td_errors.pow(2),
        delta * (td_errors.abs() - 0.5 * delta)
    ).mean()


def get_device(device_id: Optional[Union[int, str]] = None) -> torch.device:
    """사용 가능한 최적의 디바이스 반환
    
    Args:
        device_id: 특정 GPU ID 또는 'cpu' 지정 (None이면 자동 선택)
        
    Returns:
        torch.device 객체
    """
    if device_id is not None:
        if isinstance(device_id, str) and device_id == 'cpu':
            return torch.device('cpu')
        elif isinstance(device_id, int) and torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                return torch.device(f'cuda:{device_id}')
            else:
                warnings.warn(f"GPU {device_id} not available. Using cuda:0")
                return torch.device('cuda:0')
    
    # 자동 선택
    if torch.cuda.is_available():
        # 가장 메모리가 많은 GPU 선택
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                free_memory.append(free)
            best_gpu = free_memory.index(max(free_memory))
            return torch.device(f'cuda:{best_gpu}')
        else:
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def get_gpu_memory_info(device: Optional[torch.device] = None) -> dict:
    """GPU 메모리 정보 반환
    
    Args:
        device: 조회할 디바이스 (None이면 현재 디바이스)
        
    Returns:
        메모리 정보 딕셔너리
    """
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        if device.type != 'cuda':
            return {'available': False}
        device = device.index if device.index is not None else 0
    
    if not torch.cuda.is_available():
        return {'available': False}
    
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    cached = torch.cuda.memory_reserved(device)
    free = total - allocated
    
    return {
        'available': True,
        'device': device,
        'total': total / 1024**3,  # GB
        'allocated': allocated / 1024**3,  # GB
        'cached': cached / 1024**3,  # GB
        'free': free / 1024**3,  # GB
        'utilization': allocated / total * 100  # %
    }


def optimize_gpu_memory():
    """GPU 메모리 최적화"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def enable_mixed_precision() -> torch.cuda.amp.GradScaler:
    """Mixed Precision Training을 위한 GradScaler 생성
    
    Returns:
        GradScaler 객체
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Mixed precision training disabled.")
        return None
    
    # AMP 지원 확인
    if not hasattr(torch.cuda.amp, 'GradScaler'):
        warnings.warn("Mixed precision training not supported in this PyTorch version.")
        return None
    
    return torch.cuda.amp.GradScaler()


def set_cuda_options(
    allow_tf32: bool = True,
    benchmark: bool = True,
    deterministic: bool = False
) -> None:
    """CUDA 최적화 옵션 설정
    
    Args:
        allow_tf32: TensorFloat-32 사용 허용 (Ampere 이상 GPU)
        benchmark: cudnn.benchmark 활성화
        deterministic: 결정적 알고리즘 사용 (성능 저하 가능)
    """
    if torch.cuda.is_available():
        # TF32 설정 (Ampere 이상 GPU에서 성능 향상)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        
        # cuDNN 설정
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        
        # 메모리 할당 최적화
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def parallel_data_loader(data_loader, device: torch.device, non_blocking: bool = True):
    """데이터 로더를 GPU 최적화된 버전으로 래핑
    
    Args:
        data_loader: 원본 데이터 로더
        device: 타겟 디바이스
        non_blocking: 비동기 전송 사용
        
    Yields:
        GPU로 전송된 데이터
    """
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            yield tuple(item.to(device, non_blocking=non_blocking) if torch.is_tensor(item) else item 
                       for item in batch)
        elif torch.is_tensor(batch):
            yield batch.to(device, non_blocking=non_blocking)
        else:
            yield batch


def set_seed(seed: int) -> None:
    """재현성을 위한 시드 설정
    
    Args:
        seed: 랜덤 시드
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 결정적 알고리즘 사용 (성능 저하 가능)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False