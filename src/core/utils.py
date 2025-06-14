import torch
import torch.nn as nn
from typing import Iterator, Tuple


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """타겟 네트워크의 소프트 업데이트 (Polyak averaging)
    
    DDPG에서 사용되는 점진적 타겟 네트워크 업데이트 방식입니다.
    target = tau * source + (1 - tau) * target
    
    Args:
        target_net: 업데이트할 타겟 네트워크
        source_net: 소스 네트워크
        tau: 업데이트 비율 (일반적으로 0.001 ~ 0.005)
    """
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


def get_device() -> torch.device:
    """사용 가능한 최적의 디바이스 반환"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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