"""
ContinuousCartPole Environment

CartPole-v1과 동일한 물리학을 가지지만 연속 행동 공간을 사용하는 환경입니다.
DQN과 DDPG를 동일한 환경에서 공정하게 비교하기 위해 설계되었습니다.

기존 CartPole-v1 (이산 행동):
- action_space: Discrete(2) {0: 왼쪽, 1: 오른쪽}
- 고정된 force_mag (10.0) 적용

ContinuousCartPole (연속 행동):  
- action_space: Box(low=-1, high=1, shape=(1,))
- 연속 행동을 force로 직접 매핑 [-force_mag, +force_mag]
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from typing import Optional


class ContinuousCartPole(gym.Env):
    """연속 행동 공간을 가진 CartPole 환경
    
    CartPole-v1의 물리학을 그대로 사용하되, 이산 행동 대신 연속 행동을 받습니다.
    이를 통해 DQN과 DDPG를 동일한 환경에서 공정하게 비교할 수 있습니다.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        """환경 초기화
        
        Args:
            render_mode: 렌더링 모드 ("human", "rgb_array", None)
        """
        # CartPole-v1과 동일한 물리 상수
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 실제로는 pole 길이의 절반
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0  # CartPole-v1과 동일
        self.tau = 0.02  # 시간 스텝 (seconds between state updates)

        # 각도 임계값 (실패 조건)
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # 연속 행동 공간: [-1, 1] 범위
        # -1은 최대 왼쪽 force, +1은 최대 오른쪽 force
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 상태 공간: CartPole-v1과 동일
        # [cart position, cart velocity, pole angle, pole angular velocity]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        """환경 스텝 실행
        
        Args:
            action: 연속 행동 [-1, 1] 범위의 1차원 배열
            
        Returns:
            observation: 다음 상태
            reward: 보상
            terminated: 에피소드 종료 여부 (실패)
            truncated: 에피소드 종료 여부 (시간 초과)
            info: 추가 정보
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # 연속 행동을 force로 변환
        # action[0] ∈ [-1, 1] → force ∈ [-force_mag, +force_mag]
        force = action[0] * self.force_mag

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # CartPole-v1과 동일한 물리 계산
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler 적분으로 상태 업데이트
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # 종료 조건 확인 (CartPole-v1과 동일)
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # 보상 계산 (CartPole-v1과 동일)
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # 방금 종료됨
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        # 추가 정보
        info = {
            "force_applied": force,
            "continuous_action": action[0],
            "cart_position": x,
            "pole_angle_degrees": math.degrees(theta)
        }

        if self.render_mode == "human":
            self.render()
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """환경 리셋
        
        Args:
            seed: 랜덤 시드
            options: 추가 옵션
            
        Returns:
            observation: 초기 상태
            info: 추가 정보
        """
        super().reset(seed=seed)
        
        # CartPole-v1과 동일한 초기화
        # 균등 분포에서 작은 랜덤 값으로 초기화
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None

        info = {
            "initial_state": self.state.copy(),
            "environment": "ContinuousCartPole-v0"
        }

        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), info

    def render(self):
        """환경 렌더링 (CartPole-v1과 동일한 시각화)"""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.screen.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.screen, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.screen, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.screen, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.screen, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.screen,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.screen,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.screen, 0, self.screen_width, carty, (0, 0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """환경 종료"""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# Gymnasium 등록용
gym.register(
    id="ContinuousCartPole-v0",
    entry_point="src.environments.continuous_cartpole:ContinuousCartPole",
    max_episode_steps=500,
    reward_threshold=475.0,
)


def create_continuous_cartpole_env(render_mode: Optional[str] = None) -> gym.Env:
    """ContinuousCartPole 환경 생성 함수
    
    Args:
        render_mode: 렌더링 모드
        
    Returns:
        ContinuousCartPole 환경 인스턴스
    """
    return ContinuousCartPole(render_mode=render_mode)


def compare_action_spaces():
    """기존 CartPole vs ContinuousCartPole 행동 공간 비교"""
    print("=" * 60)
    print("CartPole-v1 vs ContinuousCartPole-v0 비교")
    print("=" * 60)
    
    # 기존 CartPole-v1
    discrete_env = gym.make("CartPole-v1")
    print(f"CartPole-v1:")
    print(f"  Action Space: {discrete_env.action_space}")
    print(f"  Sample Actions: {[discrete_env.action_space.sample() for _ in range(5)]}")
    
    # ContinuousCartPole-v0
    continuous_env = create_continuous_cartpole_env()
    print(f"\nContinuousCartPole-v0:")
    print(f"  Action Space: {continuous_env.action_space}")
    print(f"  Sample Actions: {[continuous_env.action_space.sample() for _ in range(5)]}")
    
    print(f"\n물리 상수 비교:")
    print(f"  Force Magnitude: {continuous_env.force_mag}")
    print(f"  Gravity: {continuous_env.gravity}")
    print(f"  Time Step: {continuous_env.tau}")
    
    discrete_env.close()
    continuous_env.close()


if __name__ == "__main__":
    # 환경 테스트
    compare_action_spaces()
    
    # 간단한 실행 테스트
    print("\n" + "=" * 60)
    print("ContinuousCartPole 실행 테스트")
    print("=" * 60)
    
    env = create_continuous_cartpole_env()
    state, info = env.reset(seed=42)
    
    print(f"초기 상태: {state}")
    print(f"초기 정보: {info}")
    
    # 몇 스텝 실행
    for i in range(5):
        action = env.action_space.sample()  # 랜덤 연속 행동
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action[0]:.3f}")
        print(f"  Force: {info['force_applied']:.3f}")
        print(f"  State: [{next_state[0]:.3f}, {next_state[1]:.3f}, {next_state[2]:.3f}, {next_state[3]:.3f}]")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        
        if terminated:
            break
    
    env.close()
    print("\n환경 테스트 완료!")