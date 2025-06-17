"""
정책 분석 시각화 모듈

DQN과 DDPG의 결정적 정책 특성을 분석하고 시각화합니다.
Q-값 분포, 액터 출력, 행동 선택 패턴, 탐험 영향 등을 포함합니다.

새로운 출력 구조:
- 확장자별 디렉토리 자동 분류 (PNG -> output/png/policy/, SVG -> output/svg/policy/ 등)
- 구조화된 파일명으로 정책 분석 결과 체계적 관리
- create_structured_filename()으로 알고리즘별, 환경별 파일명 자동 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Callable
import os

from ..core.base import MultiPlotVisualizer
from ..core.utils import (
    validate_data, create_color_palette,
    get_output_path_by_extension, create_structured_filename
)


class PolicyAnalysisVisualizer(MultiPlotVisualizer):
    """
    정책 분석 시각화 클래스
    
    DQN과 DDPG의 결정적 정책 특성을 다양한 관점에서 분석합니다.
    행동 선택 메커니즘, 일관성, 탐험 영향 등을 시각화합니다.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 정책 분석 전용 색상
        self.dqn_color = self.config.chart.dqn_color
        self.ddpg_color = self.config.chart.ddpg_color
        self.q_value_cmap = 'viridis'
        self.action_cmap = 'RdYlBu'
        
    def create_visualization(self, data: Dict[str, Any], **kwargs) -> str:
        """
        정책 분석 시각화 생성
        
        Args:
            data: 분석할 데이터 (agents, environments 포함)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 파일의 경로
        """
        if not self.validate_data(data):
            raise ValueError("유효하지 않은 데이터입니다")
        
        # 기본 결정적 정책 분석 생성
        return self.analyze_deterministic_policies(**data, **kwargs)
    
    def analyze_deterministic_policies(self,
                                     dqn_agent,
                                     ddpg_agent,
                                     dqn_env,
                                     ddpg_env,
                                     save_filename: str = "deterministic_policy_analysis.png") -> str:
        """
        결정적 정책 특성 분석 (원본 experiments/visualizations.py에서 번역된 버전)
        
        Args:
            dqn_agent: DQN 에이전트
            ddpg_agent: DDPG 에이전트
            dqn_env: DQN 환경
            ddpg_env: DDPG 환경
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 3, 
            figsize=(18, 12),
            title='DQN vs DDPG Deterministic Policy Analysis'
        )
        
        # DQN 분석
        self._analyze_dqn_policy(axes[0], dqn_agent, dqn_env)
        
        # DDPG 분석
        self._analyze_ddpg_policy(axes[1], ddpg_agent, ddpg_env)
        
        plt.tight_layout()
        
        file_path = self.save_figure(fig, save_filename)
        
        self.log_visualization_info(
            "Deterministic Policy Analysis",
            {
                "dqn_action_space": str(dqn_env.action_space),
                "ddpg_action_space": str(ddpg_env.action_space),
                "analysis_type": "policy_determinism"
            }
        )
        
        return file_path
    
    def _analyze_dqn_policy(self, axes: List[plt.Axes], dqn_agent, dqn_env):
        """DQN 정책 분석 (3개 서브플롯)"""
        
        # 1. Q-값 히트맵
        ax = axes[0]
        states = np.array([dqn_env.observation_space.sample() for _ in range(50)])
        q_values_matrix = []
        
        for state in states:
            q_vals = dqn_agent.get_q_values(state)
            q_values_matrix.append(q_vals)
        
        q_values_matrix = np.array(q_values_matrix)
        im = ax.imshow(q_values_matrix.T, aspect='auto', cmap=self.q_value_cmap)
        ax.set_xlabel('State Index')
        ax.set_ylabel('Action')
        ax.set_title('DQN: Q-value Distribution')
        plt.colorbar(im, ax=ax)
        
        # 2. 행동 선택 분포
        ax = axes[1]
        selected_actions = [np.argmax(q_vals) for q_vals in q_values_matrix]
        action_counts = np.bincount(selected_actions, minlength=dqn_env.action_space.n)
        
        bars = ax.bar(range(len(action_counts)), action_counts, 
                     color=self.dqn_color, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Action')
        ax.set_ylabel('Selection Count')
        ax.set_title('DQN: Action Selection Distribution')
        
        # 값 표시
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom')
        
        # 3. Q-값 차이 분석
        ax = axes[2]
        q_differences = []
        
        for q_vals in q_values_matrix:
            sorted_q = np.sort(q_vals)[::-1]
            if len(sorted_q) > 1:
                q_differences.append(sorted_q[0] - sorted_q[1])
        
        ax.hist(q_differences, bins=20, alpha=0.7, edgecolor='black',
               color=self.dqn_color)
        ax.set_xlabel('Q-value Difference (Best - Second Best)')
        ax.set_ylabel('Frequency')
        ax.set_title('DQN: Exploration Impact')
        
        # 통계 정보 추가
        if q_differences:
            mean_diff = np.mean(q_differences)
            ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_diff:.3f}')
            ax.legend()
    
    def _analyze_ddpg_policy(self, axes: List[plt.Axes], ddpg_agent, ddpg_env):
        """DDPG 정책 분석 (3개 서브플롯)"""
        
        # 1. 액터 출력 분포
        ax = axes[0]
        states = np.array([ddpg_env.observation_space.sample() for _ in range(100)])
        actions = []
        
        for state in states:
            action = ddpg_agent.get_deterministic_action(state)
            actions.append(action)
        
        actions = np.array(actions)
        
        # 각 행동 차원별로 히스토그램
        if actions.ndim > 1:
            for i in range(actions.shape[1]):
                ax.hist(actions[:, i], bins=20, alpha=0.7, 
                       label=f'Action Dim {i}', edgecolor='black')
        else:
            ax.hist(actions, bins=20, alpha=0.7, 
                   color=self.ddpg_color, edgecolor='black')
        
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
        ax.set_title('DDPG: Actor Output')
        if actions.ndim > 1:
            ax.legend()
        
        # 2. 행동 일관성 테스트
        ax = axes[1]
        test_state = ddpg_env.observation_space.sample()
        repeated_actions = []
        
        for _ in range(20):
            action = ddpg_agent.get_deterministic_action(test_state)
            repeated_actions.append(action)
        
        repeated_actions = np.array(repeated_actions)
        
        if repeated_actions.ndim > 1:
            std_per_dim = np.std(repeated_actions, axis=0)
            bars = ax.bar(range(len(std_per_dim)), std_per_dim,
                         color=self.ddpg_color, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Action Dimension')
            ax.set_ylabel('Standard Deviation')
            
            # 값 표시
            for bar, std_val in zip(bars, std_per_dim):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{std_val:.1e}',
                       ha='center', va='bottom', fontsize=8)
        else:
            std_val = np.std(repeated_actions)
            ax.bar([0], [std_val], color=self.ddpg_color, alpha=0.8)
            ax.set_ylabel('Standard Deviation')
            ax.text(0, std_val, f'{std_val:.1e}', ha='center', va='bottom')
        
        ax.set_title('DDPG: Action Consistency (Lower is More Deterministic)')
        ax.set_yscale('log')
        
        # 3. 노이즈의 영향
        ax = axes[2]
        deterministic_actions = []
        noisy_actions = []
        
        for state in states[:20]:
            det_action = ddpg_agent.select_action(state, add_noise=False)
            noisy_action = ddpg_agent.select_action(state, add_noise=True)
            
            deterministic_actions.append(det_action)
            noisy_actions.append(noisy_action)
        
        deterministic_actions = np.array(deterministic_actions)
        noisy_actions = np.array(noisy_actions)
        
        # L2 거리 계산
        if deterministic_actions.ndim > 1:
            differences = np.linalg.norm(noisy_actions - deterministic_actions, axis=1)
        else:
            differences = np.abs(noisy_actions - deterministic_actions)
        
        ax.hist(differences, bins=15, alpha=0.7, edgecolor='black',
               color=self.ddpg_color)
        ax.set_xlabel('Action Difference (L2 norm)' if deterministic_actions.ndim > 1 else 'Action Difference')
        ax.set_ylabel('Frequency')
        ax.set_title('DDPG: Action Change due to Noise')
        
        # 통계 정보 추가
        mean_diff = np.mean(differences)
        ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_diff:.3f}')
        ax.legend()
    
    def plot_action_space_analysis(self,
                                 dqn_agent,
                                 ddpg_agent,
                                 dqn_env,
                                 ddpg_env,
                                 save_filename: str = "action_space_analysis.png") -> str:
        """
        행동 공간 분석
        
        Args:
            dqn_agent: DQN 에이전트
            ddpg_agent: DDPG 에이전트  
            dqn_env: DQN 환경
            ddpg_env: DDPG 환경
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 2,
            figsize=(15, 10),
            title="Action Space Analysis"
        )
        
        # 1. DQN 행동 공간 커버리지
        self._plot_dqn_action_coverage(axes[0], dqn_agent, dqn_env)
        
        # 2. DDPG 행동 공간 커버리지
        self._plot_ddpg_action_coverage(axes[1], ddpg_agent, ddpg_env)
        
        # 3. 행동 다양성 비교
        self._plot_action_diversity(axes[2], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        # 4. 행동 집중도 분석
        self._plot_action_concentration(axes[3], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        plt.tight_layout()
        
        return self.save_figure(fig, save_filename)
    
    def _plot_dqn_action_coverage(self, ax: plt.Axes, dqn_agent, dqn_env):
        """DQN 행동 공간 커버리지"""
        states = [dqn_env.observation_space.sample() for _ in range(1000)]
        actions = []
        
        for state in states:
            action = dqn_agent.select_action(state, training=False)  # 탐험 없이
            actions.append(action)
        
        action_counts = np.bincount(actions, minlength=dqn_env.action_space.n)
        action_probs = action_counts / len(actions)
        
        bars = ax.bar(range(len(action_probs)), action_probs,
                     color=self.dqn_color, alpha=0.8, edgecolor='black')
        
        self.setup_subplot(ax, "DQN Action Coverage", "Action", "Probability")
        
        # 값 표시
        for bar, prob in zip(bars, action_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_ddpg_action_coverage(self, ax: plt.Axes, ddpg_agent, ddpg_env):
        """DDPG 행동 공간 커버리지"""
        states = [ddpg_env.observation_space.sample() for _ in range(1000)]
        actions = []
        
        for state in states:
            action = ddpg_agent.select_action(state, add_noise=False)
            actions.append(action)
        
        actions = np.array(actions)
        
        # 연속 행동 공간이므로 히스토그램으로 분포 표시
        if actions.ndim > 1:
            for i in range(actions.shape[1]):
                ax.hist(actions[:, i], bins=30, alpha=0.6,
                       label=f'Dim {i}', density=True)
            ax.legend()
        else:
            ax.hist(actions, bins=30, alpha=0.8, color=self.ddpg_color, density=True)
        
        self.setup_subplot(ax, "DDPG Action Coverage", "Action Value", "Density")
    
    def _plot_action_diversity(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """행동 다양성 비교"""
        
        # DQN 엔트로피 계산
        states = [dqn_env.observation_space.sample() for _ in range(500)]
        dqn_actions = []
        for state in states:
            action = dqn_agent.select_action(state, training=False)
            dqn_actions.append(action)
        
        dqn_counts = np.bincount(dqn_actions, minlength=dqn_env.action_space.n)
        dqn_probs = dqn_counts / len(dqn_actions)
        dqn_entropy = -np.sum(dqn_probs * np.log(dqn_probs + 1e-8))
        
        # DDPG 분산 계산 (다양성의 대리 지표)
        states = [ddpg_env.observation_space.sample() for _ in range(500)]
        ddpg_actions = []
        for state in states:
            action = ddpg_agent.select_action(state, add_noise=False)
            ddpg_actions.append(action)
        
        ddpg_actions = np.array(ddpg_actions)
        if ddpg_actions.ndim > 1:
            ddpg_variance = np.mean(np.var(ddpg_actions, axis=0))
        else:
            ddpg_variance = np.var(ddpg_actions)
        
        # 정규화된 값으로 비교
        algorithms = ['DQN\\n(Entropy)', 'DDPG\\n(Variance)']
        values = [dqn_entropy, ddpg_variance]
        colors = [self.dqn_color, self.ddpg_color]
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.8, edgecolor='black')
        
        self.setup_subplot(ax, "Action Diversity", "Algorithm", "Diversity Measure")
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
    
    def _plot_action_concentration(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """행동 집중도 분석"""
        
        states = [dqn_env.observation_space.sample() for _ in range(200)]
        
        # DQN Q-값 분산
        dqn_q_variances = []
        for state in states:
            q_values = dqn_agent.get_q_values(state)
            dqn_q_variances.append(np.var(q_values))
        
        # DDPG 액터 출력 일관성
        states = [ddpg_env.observation_space.sample() for _ in range(200)]
        ddpg_consistencies = []
        for state in states:
            repeated_actions = []
            for _ in range(5):  # 같은 상태에서 5번 행동 선택
                action = ddpg_agent.get_deterministic_action(state)
                repeated_actions.append(action)
            
            repeated_actions = np.array(repeated_actions)
            if repeated_actions.ndim > 1:
                consistency = np.mean(np.var(repeated_actions, axis=0))
            else:
                consistency = np.var(repeated_actions)
            ddpg_consistencies.append(consistency)
        
        # 박스 플롯으로 분포 비교
        data_to_plot = [dqn_q_variances, ddpg_consistencies]
        labels = ['DQN\\nQ-Value Variance', 'DDPG\\nAction Consistency']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 색상 설정
        colors = [self.dqn_color, self.ddpg_color]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        self.setup_subplot(ax, "Action Concentration", "Algorithm", "Concentration Measure")
    
    def plot_policy_stability_analysis(self,
                                     dqn_agent,
                                     ddpg_agent, 
                                     dqn_env,
                                     ddpg_env,
                                     save_filename: str = "policy_stability_analysis.png") -> str:
        """
        정책 안정성 분석
        
        Args:
            dqn_agent: DQN 에이전트
            ddpg_agent: DDPG 에이전트
            dqn_env: DQN 환경
            ddpg_env: DDPG 환경
            save_filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        fig, axes = self.create_subplot_grid(
            2, 2,
            figsize=(15, 10),
            title="Policy Stability Analysis"
        )
        
        # 1. 상태 섭동에 대한 민감도
        self._plot_state_perturbation_sensitivity(axes[0], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        # 2. 반복 실행 일관성
        self._plot_action_consistency(axes[1], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        # 3. 탐험 vs 착취 밸런스
        self._plot_exploration_exploitation_balance(axes[2], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        # 4. 정책 그래디언트 분석
        self._plot_policy_gradient_magnitude(axes[3], dqn_agent, ddpg_agent, dqn_env, ddpg_env)
        
        plt.tight_layout()
        
        return self.save_figure(fig, save_filename)
    
    def _plot_state_perturbation_sensitivity(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """상태 섭동에 대한 민감도 테스트"""
        base_state = dqn_env.observation_space.sample()
        perturbation_levels = np.linspace(0, 0.1, 20)
        
        dqn_action_changes = []
        ddpg_action_changes = []
        
        base_dqn_action = dqn_agent.select_action(base_state, training=False)
        base_ddpg_action = ddpg_agent.select_action(base_state, add_noise=False)
        
        for level in perturbation_levels:
            # 상태에 노이즈 추가
            noise = np.random.normal(0, level, size=base_state.shape)
            perturbed_state = base_state + noise
            
            # 클리핑 (관찰 공간 범위 내로)
            if hasattr(dqn_env.observation_space, 'low') and hasattr(dqn_env.observation_space, 'high'):
                perturbed_state = np.clip(perturbed_state, 
                                        dqn_env.observation_space.low, 
                                        dqn_env.observation_space.high)
            
            new_dqn_action = dqn_agent.select_action(perturbed_state, training=False)
            new_ddpg_action = ddpg_agent.select_action(perturbed_state, add_noise=False)
            
            # 행동 변화 측정
            dqn_change = 1 if new_dqn_action != base_dqn_action else 0
            ddpg_change = np.linalg.norm(new_ddpg_action - base_ddpg_action)
            
            dqn_action_changes.append(dqn_change)
            ddpg_action_changes.append(ddpg_change)
        
        ax.plot(perturbation_levels, dqn_action_changes, 'o-', 
               label='DQN', color=self.dqn_color, linewidth=2)
        ax.plot(perturbation_levels, ddpg_action_changes, 's-', 
               label='DDPG', color=self.ddpg_color, linewidth=2)
        
        self.setup_subplot(ax, "State Perturbation Sensitivity", "Perturbation Level", "Action Change")
        ax.legend()
    
    def _plot_action_consistency(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """반복 실행 일관성 테스트"""
        test_states = [dqn_env.observation_space.sample() for _ in range(50)]
        
        dqn_consistencies = []
        ddpg_consistencies = []
        
        for state in test_states:
            # DQN 일관성 (결정적이므로 항상 1이어야 함)
            dqn_actions = [dqn_agent.select_action(state, training=False) for _ in range(10)]
            dqn_consistency = len(set(dqn_actions)) == 1
            dqn_consistencies.append(1.0 if dqn_consistency else 0.0)
            
            # DDPG 일관성 (결정적이므로 분산이 낮아야 함)
            ddpg_actions = [ddpg_agent.select_action(state, add_noise=False) for _ in range(10)]
            ddpg_actions = np.array(ddpg_actions)
            if ddpg_actions.ndim > 1:
                ddpg_variance = np.mean(np.var(ddpg_actions, axis=0))
            else:
                ddpg_variance = np.var(ddpg_actions)
            ddpg_consistencies.append(1.0 / (1.0 + ddpg_variance))  # 분산의 역수
        
        # 박스 플롯
        data = [dqn_consistencies, ddpg_consistencies]
        labels = ['DQN', 'DDPG']
        
        box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = [self.dqn_color, self.ddpg_color]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        self.setup_subplot(ax, "Action Consistency", "Algorithm", "Consistency Score")
    
    def _plot_exploration_exploitation_balance(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """탐험 vs 착취 밸런스 분석"""
        states = [dqn_env.observation_space.sample() for _ in range(100)]
        
        # DQN 엡실론-그리디의 영향
        dqn_explorative_actions = []
        dqn_exploitative_actions = []
        
        for state in states:
            exploitative_action = dqn_agent.select_action(state, training=False)  # 탐험 없음
            explorative_action = dqn_agent.select_action(state, training=True)   # 탐험 있음
            
            dqn_exploitative_actions.append(exploitative_action)
            dqn_explorative_actions.append(explorative_action)
        
        # 탐험과 착취 행동의 차이 비율
        different_actions = sum(1 for exp, expl in zip(dqn_explorative_actions, dqn_exploitative_actions) 
                              if exp != expl)
        dqn_exploration_rate = different_actions / len(states)
        
        # DDPG 노이즈의 영향
        ddpg_deterministic_actions = []
        ddpg_noisy_actions = []
        
        for state in states:
            deterministic_action = ddpg_agent.select_action(state, add_noise=False)
            noisy_action = ddpg_agent.select_action(state, add_noise=True)
            
            ddpg_deterministic_actions.append(deterministic_action)
            ddpg_noisy_actions.append(noisy_action)
        
        ddpg_deterministic_actions = np.array(ddpg_deterministic_actions)
        ddpg_noisy_actions = np.array(ddpg_noisy_actions)
        
        # 평균 행동 변화
        if ddpg_deterministic_actions.ndim > 1:
            action_differences = np.linalg.norm(ddpg_noisy_actions - ddpg_deterministic_actions, axis=1)
        else:
            action_differences = np.abs(ddpg_noisy_actions - ddpg_deterministic_actions)
        
        ddpg_exploration_magnitude = np.mean(action_differences)
        
        # 정규화하여 비교
        algorithms = ['DQN\\n(ε-greedy)', 'DDPG\\n(Noise)']
        values = [dqn_exploration_rate, ddpg_exploration_magnitude]
        colors = [self.dqn_color, self.ddpg_color]
        
        bars = ax.bar(algorithms, values, color=colors, alpha=0.8, edgecolor='black')
        
        self.setup_subplot(ax, "Exploration Impact", "Algorithm", "Exploration Measure")
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
    
    def _plot_policy_gradient_magnitude(self, ax: plt.Axes, dqn_agent, ddpg_agent, dqn_env, ddpg_env):
        """정책 그래디언트 크기 분석 (근사)"""
        
        # 이 부분은 실제 그래디언트 계산이 필요하므로 
        # 간단한 Q-값 변화와 액터 출력 변화로 대체
        
        states = [dqn_env.observation_space.sample() for _ in range(50)]
        
        dqn_sensitivities = []
        ddpg_sensitivities = []
        
        for state in states:
            # DQN: Q-값의 최대-최소 차이로 정책 민감도 근사
            q_values = dqn_agent.get_q_values(state)
            dqn_sensitivity = np.max(q_values) - np.min(q_values)
            dqn_sensitivities.append(dqn_sensitivity)
            
            # DDPG: 작은 상태 변화에 대한 액터 출력 변화로 민감도 근사
            original_action = ddpg_agent.get_deterministic_action(state)
            
            # 작은 상태 섭동
            epsilon = 0.01
            perturbed_state = state + np.random.normal(0, epsilon, size=state.shape)
            perturbed_action = ddpg_agent.get_deterministic_action(perturbed_state)
            
            if original_action.ndim > 0:
                sensitivity = np.linalg.norm(perturbed_action - original_action) / epsilon
            else:
                sensitivity = np.abs(perturbed_action - original_action) / epsilon
            
            ddpg_sensitivities.append(sensitivity)
        
        # 분포 비교
        data = [dqn_sensitivities, ddpg_sensitivities]
        labels = ['DQN\\n(Q-value Range)', 'DDPG\\n(Action Sensitivity)']
        
        box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = [self.dqn_color, self.ddpg_color]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        self.setup_subplot(ax, "Policy Sensitivity", "Algorithm", "Sensitivity Measure")
        ax.set_yscale('log')