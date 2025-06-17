#!/usr/bin/env python3
"""
DQN vs DDPG 전체 비교 실험
학습부터 평가까지 전체 파이프라인을 실행합니다.
"""

import os
import sys
import yaml
import torch
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import argparse

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.agents import DQNAgent, DDPGAgent
from src.environments.wrappers import create_dqn_env, create_ddpg_env
from src.core.utils import set_seed
from src.core.video_manager import VideoConfig, VideoManager
from src.core.dual_recorder import DualVideoRecorder, DualRecordingConfig, create_dual_recording_env
from src.core.recording_scheduler import RecordingScheduler, create_recording_scheduler_from_config
from experiments.metrics import MetricsTracker, evaluate_agent
# 새로운 시각화 모듈 import
from src.visualization.charts.learning_curves import LearningCurveVisualizer
from src.visualization.charts.comparison import ComparisonChartVisualizer
from src.visualization.charts.policy_analysis import PolicyAnalysisVisualizer
from src.visualization.core.config import VisualizationConfig
# Define create_experiment_report wrapper for backward compatibility
def create_experiment_report(results: Dict, output_dir: str):
    """Create comprehensive experiment report using new visualization system"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate learning curves
    if 'dqn' in results and 'ddpg' in results:
        visualizer = LearningCurveVisualizer()
        visualizer.plot_comparison(
            results['dqn']['metrics'],
            results['ddpg']['metrics'],
            os.path.join(output_dir, 'learning_curves.png')
        )
    
    # Generate comparison charts
    comparison_viz = ComparisonChartVisualizer()
    comparison_viz.create_comprehensive_comparison(results, output_dir)
    
    # Generate policy analysis if available
    if 'policy_analysis' in results:
        policy_viz = PolicyAnalysisVisualizer()
        policy_viz.analyze_policies(results['policy_analysis'], output_dir)


def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    # 절대 경로로 변환
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_dqn(config: Dict, dual_recorder: DualVideoRecorder = None, 
               scheduler: RecordingScheduler = None) -> Tuple[DQNAgent, Dict]:
    """DQN 훈련"""
    print("\n" + "="*50)
    print("DQN 훈련 시작")
    print("="*50)
    
    # 환경 설정
    env_name = config['environment']['name']
    state_dim = None
    action_dim = None
    
    # 임시 환경으로 차원 확인
    temp_env = create_dqn_env(env_name)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()
    
    # 에이전트 생성
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config['agent']
    )
    
    # 메트릭 추적
    metrics = MetricsTracker()
    
    # 훈련 루프
    for episode in tqdm(range(config['training']['episodes']), desc="DQN Training"):
        # 환경 설정 (녹화 여부 결정)
        should_record = False
        if scheduler:
            should_record = scheduler.should_record_episode('dqn', episode + 1)
        
        if dual_recorder and should_record:
            env = create_dual_recording_env(env_name, dual_recorder, 'dqn', episode + 1, True)
        else:
            env = create_dqn_env(env_name)
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(config['training']['max_steps_per_episode']):
            # 행동 선택
            action = agent.select_action(state)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 학습
            if agent.buffer.is_ready(agent.batch_size):
                train_metrics = agent.update()
                metrics.add_training_step(train_metrics)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 환경 종료
        env.close()
        
        # 에피소드 메트릭 추가
        metrics.add_episode(episode_reward, episode_length)
        
        # 스케줄러에 에피소드 완료 보고
        if scheduler:
            scheduler.report_episode_completion(
                'dqn', episode + 1, episode_reward, episode_length, 
                terminated, {'training_step': episode}
            )
            if should_record:
                scheduler.mark_episode_recorded(episode + 1)
        
        # 주기적 평가
        if episode % config['training']['eval_freq'] == 0:
            eval_env = create_dqn_env(env_name)
            eval_result = evaluate_agent(agent, eval_env, episodes=config['training']['eval_episodes'])
            eval_env.close()
            stats = metrics.get_current_stats()
            
            print(f"Episode {episode}: "
                  f"Reward {stats.get('mean_reward', 0):.2f}±{stats.get('std_reward', 0):.2f}, "
                  f"Eval {eval_result['mean_reward']:.2f}")
            
            if should_record:
                print(f"  📹 에피소드 {episode + 1} 녹화 완료")
    
    # 결과 반환
    result = {
        'episode_rewards': metrics.episode_rewards,
        'episode_lengths': metrics.episode_lengths,
        'training_losses': metrics.training_losses,
        'q_values': metrics.q_values
    }
    
    return agent, result


def train_ddpg(config: Dict, dual_recorder: DualVideoRecorder = None,
                scheduler: RecordingScheduler = None) -> Tuple[DDPGAgent, Dict]:
    """DDPG 훈련"""
    print("\n" + "="*50)
    print("DDPG 훈련 시작")
    print("="*50)
    
    # 환경 설정
    env_name = config['environment']['name']
    
    # 임시 환경으로 차원 확인
    temp_env = create_ddpg_env(env_name)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    # 에이전트 생성
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config['agent']
    )
    
    # 메트릭 추적
    metrics = MetricsTracker()
    
    # 워밍업 (무작위 행동)
    print("워밍업 중...")
    warmup_env = create_ddpg_env(env_name)
    for _ in range(config['training'].get('warmup_steps', 1000)):
        state, _ = warmup_env.reset()
        action = warmup_env.action_space.sample()
        next_state, reward, terminated, truncated, _ = warmup_env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        
        if done:
            continue
    warmup_env.close()
    
    # 훈련 루프
    for episode in tqdm(range(config['training']['episodes']), desc="DDPG Training"):
        # 환경 설정 (녹화 여부 결정)
        should_record = False
        if scheduler:
            should_record = scheduler.should_record_episode('ddpg', episode + 1)
        
        if dual_recorder and should_record:
            env = create_dual_recording_env(env_name, dual_recorder, 'ddpg', episode + 1, True)
        else:
            env = create_ddpg_env(env_name)
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        agent.reset_noise()
        
        for step in range(config['training']['max_steps_per_episode']):
            # 행동 선택 (노이즈 포함)
            action = agent.select_action(state, add_noise=True)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 학습
            if agent.buffer.is_ready(agent.batch_size):
                train_metrics = agent.update()
                metrics.add_training_step(train_metrics)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 환경 종료
        env.close()
        
        # 에피소드 메트릭 추가
        metrics.add_episode(episode_reward, episode_length)
        
        # 스케줄러에 에피소드 완료 보고
        if scheduler:
            scheduler.report_episode_completion(
                'ddpg', episode + 1, episode_reward, episode_length,
                terminated, {'training_step': episode}
            )
            if should_record:
                scheduler.mark_episode_recorded(episode + 1)
        
        # 주기적 평가
        if episode % config['training']['eval_freq'] == 0:
            eval_env = create_ddpg_env(env_name)
            eval_result = evaluate_agent(agent, eval_env, episodes=config['training']['eval_episodes'])
            eval_env.close()
            stats = metrics.get_current_stats()
            
            print(f"Episode {episode}: "
                  f"Reward {stats.get('mean_reward', 0):.2f}±{stats.get('std_reward', 0):.2f}, "
                  f"Eval {eval_result['mean_reward']:.2f}")
            
            if should_record:
                print(f"  📹 에피소드 {episode + 1} 녹화 완료")
    
    # 결과 반환
    result = {
        'episode_rewards': metrics.episode_rewards,
        'episode_lengths': metrics.episode_lengths,
        'training_losses': metrics.training_losses,
        'q_values': metrics.q_values
    }
    
    return agent, result


def run_final_evaluation(dqn_agent: DQNAgent, ddpg_agent: DDPGAgent) -> Tuple[Dict, Dict]:
    """최종 평가"""
    print("\n" + "="*50)
    print("최종 평가")
    print("="*50)
    
    # DQN 평가
    dqn_env = create_dqn_env()
    dqn_eval = evaluate_agent(dqn_agent, dqn_env, episodes=50, deterministic=True)
    dqn_env.close()
    
    print(f"DQN 최종 성능: {dqn_eval['mean_reward']:.2f}±{dqn_eval['std_reward']:.2f}")
    
    # DDPG 평가
    ddpg_env = create_ddpg_env()
    ddpg_eval = evaluate_agent(ddpg_agent, ddpg_env, episodes=50, deterministic=True)
    ddpg_env.close()
    
    print(f"DDPG 최종 성능: {ddpg_eval['mean_reward']:.2f}±{ddpg_eval['std_reward']:.2f}")
    
    return dqn_eval, ddpg_eval


def main():
    """메인 실험 실행"""
    parser = argparse.ArgumentParser(description='DQN vs DDPG 비교 실험')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--save-models', action='store_true', help='훈련된 모델 저장')
    parser.add_argument('--results-dir', type=str, default='results', help='결과 저장 디렉토리')
    
    # 이중 녹화 옵션
    parser.add_argument('--dual-video', action='store_true', help='이중 비디오 녹화 활성화')
    parser.add_argument('--video-config', type=str, default='configs/video_recording.yaml', 
                       help='비디오 녹화 설정 파일')
    parser.add_argument('--video-preset', type=str, choices=['low', 'medium', 'high', 'demo'],
                       help='비디오 품질 프리셋')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 작업 디렉토리를 루트로 변경
    os.chdir(project_root)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("DQN vs DDPG 비교 실험 시작")
    print(f"시드: {args.seed}")
    print(f"결과 저장 경로: {args.results_dir}")
    print(f"이중 비디오 녹화: {'활성화' if args.dual_video else '비활성화'}")
    
    # 이중 녹화 시스템 초기화
    dual_recorder = None
    scheduler = None
    
    if args.dual_video:
        print("\n🎬 이중 녹화 시스템 초기화 중...")
        
        # 비디오 설정 로드
        try:
            video_config_dict = load_config(args.video_config)
        except FileNotFoundError:
            print(f"[WARNING] 비디오 설정 파일을 찾을 수 없습니다: {args.video_config}")
            print("[INFO] 기본 설정을 사용합니다.")
            video_config_dict = {}
        
        # 비디오 매니저 설정
        if args.video_preset:
            video_config = VideoConfig.get_preset(args.video_preset)
        else:
            video_config = VideoConfig.from_yaml(args.video_config) if video_config_dict else VideoConfig()
        
        video_manager = VideoManager(video_config)
        
        # 이중 녹화 설정
        dual_config = DualRecordingConfig.from_yaml_config(video_config_dict)
        dual_recorder = DualVideoRecorder(video_manager, dual_config)
        
        # 스케줄러 설정
        try:
            scheduler = create_recording_scheduler_from_config(args.video_config)
        except:
            print("[INFO] 기본 스케줄러를 사용합니다.")
            from src.core.recording_scheduler import create_default_recording_scheduler
            scheduler = create_default_recording_scheduler()
        
        print("✅ 이중 녹화 시스템 초기화 완료")
        summary = scheduler.get_recording_summary()
        print(f"📋 녹화 계획: 초기 {len(scheduler.config.initial_episodes)}개, "
              f"주기 {scheduler.config.interval_episodes}에피소드마다, "
              f"최대 {summary['config']['max_recordings']}개 선택적 녹화")
    
    # 설정 로드
    dqn_config = load_config('configs/dqn_config.yaml')
    ddpg_config = load_config('configs/ddpg_config.yaml')
    
    # 시드 설정을 config에 반영
    dqn_config['environment']['seed'] = args.seed
    ddpg_config['environment']['seed'] = args.seed
    
    # DQN 훈련
    dqn_agent, dqn_results = train_dqn(dqn_config, dual_recorder, scheduler)
    
    # DDPG 훈련
    ddpg_agent, ddpg_results = train_ddpg(ddpg_config, dual_recorder, scheduler)
    
    # 최종 평가
    dqn_eval, ddpg_eval = run_final_evaluation(dqn_agent, ddpg_agent)
    
    # 결과 시각화 (새로운 모듈 사용)
    print("\n결과 시각화 중...")
    
    # 시각화 설정
    viz_config = VisualizationConfig()
    
    # 1. 학습 곡선 시각화
    with LearningCurveVisualizer(output_dir=args.results_dir, config=viz_config) as viz:
        learning_data = {
            'dqn': dqn_results,
            'ddpg': ddpg_results
        }
        learning_curves_path = viz.plot_comprehensive_learning_curves(
            dqn_results, ddpg_results,
            save_filename="learning_curves.png"
        )
        print(f"  ✅ 학습 곡선 저장: {learning_curves_path}")
    
    # 2. 성능 비교 시각화
    with ComparisonChartVisualizer(output_dir=args.results_dir, config=viz_config) as viz:
        comparison_data = {
            'dqn': {'episode_rewards': dqn_results['episode_rewards'], **dqn_eval},
            'ddpg': {'episode_rewards': ddpg_results['episode_rewards'], **ddpg_eval}
        }
        comparison_path = viz.plot_performance_comparison(
            comparison_data['dqn'], comparison_data['ddpg'],
            save_filename="performance_comparison.png"
        )
        print(f"  ✅ 성능 비교 저장: {comparison_path}")
    
    # 3. 결정적 정책 시각화 
    with PolicyAnalysisVisualizer(output_dir=args.results_dir, config=viz_config) as viz:
        dqn_env = create_dqn_env()
        ddpg_env = create_ddpg_env()
        
        policy_path = viz.visualize_deterministic_policies(
            dqn_agent, ddpg_agent, dqn_env, ddpg_env,
            save_filename="deterministic_policy_analysis.png"
        )
        print(f"  ✅ 정책 분석 저장: {policy_path}")
        
        dqn_env.close()
        ddpg_env.close()
    
    # 4. 실험 리포트 생성
    experiment_results = {
        'dqn_eval': dqn_eval,
        'ddpg_eval': ddpg_eval,
        'dqn_training': dqn_results,
        'ddpg_training': ddpg_results
    }
    
    create_experiment_report(
        experiment_results,
        save_path=f"{args.results_dir}/experiment_report.md"
    )
    
    # 모델 저장
    if args.save_models:
        print("훈련된 모델 저장 중...")
        dqn_agent.save(f"{args.results_dir}/dqn_model.pth")
        ddpg_agent.save(f"{args.results_dir}/ddpg_model.pth")
    
    # 이중 녹화 통계 출력
    if args.dual_video and dual_recorder and scheduler:
        print("\n🎬 비디오 녹화 통계:")
        recording_stats = dual_recorder.get_recording_stats()
        print(f"  📊 총 녹화 에피소드: {recording_stats['total_episodes']}")
        print(f"  📹 전체 녹화: {recording_stats['full_recordings']}")
        print(f"  ⭐ 선택적 녹화: {recording_stats['selective_recordings']}")
        print(f"  🎞️ 처리된 프레임: {recording_stats['total_frames_processed']:,}")
        print(f"  ⏱️ 평균 처리 시간: {recording_stats['average_processing_time']*1000:.2f}ms/프레임")
        
        # 알고리즘별 성능 요약
        for algorithm in ['dqn', 'ddpg']:
            summary = scheduler.get_recording_summary(algorithm)
            if 'performance' in summary:
                perf = summary['performance']
                print(f"\n  📈 {algorithm.upper()} 성능:")
                print(f"    • 최고 점수: {perf['best_score']:.2f}")
                print(f"    • 평균 점수: {perf['average_score']:.2f}")
                print(f"    • 달성 마일스톤: {perf['achieved_milestones']}")
        
        # 스케줄러 상태 저장
        schedule_save_path = f"{args.results_dir}/recording_schedule.json"
        scheduler.save_schedule_state(schedule_save_path)
        
        # 비디오 저장 경로 안내
        print(f"\n📁 비디오 저장 위치:")
        print(f"  • 전체 녹화: videos/{{algorithm}}/full/")
        print(f"  • 선택적 녹화: videos/{{algorithm}}/highlights/")
        print(f"  • 설정 및 통계: {schedule_save_path}")
    
    print("\n" + "="*50)
    print("실험 완료!")
    print("="*50)
    print(f"결과는 '{args.results_dir}' 디렉토리에 저장되었습니다.")
    print("\n주요 결과:")
    print(f"- DQN: {dqn_eval['mean_reward']:.2f}±{dqn_eval['std_reward']:.2f}")
    print(f"- DDPG: {ddpg_eval['mean_reward']:.2f}±{ddpg_eval['std_reward']:.2f}")
    
    print("\n결정적 정책 확인:")
    print("✓ DQN: Q-값 argmax를 통한 암묵적 결정적 정책")
    print("✓ DDPG: 액터 네트워크를 통한 명시적 결정적 정책")


if __name__ == "__main__":
    main()