"""
스마트 녹화 스케줄링 시스템
성능 향상 감지, 마일스톤 달성, 주기적 녹화를 자동으로 관리
"""

import time
import yaml
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json


@dataclass
class ScheduleConfig:
    """스케줄링 설정"""
    # 초기 에피소드 (항상 녹화)
    initial_episodes: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # 주기적 녹화
    interval_episodes: int = 50
    
    # 마지막 N개 에피소드
    final_episodes: int = 10
    
    # 성능 향상 감지
    improvement_threshold: float = 0.1  # 10% 향상
    performance_window: int = 10  # 최근 N 에피소드 평균
    
    # 마일스톤 기반 녹화
    record_on_best_score: bool = True
    record_on_milestone: bool = True
    
    # 최대 선택적 녹화 수 (저장공간 관리)
    max_selective_recordings: int = 50
    
    @classmethod
    def from_yaml_config(cls, config_dict: Dict, algorithm: str) -> 'ScheduleConfig':
        """YAML 설정에서 스케줄 설정 생성"""
        dual_config = config_dict.get('dual_recording', {})
        selective_config = dual_config.get('selective_recording', {})
        schedule_config = selective_config.get('schedule', {})
        performance_config = selective_config.get('performance_triggers', {})
        
        return cls(
            initial_episodes=schedule_config.get('initial_episodes', [1, 2, 3, 5, 10]),
            interval_episodes=schedule_config.get('interval_episodes', 50),
            final_episodes=schedule_config.get('final_episodes', 10),
            improvement_threshold=performance_config.get('improvement_threshold', 0.1),
            record_on_best_score=performance_config.get('record_on_best_score', True),
            record_on_milestone=performance_config.get('record_on_milestone', True)
        )


@dataclass
class EpisodeData:
    """에피소드 데이터"""
    episode_id: int
    algorithm: str
    score: float
    episode_length: int
    timestamp: float
    is_terminated: bool
    additional_info: Dict = field(default_factory=dict)


class PerformanceTracker:
    """성능 추적기"""
    
    def __init__(self, algorithm: str, target_score: Optional[float] = None, 
                 milestones: Optional[List[float]] = None):
        self.algorithm = algorithm
        self.target_score = target_score
        self.milestones = set(milestones or [])
        
        # 성능 기록
        self.episode_history: List[EpisodeData] = []
        self.best_score = float('-inf') if algorithm == 'dqn' else float('-inf')
        self.achieved_milestones: Set[float] = set()
        
        # 통계
        self.stats = {
            'total_episodes': 0,
            'average_score': 0.0,
            'recent_average': 0.0,
            'improvement_rate': 0.0,
            'success_rate': 0.0
        }
    
    def add_episode(self, episode_data: EpisodeData):
        """에피소드 데이터 추가"""
        self.episode_history.append(episode_data)
        self.stats['total_episodes'] += 1
        
        # 최고 점수 업데이트
        if episode_data.score > self.best_score:
            self.best_score = episode_data.score
        
        # 마일스톤 확인
        self._check_milestones(episode_data.score)
        
        # 통계 업데이트
        self._update_stats()
    
    def _check_milestones(self, score: float):
        """마일스톤 달성 확인"""
        for milestone in self.milestones:
            if milestone not in self.achieved_milestones:
                # 알고리즘별 마일스톤 비교 로직
                if self._milestone_achieved(score, milestone):
                    self.achieved_milestones.add(milestone)
                    print(f"[INFO] 마일스톤 달성! {self.algorithm}: {milestone}")
    
    def _milestone_achieved(self, score: float, milestone: float) -> bool:
        """마일스톤 달성 여부 확인"""
        if self.algorithm == 'dqn':
            # DQN: 점수가 높을수록 좋음
            return score >= milestone
        elif self.algorithm == 'ddpg':
            # DDPG (Pendulum): 점수가 높을수록 좋음 (음수이지만 -200이 -1000보다 좋음)
            return score >= milestone
        else:
            return score >= milestone
    
    def _update_stats(self):
        """통계 업데이트"""
        if not self.episode_history:
            return
        
        scores = [ep.score for ep in self.episode_history]
        
        # 전체 평균
        self.stats['average_score'] = np.mean(scores)
        
        # 최근 평균 (최근 10개 에피소드)
        recent_scores = scores[-10:]
        self.stats['recent_average'] = np.mean(recent_scores)
        
        # 개선율 계산 (최근 vs 이전)
        if len(scores) >= 20:
            recent_avg = np.mean(scores[-10:])
            previous_avg = np.mean(scores[-20:-10])
            if previous_avg != 0:
                self.stats['improvement_rate'] = (recent_avg - previous_avg) / abs(previous_avg)
            else:
                self.stats['improvement_rate'] = 0.0
        
        # 성공률 (목표 점수 기준)
        if self.target_score is not None:
            successful_episodes = sum(1 for score in scores if self._milestone_achieved(score, self.target_score))
            self.stats['success_rate'] = successful_episodes / len(scores)
    
    def has_significant_improvement(self, threshold: float = 0.1, window: int = 10) -> bool:
        """유의미한 성능 향상이 있었는지 확인"""
        if len(self.episode_history) < window * 2:
            return False
        
        recent_scores = [ep.score for ep in self.episode_history[-window:]]
        previous_scores = [ep.score for ep in self.episode_history[-window*2:-window]]
        
        recent_avg = np.mean(recent_scores)
        previous_avg = np.mean(previous_scores)
        
        if previous_avg == 0:
            return False
        
        improvement = (recent_avg - previous_avg) / abs(previous_avg)
        return improvement >= threshold
    
    def is_new_best_score(self, score: float) -> bool:
        """새로운 최고 점수인지 확인"""
        return score > self.best_score
    
    def get_recent_milestone_achievements(self, window: int = 5) -> List[float]:
        """최근 달성된 마일스톤 반환"""
        if len(self.episode_history) < window:
            return []
        
        recent_episodes = self.episode_history[-window:]
        recent_milestones = []
        
        for episode in recent_episodes:
            for milestone in self.milestones:
                if (milestone in self.achieved_milestones and 
                    self._milestone_achieved(episode.score, milestone)):
                    recent_milestones.append(milestone)
        
        return list(set(recent_milestones))


class RecordingScheduler:
    """스마트 녹화 스케줄러
    
    다양한 조건에 따라 언제 고품질 녹화를 할지 결정
    """
    
    def __init__(self, config: ScheduleConfig, algorithm_configs: Dict[str, Dict]):
        self.config = config
        self.algorithm_configs = algorithm_configs
        
        # 성능 추적기들
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        
        # 녹화 기록
        self.scheduled_episodes: Set[int] = set()
        self.recorded_episodes: Set[int] = set()
        
        # 알고리즘별 추적기 초기화
        for algorithm, algo_config in algorithm_configs.items():
            target_score = algo_config.get('target_score')
            milestones = algo_config.get('milestones', [])
            
            self.performance_trackers[algorithm] = PerformanceTracker(
                algorithm, target_score, milestones
            )
        
        # 초기 에피소드 스케줄링
        self.scheduled_episodes.update(self.config.initial_episodes)
        
        print(f"[INFO] 녹화 스케줄러 초기화: {algorithm_configs.keys()}")
        print(f"[INFO] 초기 녹화 에피소드: {self.config.initial_episodes}")
    
    def should_record_episode(self, algorithm: str, episode_id: int, 
                            episode_score: Optional[float] = None) -> bool:
        """해당 에피소드를 녹화해야 하는지 결정
        
        Args:
            algorithm: 알고리즘 이름
            episode_id: 에피소드 ID
            episode_score: 에피소드 점수 (사후 평가용)
            
        Returns:
            녹화 여부
        """
        # 이미 녹화된 에피소드는 제외
        if episode_id in self.recorded_episodes:
            return False
        
        # 저장공간 제한 확인
        if len(self.recorded_episodes) >= self.config.max_selective_recordings:
            return False
        
        reasons = []
        should_record = False
        
        # 1. 스케줄된 에피소드 확인
        if episode_id in self.scheduled_episodes:
            reasons.append("scheduled")
            should_record = True
        
        # 2. 주기적 녹화
        if episode_id % self.config.interval_episodes == 0:
            reasons.append("interval")
            should_record = True
        
        # 3. 성능 기반 판단 (사후 평가)
        if episode_score is not None and algorithm in self.performance_trackers:
            tracker = self.performance_trackers[algorithm]
            
            # 새로운 최고 점수
            if self.config.record_on_best_score and tracker.is_new_best_score(episode_score):
                reasons.append("new_best")
                should_record = True
            
            # 유의미한 성능 향상
            if tracker.has_significant_improvement(self.config.improvement_threshold):
                reasons.append("improvement")
                should_record = True
        
        if should_record:
            self.scheduled_episodes.add(episode_id)
            print(f"[INFO] 에피소드 {episode_id} 녹화 예정: {', '.join(reasons)}")
        
        return should_record
    
    def report_episode_completion(self, algorithm: str, episode_id: int, 
                                episode_score: float, episode_length: int,
                                is_terminated: bool, additional_info: Dict = None):
        """에피소드 완료 보고 및 성능 추적
        
        Args:
            algorithm: 알고리즘 이름
            episode_id: 에피소드 ID
            episode_score: 최종 점수
            episode_length: 에피소드 길이
            is_terminated: 정상 종료 여부
            additional_info: 추가 정보
        """
        if algorithm not in self.performance_trackers:
            return
        
        # 에피소드 데이터 생성
        episode_data = EpisodeData(
            episode_id=episode_id,
            algorithm=algorithm,
            score=episode_score,
            episode_length=episode_length,
            timestamp=time.time(),
            is_terminated=is_terminated,
            additional_info=additional_info or {}
        )
        
        # 성능 추적기에 추가
        tracker = self.performance_trackers[algorithm]
        tracker.add_episode(episode_data)
        
        # 사후 녹화 판단
        should_record = self.should_record_episode(algorithm, episode_id, episode_score)
        
        # 마일스톤 달성 확인
        if self.config.record_on_milestone:
            recent_milestones = tracker.get_recent_milestone_achievements()
            if recent_milestones:
                self.scheduled_episodes.add(episode_id)
                print(f"[INFO] 마일스톤 달성으로 에피소드 {episode_id} 녹화: {recent_milestones}")
    
    def mark_episode_recorded(self, episode_id: int):
        """에피소드 녹화 완료 마킹"""
        self.recorded_episodes.add(episode_id)
        if episode_id in self.scheduled_episodes:
            self.scheduled_episodes.remove(episode_id)
    
    def get_next_scheduled_episodes(self, current_episode: int, lookahead: int = 10) -> List[int]:
        """다음 예정된 녹화 에피소드 목록"""
        upcoming = []
        for i in range(current_episode + 1, current_episode + lookahead + 1):
            if self.should_record_episode("dqn", i) or self.should_record_episode("ddpg", i):
                upcoming.append(i)
        return upcoming
    
    def get_recording_summary(self, algorithm: str = None) -> Dict:
        """녹화 요약 정보"""
        summary = {
            'total_scheduled': len(self.scheduled_episodes),
            'total_recorded': len(self.recorded_episodes),
            'remaining_slots': max(0, self.config.max_selective_recordings - len(self.recorded_episodes)),
            'config': {
                'initial_episodes': self.config.initial_episodes,
                'interval_episodes': self.config.interval_episodes,
                'improvement_threshold': self.config.improvement_threshold,
                'max_recordings': self.config.max_selective_recordings
            }
        }
        
        # 알고리즘별 성능 정보
        if algorithm and algorithm in self.performance_trackers:
            tracker = self.performance_trackers[algorithm]
            summary['performance'] = {
                'total_episodes': tracker.stats['total_episodes'],
                'best_score': tracker.best_score,
                'average_score': tracker.stats['average_score'],
                'recent_average': tracker.stats['recent_average'],
                'improvement_rate': tracker.stats['improvement_rate'],
                'achieved_milestones': list(tracker.achieved_milestones)
            }
        
        return summary
    
    def update_final_episodes_schedule(self, total_episodes: int):
        """최종 에피소드 스케줄 업데이트"""
        if total_episodes <= self.config.final_episodes:
            return
        
        final_episode_start = total_episodes - self.config.final_episodes + 1
        final_episodes = list(range(final_episode_start, total_episodes + 1))
        
        self.scheduled_episodes.update(final_episodes)
        print(f"[INFO] 최종 {self.config.final_episodes}개 에피소드 녹화 예약: {final_episodes}")
    
    def save_schedule_state(self, save_path: str):
        """스케줄 상태 저장"""
        state = {
            'scheduled_episodes': list(self.scheduled_episodes),
            'recorded_episodes': list(self.recorded_episodes),
            'config': {
                'initial_episodes': self.config.initial_episodes,
                'interval_episodes': self.config.interval_episodes,
                'final_episodes': self.config.final_episodes,
                'improvement_threshold': self.config.improvement_threshold,
                'max_selective_recordings': self.config.max_selective_recordings
            },
            'performance_stats': {}
        }
        
        # 성능 통계 추가
        for algorithm, tracker in self.performance_trackers.items():
            state['performance_stats'][algorithm] = {
                'total_episodes': tracker.stats['total_episodes'],
                'best_score': tracker.best_score,
                'stats': tracker.stats,
                'achieved_milestones': list(tracker.achieved_milestones)
            }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[INFO] 스케줄 상태 저장: {save_path}")
    
    def load_schedule_state(self, load_path: str):
        """스케줄 상태 로드"""
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            self.scheduled_episodes = set(state.get('scheduled_episodes', []))
            self.recorded_episodes = set(state.get('recorded_episodes', []))
            
            print(f"[INFO] 스케줄 상태 로드: {load_path}")
            print(f"[INFO] 예약된 에피소드: {len(self.scheduled_episodes)}개")
            print(f"[INFO] 완료된 에피소드: {len(self.recorded_episodes)}개")
            
        except FileNotFoundError:
            print(f"[INFO] 스케줄 상태 파일 없음: {load_path}")
        except Exception as e:
            print(f"[WARNING] 스케줄 상태 로드 실패: {e}")


def create_recording_scheduler_from_config(config_path: str) -> RecordingScheduler:
    """YAML 설정 파일에서 녹화 스케줄러 생성"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 알고리즘별 설정 추출
    algorithm_configs = config_dict.get('algorithms', {})
    
    # 스케줄 설정 생성
    schedule_config = ScheduleConfig.from_yaml_config(config_dict, 'dqn')  # 기본값으로 dqn 사용
    
    return RecordingScheduler(schedule_config, algorithm_configs)


def create_default_recording_scheduler() -> RecordingScheduler:
    """기본 녹화 스케줄러 생성"""
    default_config = ScheduleConfig()
    
    default_algorithms = {
        'dqn': {
            'environment': 'CartPole-v1',
            'target_score': 475,
            'milestones': [100, 200, 300, 400]
        },
        'ddpg': {
            'environment': 'Pendulum-v1',
            'target_score': -200,
            'milestones': [-1000, -500, -300, -200]
        }
    }
    
    return RecordingScheduler(default_config, default_algorithms)