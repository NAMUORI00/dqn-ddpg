"""
비디오 매니저 - 비디오 녹화 및 관리 시스템
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shutil


@dataclass
class VideoConfig:
    """비디오 녹화 설정"""
    # 기본 설정
    fps: int = 30
    resolution: Tuple[int, int] = (640, 480)
    quality: str = "medium"
    
    # 저장 설정
    save_base_path: str = "videos"
    max_storage_gb: float = 10.0
    auto_cleanup: bool = True
    
    # 오버레이 설정
    show_overlay: bool = True
    show_episode: bool = True
    show_reward: bool = True
    show_steps: bool = True
    
    # 압축 설정
    compress_after_recording: bool = True
    keep_raw_files: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VideoConfig':
        """YAML 파일에서 설정 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict.get('video', {}))
    
    @classmethod
    def get_preset(cls, preset_name: str) -> 'VideoConfig':
        """사전 정의된 프리셋 반환"""
        presets = {
            'low': cls(
                fps=15,
                resolution=(320, 240),
                quality='low',
                max_storage_gb=5.0
            ),
            'medium': cls(
                fps=30,
                resolution=(640, 480), 
                quality='medium',
                max_storage_gb=10.0
            ),
            'high': cls(
                fps=30,
                resolution=(1280, 720),
                quality='high',
                max_storage_gb=20.0
            ),
            'demo': cls(
                fps=30,
                resolution=(640, 480),
                quality='high',
                show_overlay=True,
                max_storage_gb=5.0
            )
        }
        
        return presets.get(preset_name, presets['medium'])


class VideoManager:
    """비디오 파일 관리자
    
    비디오 파일 생성, 저장, 정리, 메타데이터 관리를 담당합니다.
    """
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.base_path = Path(config.save_base_path)
        self.metadata_file = self.base_path / "video_metadata.json"
        
        # 디렉토리 구조 생성
        self._setup_directories()
        
        # 메타데이터 로드
        self.metadata = self._load_metadata()
    
    def _setup_directories(self):
        """비디오 저장 디렉토리 구조 생성"""
        directories = [
            self.base_path,
            self.base_path / "dqn" / "full",
            self.base_path / "dqn" / "highlights", 
            self.base_path / "ddpg" / "full",
            self.base_path / "ddpg" / "highlights",
            self.base_path / "comparison",
            self.base_path / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_videos': 0,
                'total_size_mb': 0,
                'experiments': {},
                'created_at': None,
                'last_updated': None
            }
    
    def _save_metadata(self):
        """메타데이터 저장"""
        import time
        self.metadata['last_updated'] = time.time()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_video_path(self, algorithm: str, video_type: str, episode_id: int) -> Path:
        """비디오 파일 경로 생성
        
        Args:
            algorithm: 'dqn' 또는 'ddpg'
            video_type: 'full' 또는 'highlights'
            episode_id: 에피소드 ID
            
        Returns:
            비디오 파일 경로
        """
        return self.base_path / algorithm / video_type / f"episode_{episode_id:03d}.mp4"
    
    def get_video_config_for_episode(self, algorithm: str, episode_id: int, 
                                   is_highlight: bool = False) -> Dict:
        """에피소드용 비디오 설정 생성"""
        video_type = "highlights" if is_highlight else "full"
        save_path = self.base_path / algorithm / video_type
        
        # 하이라이트는 고품질, 전체는 저품질
        quality = "high" if is_highlight else "low"
        resolution = (640, 480) if is_highlight else (320, 240)
        fps = 30 if is_highlight else 15
        
        return {
            'save_path': save_path,
            'episode_id': episode_id,
            'fps': fps,
            'resolution': resolution,
            'quality': quality,
            'show_overlay': self.config.show_overlay
        }
    
    def register_video(self, algorithm: str, episode_id: int, 
                      video_type: str, file_size_mb: float, 
                      metadata: Dict):
        """비디오 등록 및 메타데이터 업데이트"""
        experiment_key = f"{algorithm}_{video_type}"
        
        if experiment_key not in self.metadata['experiments']:
            self.metadata['experiments'][experiment_key] = {
                'videos': [],
                'total_size_mb': 0,
                'count': 0
            }
        
        # 비디오 정보 추가
        video_info = {
            'episode_id': episode_id,
            'file_size_mb': file_size_mb,
            'metadata': metadata,
            'path': str(self.get_video_path(algorithm, video_type, episode_id))
        }
        
        self.metadata['experiments'][experiment_key]['videos'].append(video_info)
        self.metadata['experiments'][experiment_key]['total_size_mb'] += file_size_mb
        self.metadata['experiments'][experiment_key]['count'] += 1
        
        # 전체 통계 업데이트
        self.metadata['total_videos'] += 1
        self.metadata['total_size_mb'] += file_size_mb
        
        self._save_metadata()
    
    def check_storage_limit(self) -> bool:
        """저장 공간 제한 확인"""
        current_size_gb = self.metadata['total_size_mb'] / 1024
        return current_size_gb < self.config.max_storage_gb
    
    def cleanup_old_videos(self, keep_latest: int = 10):
        """오래된 비디오 정리
        
        Args:
            keep_latest: 보관할 최신 비디오 개수
        """
        if not self.config.auto_cleanup:
            return
        
        for experiment_key, experiment_data in self.metadata['experiments'].items():
            videos = experiment_data['videos']
            
            if len(videos) > keep_latest:
                # 에피소드 ID 기준으로 정렬
                videos.sort(key=lambda x: x['episode_id'])
                
                # 오래된 비디오 삭제
                videos_to_delete = videos[:-keep_latest]
                
                for video_info in videos_to_delete:
                    video_path = Path(video_info['path'])
                    if video_path.exists():
                        video_path.unlink()
                        print(f"[INFO] 오래된 비디오 삭제: {video_path}")
                
                # 메타데이터 업데이트
                experiment_data['videos'] = videos[-keep_latest:]
                
        self._save_metadata()
    
    def get_storage_summary(self) -> Dict:
        """저장 공간 사용량 요약"""
        total_size_gb = self.metadata['total_size_mb'] / 1024
        limit_gb = self.config.max_storage_gb
        usage_percentage = (total_size_gb / limit_gb) * 100
        
        return {
            'total_size_gb': round(total_size_gb, 2),
            'limit_gb': limit_gb,
            'usage_percentage': round(usage_percentage, 1),
            'available_gb': round(limit_gb - total_size_gb, 2),
            'total_videos': self.metadata['total_videos']
        }
    
    def export_video_list(self, algorithm: str = None) -> List[Dict]:
        """비디오 목록 내보내기"""
        video_list = []
        
        for experiment_key, experiment_data in self.metadata['experiments'].items():
            if algorithm and not experiment_key.startswith(algorithm):
                continue
                
            for video_info in experiment_data['videos']:
                video_list.append({
                    'experiment': experiment_key,
                    'episode_id': video_info['episode_id'],
                    'path': video_info['path'],
                    'size_mb': video_info['file_size_mb'],
                    'metadata': video_info['metadata']
                })
        
        return sorted(video_list, key=lambda x: (x['experiment'], x['episode_id']))
    
    def compress_videos(self, target_directory: str = None):
        """비디오 압축 (향후 구현)"""
        # TODO: FFmpeg을 사용한 비디오 압축 구현
        pass
    
    def create_backup(self, backup_path: str):
        """비디오 백업 생성"""
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터와 함께 전체 비디오 디렉토리 복사
        shutil.copytree(self.base_path, backup_path / "videos", dirs_exist_ok=True)
        print(f"[INFO] 비디오 백업 완료: {backup_path}")


def create_video_config_from_args(args) -> Optional[VideoConfig]:
    """명령행 인자에서 비디오 설정 생성"""
    if not getattr(args, 'dual_video', False) and not getattr(args, 'record_video', False):
        return None
    
    # 프리셋 사용
    if hasattr(args, 'video_preset') and args.video_preset:
        return VideoConfig.get_preset(args.video_preset)
    
    # 커스텀 설정
    config = VideoConfig()
    
    if hasattr(args, 'video_quality'):
        config.quality = args.video_quality
    
    if hasattr(args, 'video_resolution'):
        width, height = map(int, args.video_resolution.split('x'))
        config.resolution = (width, height)
    
    if hasattr(args, 'video_fps'):
        config.fps = args.video_fps
    
    return config