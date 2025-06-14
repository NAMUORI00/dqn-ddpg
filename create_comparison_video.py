#!/usr/bin/env python3
"""
DQN과 DDPG 게임플레이 비교 영상 생성기
두 알고리즘의 실제 플레이 영상을 나란히 배치하여 비교 영상을 생성합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from typing import List, Tuple, Optional


class GameplayComparisonVideoCreator:
    """게임플레이 비교 영상 생성기"""
    
    def __init__(self, output_dir: str = "videos/comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_best_episodes(self, algorithm: str, video_type: str = "highlights") -> List[Path]:
        """최고 성능 에피소드 영상 찾기"""
        video_dir = Path(f"videos/{algorithm}/{video_type}")
        
        if not video_dir.exists():
            print(f"[WARNING] {video_dir} 디렉토리가 없습니다.")
            return []
        
        # 모든 mp4 파일 찾기
        videos = list(video_dir.glob("episode_*.mp4"))
        
        # 에피소드 번호로 정렬 (높은 번호가 나중 = 더 나은 성능)
        videos.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # 마지막 5개 선택
        return videos[-5:] if len(videos) >= 5 else videos
    
    def create_side_by_side_video(self, 
                                  dqn_video_path: Path, 
                                  ddpg_video_path: Path,
                                  output_path: Path,
                                  title: str = "DQN vs DDPG Comparison"):
        """두 영상을 나란히 배치한 비교 영상 생성"""
        
        print(f"[INFO] 비교 영상 생성 중...")
        print(f"  - DQN: {dqn_video_path}")
        print(f"  - DDPG: {ddpg_video_path}")
        
        # 비디오 캡처 객체 생성
        cap_dqn = cv2.VideoCapture(str(dqn_video_path))
        cap_ddpg = cv2.VideoCapture(str(ddpg_video_path))
        
        # 비디오 속성 가져오기
        fps = int(cap_dqn.get(cv2.CAP_PROP_FPS))
        width_dqn = int(cap_dqn.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_dqn = int(cap_dqn.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_ddpg = int(cap_ddpg.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_ddpg = int(cap_ddpg.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 비디오 크기 (두 영상을 나란히)
        output_width = width_dqn + width_ddpg + 20  # 중간에 여백
        output_height = max(height_dqn, height_ddpg) + 80  # 상단 제목 공간
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        
        while True:
            ret_dqn, frame_dqn = cap_dqn.read()
            ret_ddpg, frame_ddpg = cap_ddpg.read()
            
            # 둘 중 하나라도 끝나면 종료
            if not ret_dqn or not ret_ddpg:
                break
            
            # 출력 프레임 생성 (검은 배경)
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # 제목 영역 (상단 80px)
            cv2.rectangle(output_frame, (0, 0), (output_width, 80), (30, 30, 30), -1)
            
            # 제목 텍스트
            cv2.putText(output_frame, title, 
                       (output_width//2 - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # DQN 영상 (왼쪽)
            y_offset = 80 + (output_height - 80 - height_dqn) // 2
            output_frame[y_offset:y_offset+height_dqn, 0:width_dqn] = frame_dqn
            
            # DQN 라벨
            cv2.rectangle(output_frame, (0, 80), (width_dqn, 120), (0, 255, 136), -1)
            cv2.putText(output_frame, "DQN (CartPole)", 
                       (width_dqn//2 - 80, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # DDPG 영상 (오른쪽)
            x_offset = width_dqn + 20
            y_offset = 80 + (output_height - 80 - height_ddpg) // 2
            output_frame[y_offset:y_offset+height_ddpg, x_offset:x_offset+width_ddpg] = frame_ddpg
            
            # DDPG 라벨
            cv2.rectangle(output_frame, (x_offset, 80), (x_offset + width_ddpg, 120), (255, 107, 107), -1)
            cv2.putText(output_frame, "DDPG (Pendulum)", 
                       (x_offset + width_ddpg//2 - 90, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # 프레임 정보
            cv2.putText(output_frame, f"Frame: {frame_count}", 
                       (10, output_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 프레임 쓰기
            out.write(output_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"[INFO] 진행률: {frame_count} 프레임 처리됨")
        
        # 정리
        cap_dqn.release()
        cap_ddpg.release()
        out.release()
        
        print(f"[INFO] 비교 영상 생성 완료: {output_path}")
        print(f"[INFO] 총 {frame_count} 프레임")
        
        return output_path
    
    def create_multiple_comparisons(self):
        """여러 에피소드의 비교 영상 생성"""
        
        # 최고 성능 에피소드 찾기
        dqn_videos = self.find_best_episodes("dqn", "highlights")
        ddpg_videos = self.find_best_episodes("ddpg", "highlights")
        
        if not dqn_videos:
            print("[ERROR] DQN 영상을 찾을 수 없습니다.")
            return
        
        if not ddpg_videos:
            print("[ERROR] DDPG 영상을 찾을 수 없습니다.")
            return
        
        # 비교할 쌍 수
        num_comparisons = min(len(dqn_videos), len(ddpg_videos), 3)
        
        comparison_videos = []
        
        for i in range(num_comparisons):
            output_name = f"comparison_best_{i+1}.mp4"
            output_path = self.output_dir / output_name
            
            self.create_side_by_side_video(
                dqn_videos[-(i+1)],  # 뒤에서부터 (최신/최고 성능)
                ddpg_videos[-(i+1)],
                output_path,
                f"DQN vs DDPG - Best Performance #{i+1}"
            )
            
            comparison_videos.append(output_path)
        
        # 초기 vs 최종 비교
        if len(dqn_videos) > 1 and len(ddpg_videos) > 1:
            # 초기 에피소드 찾기
            dqn_early = self.find_best_episodes("dqn", "full")[:1]  # 첫 번째
            ddpg_early = self.find_best_episodes("ddpg", "full")[:1]
            
            if dqn_early and ddpg_early:
                output_path = self.output_dir / "comparison_early_vs_late.mp4"
                self.create_side_by_side_video(
                    dqn_early[0],
                    ddpg_early[0],
                    output_path,
                    "DQN vs DDPG - Early Training Stage"
                )
                comparison_videos.append(output_path)
        
        return comparison_videos
    
    def create_montage_video(self, max_duration: int = 60):
        """여러 에피소드의 하이라이트를 모은 몽타주 영상"""
        print("[INFO] 몽타주 영상 생성 시작...")
        
        # 하이라이트 영상 수집
        dqn_highlights = list(Path("videos/dqn/highlights").glob("*.mp4"))
        ddpg_highlights = list(Path("videos/ddpg/highlights").glob("*.mp4"))
        
        if not dqn_highlights or not ddpg_highlights:
            print("[WARNING] 하이라이트 영상이 부족합니다.")
            return None
        
        # TODO: 실제 몽타주 구현
        # 여러 영상에서 짧은 클립을 추출하여 편집
        
        print("[INFO] 몽타주 기능은 추후 구현 예정입니다.")
        return None


def main():
    parser = argparse.ArgumentParser(description="게임플레이 비교 영상 생성")
    
    parser.add_argument("--dqn-video", type=str, 
                       help="특정 DQN 영상 경로")
    parser.add_argument("--ddpg-video", type=str,
                       help="특정 DDPG 영상 경로")
    parser.add_argument("--output", type=str, default="videos/comparison",
                       help="출력 디렉토리")
    parser.add_argument("--auto", action="store_true",
                       help="자동으로 최고 성능 에피소드들 비교")
    parser.add_argument("--title", type=str, default="DQN vs DDPG Comparison",
                       help="영상 제목")
    
    args = parser.parse_args()
    
    creator = GameplayComparisonVideoCreator(args.output)
    
    if args.auto:
        # 자동으로 여러 비교 영상 생성
        videos = creator.create_multiple_comparisons()
        if videos:
            print(f"\n✅ 생성된 비교 영상들:")
            for video in videos:
                print(f"  - {video}")
    
    elif args.dqn_video and args.ddpg_video:
        # 특정 영상 비교
        output_path = Path(args.output) / "custom_comparison.mp4"
        creator.create_side_by_side_video(
            Path(args.dqn_video),
            Path(args.ddpg_video),
            output_path,
            args.title
        )
        print(f"\n✅ 비교 영상 생성 완료: {output_path}")
    
    else:
        print("사용법:")
        print("  자동 비교: python create_comparison_video.py --auto")
        print("  수동 비교: python create_comparison_video.py --dqn-video path/to/dqn.mp4 --ddpg-video path/to/ddpg.mp4")


if __name__ == "__main__":
    main()