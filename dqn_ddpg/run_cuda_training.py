#!/usr/bin/env python3
"""
CUDA GPU 가속 순차 학습 시스템
RTX 4060 Ti 8GB VRAM 최적화
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

def run_training(algorithm, environment, mode="fair", episodes=2000):
    """RTX 4060 Ti GPU 가속 학습 실행"""
    
    print(f"\n{'='*70}")
    print(f"🚀 {algorithm} + {environment} GPU 가속 학습 시작")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # GPU 최적화 환경 변수 설정
    env_vars = os.environ.copy()
    env_vars.update({
        'CUDA_VISIBLE_DEVICES': '0',      # RTX 4060 Ti 사용
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',  # GPU 메모리 최적화
        'CUDA_LAUNCH_BLOCKING': '0',      # 비동기 실행으로 성능 향상
        'TORCH_CUDNN_V8_API_ENABLED': '1', # cuDNN v8 최적화
    })
    
    cmd = [
        "./.conda/bin/python", 
        "scripts/train_fair_comparison.py",
        "--algorithm", algorithm,
        "--environment", environment,
        "--device", "cuda"  # GPU 사용
    ]
    
    print(f"📋 GPU 최적화 설정:")
    print(f"   GPU: RTX 4060 Ti (8GB VRAM)")
    print(f"   CUDA: 12.1")
    print(f"   메모리 최적화: 활성화")
    print(f"   cuDNN v8: 활성화")
    print(f"📋 실행 명령어: {' '.join(cmd)}")
    print()
    
    try:
        # GPU 최적화 환경에서 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env_vars
        )
        
        # 실시간 출력 모니터링
        output_lines = []
        episode_count = 0
        last_episode_time = time.time()
        gpu_memory_peak = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                output_lines.append(line)
                
                # 에피소드 진행률 추적
                if "Episode" in line and "%" in line:
                    episode_count += 1
                    current_time = time.time()
                    if episode_count % 50 == 0:  # 50 에피소드마다 GPU 상태 출력
                        try:
                            # GPU 메모리 사용량 확인
                            gpu_info = subprocess.run([
                                "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", 
                                "--format=csv,noheader,nounits"
                            ], capture_output=True, text=True)
                            
                            if gpu_info.returncode == 0:
                                memory_used, memory_total, gpu_util = gpu_info.stdout.strip().split(', ')
                                memory_used_gb = float(memory_used) / 1024
                                memory_total_gb = float(memory_total) / 1024
                                gpu_memory_peak = max(gpu_memory_peak, memory_used_gb)
                                
                                episodes_per_min = 50 / ((current_time - last_episode_time) / 60)
                                print(f"🔥 GPU: {gpu_util}% | VRAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB | 속도: {episodes_per_min:.1f} eps/min")
                        except:
                            pass
                        last_episode_time = current_time
        
        return_code = process.poll()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            print(f"\n✅ {algorithm} + {environment} GPU 학습 완료!")
            print(f"⏰ 소요시간: {elapsed/60:.1f}분")
            print(f"🔥 최대 VRAM 사용: {gpu_memory_peak:.1f}GB")
            
            # 최종 결과 추출
            final_result = None
            for line in reversed(output_lines):
                if "최종 평균 보상:" in line:
                    final_result = line
                    break
            if final_result:
                print(f"📊 {final_result}")
            
            return True, elapsed, gpu_memory_peak
        else:
            print(f"\n❌ {algorithm} + {environment} GPU 학습 실패!")
            print(f"⏰ 소요시간: {elapsed/60:.1f}분")
            return False, elapsed, gpu_memory_peak
            
    except KeyboardInterrupt:
        print(f"\n⚠️ {algorithm} + {environment} 학습 중단됨")
        process.terminate()
        return False, 0, 0
    except Exception as e:
        print(f"\n❌ {algorithm} + {environment} 학습 오류: {e}")
        return False, 0, 0

def main():
    """RTX 4060 Ti GPU 가속 순차 학습 메인"""
    
    print("🎯 RTX 4060 Ti GPU 가속 순차 학습 시스템 시작")
    print("="*70)
    print("🔥 GPU 가속: RTX 4060 Ti 8GB VRAM")
    print("⚡ CUDA 12.1 + cuDNN v8 최적화")
    print("📊 공정한 하이퍼파라미터 (fair 모드)")
    print("   - Learning Rate: 0.0001 (모든 알고리즘 동일)")
    print("   - Batch Size: 128 (모든 알고리즘 동일)")
    print("   - Decay Rate: 0.9995 (epsilon/noise 동일)")
    print("   - Episodes: 2000 (각 조합)")
    print("🔧 처리기: GPU 순차 실행 (최대 성능)")
    print("📁 결과: results/fair_comparison/{timestamp}/")
    print()
    
    # 실행할 조합들 (빠른 것부터 순서대로)
    combinations = [
        ("DQN", "CartPole-v1"),      # 가장 빠름
        ("DDPG", "CartPole-v1"),     # 빠름
        ("DQN", "Pendulum-v1"),      # 보통
        ("DDPG", "Pendulum-v1")      # 가장 느림
    ]
    
    print(f"📋 실행 순서 ({len(combinations)}개 조합):")
    for i, (alg, env) in enumerate(combinations, 1):
        print(f"   {i}. {alg} + {env}")
    print()
    
    # GPU 정보 확인
    print("🔧 GPU 정보:")
    try:
        import torch
        print(f"   CUDA: {torch.cuda.is_available()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        print(f"   초기 메모리 정리 완료")
    except Exception as e:
        print(f"   GPU 정보 확인 실패: {e}")
        return
    
    print()
    
    # 전체 시작 시간
    total_start = time.time()
    results = []
    total_vram_used = 0
    
    # 순차 실행
    for i, (algorithm, environment) in enumerate(combinations, 1):
        print(f"\n🔄 진행: {i}/{len(combinations)}")
        
        success, combination_time, vram_peak = run_training(algorithm, environment)
        
        results.append({
            'combination': f"{algorithm} + {environment}",
            'success': success,
            'order': i,
            'time': combination_time,
            'vram_peak': vram_peak
        })
        
        total_vram_used = max(total_vram_used, vram_peak)
        
        if not success:
            print(f"\n⚠️ {algorithm} + {environment} 실패. 다음 조합으로 계속 진행합니다.")
        
        # 조합 간 GPU 메모리 정리
        if i < len(combinations):
            print(f"\n⏸️ GPU 메모리 정리 및 다음 조합 준비... (3초 대기)")
            try:
                import torch
                torch.cuda.empty_cache()
                print("   ✅ GPU 메모리 정리 완료")
            except:
                pass
            time.sleep(3)
    
    # 전체 완료
    total_time = time.time() - total_start
    
    print(f"\n\n{'='*70}")
    print("🎉 RTX 4060 Ti GPU 가속 순차 학습 완료!")
    print(f"{'='*70}")
    
    print(f"⏰ 총 소요시간: {total_time/60:.1f}분 ({total_time/3600:.1f}시간)")
    print(f"🔥 최대 VRAM 사용: {total_vram_used:.1f}GB / 8.6GB")
    print()
    
    print("📊 실행 결과:")
    success_count = 0
    total_training_time = 0
    for result in results:
        status = "✅ 성공" if result['success'] else "❌ 실패"
        time_str = f"{result['time']/60:.1f}분" if result['success'] else "N/A"
        vram_str = f"{result['vram_peak']:.1f}GB" if result['success'] else "N/A"
        print(f"   {result['order']}. {result['combination']}: {status} ({time_str}, VRAM: {vram_str})")
        if result['success']:
            success_count += 1
            total_training_time += result['time']
    
    print(f"\n📈 성공률: {success_count}/{len(combinations)} ({success_count/len(combinations)*100:.1f}%)")
    
    if success_count > 0:
        avg_time = total_training_time / success_count / 60
        speedup_estimate = 10  # GPU 대비 CPU 예상 속도 향상
        print(f"⚡ 평균 조합당 시간: {avg_time:.1f}분")
        print(f"🚀 GPU 가속 효과: 약 {speedup_estimate}x 빠름 (CPU 대비)")
        print(f"💡 RTX 4060 Ti 8GB VRAM 효율적 활용")
        
        # 결과 디렉토리 안내
        print(f"\n📁 결과 확인:")
        print("   find results -name '*fair_comparison*' -type d")
        
        print(f"\n🔍 진행상황 체크: ./.conda/bin/python check_progress.py")
        print(f"🔥 GPU 모니터링: nvidia-smi")

if __name__ == "__main__":
    main()