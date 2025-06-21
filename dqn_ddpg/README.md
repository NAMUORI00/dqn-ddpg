# DQN vs DDPG Cross-Environment Comparison

교차 환경 강화학습 알고리즘 비교 프로젝트

## 핵심 기능

**4가지 알고리즘-환경 조합 지원:**
1. CartPole-v1 + DQN (자연스러운 조합)
2. CartPole-v1 + DDPG (DiscreteDDPG 사용)
3. Pendulum-v1 + DQN (DiscretizedDQN 사용)  
4. Pendulum-v1 + DDPG (자연스러운 조합)

## 환경 설정

```bash
# 로컬 conda 환경 (권장)
conda create --prefix ./.conda python=3.11 -y
./.conda/bin/pip install -r requirements.txt

# 또는 전역 환경
conda create -n ddpg_dqn python=3.11
conda activate ddpg_dqn
pip install -r requirements.txt
```

## 빠른 시작

```bash
# 1. 조합 테스트 (10 에피소드)
python scripts/quick_test_combinations.py

# 2. 전체 학습 (2000 에피소드 x 4조합)
python scripts/train_cross_environment.py

# 3. 특정 조합만 실행
python scripts/train_cross_environment.py --algorithm DQN --environment CartPole-v1
```

## 프로젝트 구조

```
src/
├── agents/           # DQN, DDPG, DiscreteDDPG, DiscretizedDQN
├── environments/     # 환경 팩토리 및 래퍼
├── networks/         # Actor, Critic, Q-Network
├── core/            # 버퍼, 유틸리티, 모니터링
└── visualization/   # 차트 및 비디오 생성

scripts/
├── quick_test_combinations.py    # 빠른 테스트
└── train_cross_environment.py    # 전체 학습

configs/             # YAML 설정 파일
results/             # 학습 결과
models/              # 저장된 모델
output/              # 시각화 결과
```

## 핵심 혁신

- **공정한 비교**: 동일한 환경에서 모든 알고리즘 테스트
- **행동 공간 적응**: DQN↔연속환경, DDPG↔이산환경
- **통합 인터페이스**: 모든 에이전트가 동일한 API 제공
- **포괄적 분석**: 수렴성, 성능, 학습 시간 비교