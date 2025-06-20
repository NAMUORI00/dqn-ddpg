# DQN vs DDPG 결정적 정책 비교분석 연구 리포트

**생성일시**: 2025년 06월 15일 13:24:53
**연구 목적**: 강화학습 26강 DDPG 강의 기반 결정적 정책 특성 비교분석

---

## 📋 연구 개요

### 연구 배경
본 연구는 DQN(Deep Q-Network)과 DDPG(Deep Deterministic Policy Gradient) 알고리즘의 **결정적(deterministic) 정책** 특성을 코드 구현을 통해 비교분석합니다. 두 알고리즘은 모두 결정적 정책을 구현하지만, 그 방식과 적용 영역에서 근본적인 차이를 보입니다.

### 핵심 연구 문제
1. **결정적 정책의 구현 방식**: 암묵적(DQN) vs 명시적(DDPG) 접근법의 차이
2. **행동 공간 적응성**: 이산 vs 연속 행동 공간에서의 알고리즘 적합성
3. **탐험 전략**: ε-greedy vs 가우시안 노이즈의 효과성
4. **학습 안정성**: 각 알고리즘의 수렴 특성 및 성능 안정성

---

## 🔍 이론적 배경

### DQN (Deep Q-Network)
- **정책 유형**: 암묵적 결정적 정책
- **구현 방식**: π(s) = argmax_a Q(s,a)
- **특징**: Q-값 계산 후 최대값을 갖는 행동 선택
- **적용 영역**: 이산적 행동 공간

### DDPG (Deep Deterministic Policy Gradient)  
- **정책 유형**: 명시적 결정적 정책
- **구현 방식**: π(s) = μ(s)
- **특징**: 액터 네트워크가 직접 행동 출력
- **적용 영역**: 연속적 행동 공간

---

## 📊 실험 결과 분석

### 1. 학습 성능 비교

#### DQN 학습 성능
- **최종 성능**: 408.20
- **수렴 에피소드**: None
- **수렴 효율성**: 100.00%

#### DDPG 학습 성능
- **최종 성능**: -202.21
- **수렴 에피소드**: 50
- **수렴 효율성**: 12.50%

### 2. 결정적 정책 특성 분석

#### 정책 일관성 평가
- **DQN 결정성 점수**: 1.000
- **DDPG 결정성 점수**: 1.000
- **일관성 차이**: 0.000

#### 구현 메커니즘 비교

**DQN (암묵적 결정적 정책)**
- 메커니즘: argmax over Q-values
- 일관성률: 1.000
- Q-값 안정성: 0.000000
- 결정 신뢰도: 8.328

**DDPG (명시적 결정적 정책)**
- 메커니즘: direct actor output
- 일관성률: 1.000
- 출력 분산: 0.000000
- 노이즈 민감도: 0.153

### 3. 행동 공간 적응성 분석

#### 이산 행동 공간 (DQN)
- **환경**: CartPole-v1
- **행동 선택**: argmax 기반 명확한 선택
- **탐험 전략**: ε-greedy (확률적 무작위 선택)
- **주요 장점**: 계산 효율성, 이론적 안정성
- **한계점**: 연속 제어 불가능, 세밀한 제어 어려움

#### 연속 행동 공간 (DDPG)  
- **환경**: Pendulum-v1
- **행동 선택**: 액터 네트워크 직접 출력
- **탐험 전략**: 가우시안 노이즈 추가
- **주요 장점**: 연속적 정밀 제어, 무한 행동 공간 처리
- **한계점**: 학습 불안정성, 하이퍼파라미터 민감성

---

## 🔬 핵심 발견사항

### 1. 결정적 정책의 본질적 차이

- **정책 표현 방식**:
  - DQN: implicit (Q-values → argmax)
  - DDPG: explicit (actor network direct output)

- **행동 공간 처리**:
  - DQN: discrete (finite set)
  - DDPG: continuous (infinite set)

- **탐험 전략**:
  - DQN: epsilon-greedy (probabilistic)
  - DDPG: additive noise (deterministic + noise)

### 2. 알고리즘별 적합성

#### DQN의 강점
- 이산 행동 환경에서 자연스러운 적합성
- argmax 연산을 통한 명확하고 효율적인 행동 선택
- 상대적으로 안정적인 학습 과정
- 구현 및 디버깅의 용이성

#### DDPG의 강점  
- 연속 행동 환경에서의 필수적 역할
- 정밀한 연속 제어 가능
- 실제 로봇 제어 시스템 적용 가능
- 복잡한 연속 제어 문제 해결

### 3. 교육적 시사점

#### 알고리즘 선택 기준
1. **행동 공간 특성이 결정적 요인**
   - 이산 → DQN 자연스러운 선택
   - 연속 → DDPG 필수적 선택

2. **결정적 정책의 다양한 구현 방식**
   - 간접적 구현: 가치 함수 → 정책
   - 직접적 구현: 정책 함수 → 행동

3. **탐험-활용 트레이드오프의 환경별 최적화**
   - 이산 환경: ε-greedy의 단순함과 효과성
   - 연속 환경: 가우시안 노이즈의 세밀한 탐험

---

## 💡 결론 및 의의

### 연구 결론

1. **결정적 정책 구현의 성공**: 두 알고리즘 모두 각자의 영역에서 결정적 정책을 성공적으로 구현
2. **적응성 검증**: DDPG가 연속 행동 공간에서도 높은 결정성을 달성했으며, 이는 액터 네트워크의 안정적 학습을 의미합니다.
3. **상호 보완적 특성**: DQN과 DDPG는 경쟁 관계가 아닌 상호 보완적 알고리즘

### 교육적 가치
1. **이론과 실습의 연계**: 강화학습 이론을 실제 코드로 구현하여 직관적 이해 증진
2. **설계 철학의 이해**: 알고리즘 설계 시 행동 공간 특성을 고려한 접근법의 중요성
3. **실무 적용 가이드**: 실제 문제 해결 시 적절한 알고리즘 선택 기준 제시

### 향후 연구 방향
1. **하이브리드 접근법**: 이산-연속 혼합 행동 공간에서의 알고리즘 개발
2. **안정성 개선**: DDPG의 학습 안정성 향상 방법 연구
3. **확장성 연구**: 더 복잡한 환경에서의 성능 비교 분석

---

## 📚 참고자료

1. **DQN 논문**: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
2. **DDPG 논문**: "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)
3. **강화학습 26강**: DDPG 강의 내용
4. **구현 코드**: 본 프로젝트의 src/ 디렉토리 내 알고리즘 구현

---

**리포트 생성 완료 시각**: 2025-06-15 13:24:53
