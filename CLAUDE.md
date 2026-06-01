# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude(및 기여자)를 위한 가이드입니다. 프로젝트에서 **실제로 진행한 내용**, **코드 구조**, **실행 방법**, 그리고 **비판적 검토 / 개선점**을 정리합니다.

---

## 1. 프로젝트 개요

Microsoft **CyberBattleSim**(네트워크 침투 시뮬레이터)을 포크하여, 그 위에 **레드팀(공격자) 강화학습 에이전트**를 구현·평가하는 연구 프로젝트입니다.

- 공격자 행동을 **순차적 의사결정(MDP/POMDP)** 문제로 모델링
- 핵심 기여물: 기존 DQN을 **LSTM 기반 Deep Recurrent Q-Learning(DRQN)** 으로 확장
- 학술/교육 목적의 방어적 보안 연구. 실제 공격 도구가 아니라 추상화된 시뮬레이터 위의 RL 실험

> **베이스 vs 커스텀 구분이 중요합니다.** `cyberbattle/` 패키지 대부분(시뮬레이터, 환경, DQN/Tabular/Random 에이전트)은 Microsoft 원본 코드입니다. 이 프로젝트의 고유 작업은 아래 §3에 정리된 소수의 파일에 집중되어 있습니다.

---

## 2. 이 저장소에서 진행한 작업 (What's been done)

기여자: **HoyoenKim**(현재 git 사용자, 약 50커밋). 원본 대비 추가/변경한 핵심:

### 2.1 신규 에이전트
- **DRQN 에이전트** — [`cyberbattle/agents/baseline/agent_drqn.py`](cyberbattle/agents/baseline/agent_drqn.py)
  - 원본 `agent_dql.py`(DQN)를 LSTM 순환 신경망으로 확장한 **유일한 신규 알고리즘 코드**
  - 에피소드 단위 replay memory(`EpisodeReplayMemory`), LSTM Q-network(`DRQN`), 그리고 기존 `Learner` 인터페이스(`new_episode`/`on_step`/`exploit`/`explore`/`end_of_episode`)에 결합

### 2.2 실험 노트북 (jupytext `py:percent` 포맷)
[`notebooks/`](notebooks/) 아래에 4종 실험을 구성:
| 실험 | 파일 | 내용 |
|---|---|---|
| 1. Baseline (Chain) | `baseline_{random,rulebased,tabularq,dql,drql}.py` | Chain 네트워크에서 5개 에이전트 학습·평가 |
| 2. Transfer | `transfer_{dql,drql}.py` | 작은 Chain에서 학습 → 큰 Chain으로 일반화 (4→10, 4→20, 10→20) |
| 3. Toy CTF | `toyctf_{random,rulebased,tabularq,dql,drql}.py` | CTF형 환경에서 평가 |
| 4. Defender | `defender_{rulebased,dql,drql}.py` | 방어자(Blue team) 추가 후 강건성 재평가 |

- 공유 진입점/스크래치: [`notebook_benchmark.py`](notebooks/notebook_benchmark.py)
- 실행 스크립트: `notebooks/run_*.sh` (papermill로 노트북 실행 + 산출물 export)
- 방어자: 원본 제공 `ScanAndReimageCompromisedMachines`(주기적 스캔·탐지·reimage) 사용

### 2.3 결과물 & 문서
- [`figures/`](figures/) — 평가 GIF/PNG (README에서 참조)
- `notebooks/output/` — 실행된 노트북 + plot PNG (저장소에 커밋됨, §6 참고)
- [`doc/`](doc/) — 발표자료(`Final_CSRL.pptx`, `MidProgress_CSRL.pptx`)
- [`README.md`](README.md) — 문제정의, 모델, 실험, 결과 분석(§5), Future Work까지 상세 서술

### 2.4 보고된 핵심 결과 (README §5 요약)
- Chain/ToyCTF에서 **DQN·DRQN 모두 노드 100% 탐색·장악** 달성. DRQN이 더 적은 step으로 도달한다고 서술
- Transfer: DQN은 모든 설정 성공, DRQN은 4→20만 실패(학습 불안정성으로 해석)
- Defender 추가 시: DQN 장악률 7/11로 하락하지만 **DRQN은 11/11 유지** → 순환 메모리가 방어자 교란에 강건하다고 주장

> ⚠️ **이 결론들은 아래 §7 비판적 검토에서 지적하는 방법론적 한계로 인해 액면 그대로 신뢰하기 어렵습니다.**

---

## 3. 아키텍처 & 핵심 파일

```
cyberbattle/
├── simulation/        # [원본] 네트워크 모델, 액션, 결과, 그래프 생성 (model.py, actions.py)
├── _env/              # [원본] Gymnasium 환경 + 래퍼 (cyberbattle_env.py, defender.py, *_wrapper.py)
├── samples/           # [원본] 시나리오/토폴로지 (chainpattern/, toyctf/, active_directory/)
└── agents/baseline/
    ├── learner.py            # [원본] epsilon_greedy_search() 학습 루프 + Learner 인터페이스
    ├── agent_wrapper.py      # [원본] Feature/StateAugmentation/AbstractAction/EnvironmentBounds
    ├── agent_dql.py          # [원본] DQN 에이전트
    ├── agent_tabularqlearning.py / agent_randomcredlookup.py  # [원본] 베이스라인
    ├── agent_drqn.py         # ★[커스텀] DRQN 에이전트
    └── plotting.py           # [일부 커스텀] 결과 시각화
notebooks/             # ★[커스텀] 실험 노트북 + run_*.sh + output/
```

### DRQN 동작 방식 (agent_drqn.py 핵심)
- **State**: 전역 feature(`discovered_notowned_node_count`) + 노드별 feature(success/failed action, active property, node age)를 concat
- **Action**: `AbstractAction`(connect/local/remote 추상 액션) → 런타임에 구체 gym action으로 specialize
- **네트워크**(`DRQN`): `LazyLinear`(입력차원 자동추론) → ReLU → `LSTM(1층, hidden=256)` → `Linear`(action logits)
- **Replay**(`EpisodeReplayMemory`): 에피소드 단위로 transition 저장, 길이 `seq_len` 윈도우를 샘플링해 BPTT
- **Hidden state**: `new_episode()`에서 초기화, 행동 선택(`exploit`/`explore`) 시 `_advance_hidden_with_state`로 1회씩 carry. **학습(`optimize_model`) 시에는 hidden=None으로 시작** ← §7-1 참고
- **Target net**: `end_of_episode`에서 `target_update`(=10) 에피소드마다 동기화
- **Optimizer**: RMSprop, Huber(smooth L1) loss, grad clamp(±1)

---

## 4. 실행 방법

### 환경 구성 (Linux/WSL 기준 — README §3)
```bash
bash install-conda.sh      # conda 미설치 시
bash init.sh               # 'cybersim' 환경 생성
conda activate cybersim
```

### 실험 실행
```bash
# 개별 실험 (papermill로 노트북 실행 + output/ 에 산출물 저장)
./notebooks/run_baseline_drql.sh python3
./notebooks/run_transfer_drql.sh python3
./notebooks/run_toyctf_drql.sh python3
./notebooks/run_defender_drql.sh python3
```
- `.py` 노트북은 VSCode에서 셀 단위로 직접 실행 가능 (jupytext `py:percent`)
- papermill 파라미터(셀 태그 `parameters`): `gymid`, `env_size`, `iteration_count=9000`, `training_episode_count=50`, `eval_episode_count=5` 등

### 플랫폼 주의
- **개발/문서는 Windows이지만, 실험 실행은 Linux/WSL(conda `cybersim`)을 전제**로 합니다. `install-conda.sh`/`init.sh`는 bash 스크립트입니다.
- 테스트: `pytest` 기반 `*_test.py`가 원본에 존재. **단, 신규 `agent_drqn.py`에 대한 테스트는 없음**(§7-3).

---

## 5. 코드 컨벤션
- 커밋 메시지: `[feature]` / `[chore]` 접두사 (단, `[feture]`, `typeo` 등 오타 다수)
- DRQN 코드 주석은 **한국어**로 작성됨. 신규 코드 작성 시 주변 스타일(한국어 주석 + 영어 식별자)을 따를 것
- 노트북은 `.py`(jupytext) 원본을 편집하고, `.ipynb`/PNG는 papermill 산출물로 취급

---

## 6. 저장소 상태 메모
- `notebooks/output/`(46MB, 387파일)과 `figures/`(8.1MB)가 **git에 커밋**되어 있음. 전체 추적 파일 515개 중 ~75%가 생성 산출물
- `.gitignore`는 `notebooks/untracked/**`만 제외 — `notebooks/output/`는 추적됨

---

## 7. ⚠️ 비판적 검토 & 개선점 (Critical Review)

> 이 섹션은 향후 작업 시 **가장 먼저 손봐야 할 지점**을 우선순위대로 정리합니다. 현재 README의 결론은 아래 한계들로 인해 재검증이 필요합니다.

### 7-1. 🔴 [Critical] 모든 DRQN 실험이 `seq_len=1` — 순환성이 사실상 학습되지 않음
- **사실**: `baseline_drql.py:112`, `defender_drql.py:73`, `toyctf_drql.py:112`, `transfer_drql.py:125,205`, `notebook_benchmark.py:207` — 전부 `seq_len=1`.
- **함의**:
  - `EpisodeReplayMemory.sample()`이 길이 1 윈도우를 뽑고, `optimize_model()`의 BPTT가 **단일 timestep**에서만 일어남 → 시간적 credit assignment(순환의 본질)가 **학습되지 않음**.
  - 학습 시에는 `policy_net(state_batch, None)`로 **hidden=0**에서 시작([agent_drqn.py:402,407](cyberbattle/agents/baseline/agent_drqn.py)), 추론 시에는 hidden을 carry → **train(h=0) / inference(h≠0) 분포 불일치**.
  - 결과적으로 현재 "DRQN"은 *DQN + 추론 시에만 hidden을 흘리는 (학습되지 않은) LSTM 층*에 가깝습니다.
- **영향**: README §5 전반(5.1.1~5.1.4)이 DRQN의 우월성을 "순환 메모리가 시간적 맥락을 인코딩"한다고 귀인하지만, **현 구현으로는 그 메커니즘이 작동하지 않음**. 관측된 차이는 다른 요인(아래 7-2)일 가능성이 큼.
- **개선**: `seq_len`을 8~16 등으로 올리고, (이상적으로) replay 윈도우의 앞부분으로 hidden을 warm-up한 뒤 뒷부분에서만 loss를 계산하는 *burn-in*(R2D2 스타일)을 도입. seq_len 스윕으로 순환성의 실제 효과를 ablation.

### 7-2. 🔴 [Critical] DQN vs DRQN 비교가 공정하지 않음 (탐색 스케줄 상이)
- **사실**: DQN은 `epsilon_exponential_decay=5000, epsilon_minimum=0.10`([baseline_dql.py:104-105](notebooks/baseline_dql.py)), DRQN은 `decay=15000, minimum=0.05`([baseline_drql.py:117-118](notebooks/baseline_drql.py)).
- **함의**: DRQN이 3배 더 오래 탐색하고 마지막에 더 탐욕적으로 exploit함. **DRQN의 더 좋은 결과가 아키텍처 때문인지 더 잘 맞춘 탐색 스케줄 때문인지 분리 불가**(confound).
- **개선**: 모든 에이전트에 동일한 탐색 스케줄/예산을 적용하거나, 하이퍼파라미터를 각 에이전트별로 동일 프로토콜로 튜닝한 뒤 비교.

### 7-3. 🟠 통계적 엄밀성 부재 — seed 없음, 단일 평가
- `eval_episode_count=5`의 **단일 실행** 결과를 표로 보고. **여러 seed에 대한 평균/분산/신뢰구간이 없음**.
- 원본 저장소가 의도적으로 seed를 제거(`"updated firewall rules & removed seed"`)하여 **재현 불가**.
- Transfer 결과의 "Success/Fail"은 **이진 판정**이라 정량적 근거가 약함. DRQN 4→20 "실패"가 진짜 패턴인지 노이즈인지 구분 불가.
- README의 "Nodes Found / Nodes Exploited" 표는 노트북이 자동 산출하지 않음(노트북은 reward/availability plot만 생성) → **GIF 마지막 프레임을 수기로 읽은 값**으로 보임 → 주관적·단일 표본.
- **개선**: seed 고정 + 다중 seed(예: 5~10) 반복, 평균±표준편차/부트스트랩 CI 보고, found/exploited 지표를 노트북에서 자동 집계.

### 7-4. 🟠 `gamma=0.015` — 거의 근시안적 할인율
- 모든 에이전트가 `gamma=0.015`. γ²≈0.0002로 **사실상 즉시 보상만 고려**.
- README는 "긴 다단계 의존성/지연 보상"을 강조하는데 **할인율 설정과 정면 모순**. 장기 의존성이 중요하다면 γ는 0.9~0.99 영역이어야 함.
- **개선**: γ를 0.9~0.99로 올려 재실험하고, 보상 스케일/정규화를 함께 점검.

### 7-5. 🟡 학습 효율/안정성
- `update_q_function`이 **매 step마다 `optimize_model()` 호출**(batch=512) → 비용 큼. 보통 N step마다, 또는 충분히 쌓인 후 학습.
- 진행 중(미완료) 에피소드도 샘플 후보에 포함([agent_drqn.py:178](cyberbattle/agents/baseline/agent_drqn.py)) → 분석/디버깅 시 혼동 가능.
- **개선**: 학습 빈도 분리(train_every), warm-up step 후 학습 시작, target soft-update(τ) 옵션 검토.

### 7-6. 🟡 코드/저장소 위생
- [`notebook_benchmark.py:99-193`](notebooks/notebook_benchmark.py): DQL 전체 블록이 **삼중따옴표 문자열로 주석 처리**됨 → "benchmark"가 실제로는 DRQN+random만 실행. 죽은 코드.
- `agent_drqn.py`에 대한 **단위 테스트 없음**(`baseline_test.py`에 DRQN 미포함).
- 생성 산출물(`notebooks/output/` 46MB·387파일)이 저장소에 커밋 → 비대화. `.gitignore`에 추가하고 아티팩트는 릴리스/외부 저장소로.
- **개선**: 죽은 코드 제거 또는 파라미터화, 최소한의 DRQN smoke test 추가, output 디렉토리 git 추적 제외.

### 7-7. 🟡 README 정합성
- §4.4 소제목이 "4.4.1 Deep Q-Learning"이 두 번 중복(실제로는 Rule-Based / DQN / DRQN이어야 함).
- DRQN defender GIF의 alt-text가 `![defender_dql]`로 잘못됨.
- 오타 다수: `Evaulate`, `trnaasfer`, `feture`, `environmen`, `cybersimllm`, `typeo`.
- **개선**: 소제목/이미지 라벨 정정, 맞춤법 정리.

### 7-8. 코드 구현 정밀 검토 — DQL(원본, 동작) ↔ DRQN diff (추가 조사)

"DRQN이 잘 안 됐다"는 직감을 코드 레벨에서 추적한 결과. 단순 하이퍼파라미터가 아닌 **학습/평가 정합성** 관련 발견:

**🔴 [확인된 분기 · 성능 악화 유력] 실패한 exploit의 음성(negative) 학습 신호 제거**
- 원본 DQL은 exploit 시 선택한 abstract_action이 현재 상태에서 유효한 gym action으로 specialize되지 않으면 `(s, a, reward=0, s'=s)` transition을 replay에 push해 **"이 상태에서 이 추상 액션은 쓸모없다"를 학습**시킴 ([agent_dql.py:452-459](cyberbattle/agents/baseline/agent_dql.py)).
- DRQN은 이 부분을 **의도적으로 삭제**(`# ✅ 가짜 transition 학습 제거`, [agent_drqn.py:593-594](cyberbattle/agents/baseline/agent_drqn.py)).
- 영향: 이 환경은 추상 액션 다수가 특정 상태에서 무효라, 이 음성 신호가 Q를 유효 액션 쪽으로 미는 핵심 학습원. 제거 시 (1) 학습 transition 수 급감, (2) 무효 액션의 Q가 안 내려가 argmax가 자주 no-op → explore로 deflect → exploit 정책 품질 저하. **DRQN 부진의 직접 원인 후보.**

**🔴 [학습/평가 정합성] seq_len=1 → 평가 시 "학습된 적 없는 hidden"으로 행동 선택**
- 학습은 항상 `hidden=None`(=0)에서 1-step만 forward([agent_drqn.py:402,407](cyberbattle/agents/baseline/agent_drqn.py)) → 네트워크는 `Q(s | h=0)`만 학습.
- 그러나 rollout/평가의 행동 선택은 에피소드 동안 누적된 `h_t`(≠0)로 Q 계산([agent_drqn.py:520-548](cyberbattle/agents/baseline/agent_drqn.py)).
- 즉 **에피소드 첫 step(h=0)을 제외한 모든 step에서 학습 때 본 적 없는 hidden 조건의 (사실상 미학습) LSTM 출력으로 액션을 선택.** → "제대로 학습/평가한 게 맞나?"의 답: **순환 부분은 사실상 아니다.** 평가가 학습 분포와 어긋남.

**🟡 [Footgun] seq_len > 에피소드 길이면 학습이 조용히 안 됨**
- `sample()`은 길이 ≥ seq_len 에피소드가 없으면 `ValueError`, 이를 `optimize_model`이 잡아서 그냥 return ([agent_drqn.py:181-182, 398-399](cyberbattle/agents/baseline/agent_drqn.py)). seq_len을 키울 때 에피소드가 짧으면 **경고 없이 한 번도 학습하지 않을 수 있음**(seq_len=1 고정의 원인이 이 함정 회피였을 가능성).

**✅ [의심했으나 버그 아님 — 실측 확인]**
- *LazyLinear + optimizer 생성 순서*: `__init__`에서 첫 forward(파라미터 materialize) 전에 optimizer 생성. 구버전 PyTorch에서 문제될 수 있어 의심했으나 **torch 2.12에서 직접 테스트 결과 `fc_in` 정상 학습**(생성 순서 무관, |Δw| 동일). 버그 아님.
- *hidden 리셋/advance*: `epsilon_greedy_search`가 에피소드마다 `new_episode()` 호출([learner.py:251](cyberbattle/agents/baseline/learner.py))로 hidden 초기화, explore/exploit가 step당 정확히 1회 advance. 정상.

### 우선순위 로드맵
1. ✅ **(적용됨, §8)** `seq_len` 1→8 + 짧은 에피소드 패딩/마스크 — 순환성 실제 학습. (잔여: stored-state burn-in으로 train/infer hidden 완전 정합 — 선택)
2. ✅ **(적용됨, §8)** 실패한 exploit의 음성 학습 transition 복원 — 가장 의심되던 성능 악화 원인
3. **공정 비교 프로토콜**: 동일 탐색 스케줄/예산 (7-2) — 미적용
4. **seed 고정 + 다중 실행 + CI** 및 found/exploited **자동 집계** (7-3) — 미적용
5. **γ 재설정** 및 보상 스케일 점검 (7-4) — 미적용
6. 학습 루프 효율화, 죽은 코드/저장소 위생, 테스트, README 정정 (7-5~7-7) — 미적용

---

## 8. 적용된 수정 (Applied Fixes) — 2026-05-29

DRQN 학습/평가 정합성 버그 3건을 수정. "DRQN이 잘 안 됐다"의 직접 원인.

**(1) 실패 exploit 음성(negative) 학습 복원** — [agent_drqn.py](cyberbattle/agents/baseline/agent_drqn.py) `try_exploit_at_candidate_actor_states`
- 무효(undefined) exploit 시도 시 `(s, a, reward=0, s'=s)` transition을 다시 학습(원본 DQL과 동일). §7-8 ①의 주원인 해결.

**(2) `seq_len` 1 → 8** — DRQN 노트북 5개(`baseline_drql`, `defender_drql`, `toyctf_drql`, `transfer_drql`×2, `notebook_benchmark`)
- BPTT가 8-step 시퀀스로 흘러 **순환성이 실제로 학습**됨(에이전트 기본값도 8).
- 잔여 한계: 학습은 여전히 zero-start-state(h=0) 근사 — 표준 DRQN 방식(Hausknecht & Stone 2015). 완전 정합을 원하면 stored-state burn-in 추가 필요(다음 단계).

**(3) 짧은 에피소드 무학습 함정 제거** — `EpisodeReplayMemory.sample` + `optimize_model`
- seq_len보다 짧은 에피소드도 **패딩+마스크**로 학습에 포함. 길이≥seq_len 에피소드가 없으면 `ValueError`→조용히 무학습하던 경로 제거. 패딩 step은 `done=1`·`mask=0`으로 loss·부트스트랩에서 제외.

**검증(오프라인, torch 2.12 CPU):** 편집한 `EpisodeReplayMemory` 소스를 그대로 추출·실행 →
sample() 반환 shape `[B,8,D]`…, 마스크 정확성(유효 길이 {2,3,8}, 패딩 `done=1`·`state=0`), 마스킹 Huber loss 유한 + `fc_in` 그래디언트 흐름(|grad|>0) → **ALL CHECKS PASSED**.
전체 학습 재현은 WSL conda `cybersim` 환경에서 노트북 실행 필요.

**아직 안 고친 것 (코드 버그가 아닌 실험 설계 이슈):** DQL↔DRQN epsilon 스케줄 불일치(7-2), seed/다중실행/CI(7-3), γ=0.015(7-4), `notebook_benchmark` 죽은 코드·저장소 위생(7-5~7-6).

---

## 9. 공정·다중seed 재평가 결과 (2026-06-01)

§7-2(불공정 비교)·§7-3(seed 없음)를 교정해, 버그 수정된 DRQN을 **동일 스케줄·하이퍼파라미터·3 seed**로 DQN과 재비교. 러너: [`experiments/fair_compare.py`](experiments/fair_compare.py) (`chain|defender|toyctf`, `train_every`로 최적화 빈도↓). 결과 JSON: `experiments/results/fair_*.json`. README §5.1.5에도 반영.

| 시나리오 | DQN owned | DRQN(수정) owned |
|---|---|---|
| Defender (50ep×9000it, 3seed) | 6.47 ± 1.0 / 12 (reward 6757) | **7.33 ± 0.19 / 12** (reward 6392) |
| Chain (25ep×4000it, 3seed) | 12.0 ± 0.0 / 12 | 12.0 ± 0.0 / 12 (동률) |
| ToyCTF (10ep×1000it, 2seed, CPU) | **6.0 ± 0.0 / 10** | 5.17 ± 0.83 / 10 |

**핵심 (정직):**
- 수정 후 DRQN **정상 학습 확인**(chain 12/12, defender ~7.3) → "DRQN이 잘 안 됐다" 해소.
- 공정 비교 시 DRQN 우위는 **작고 주로 "일관성"**(±0.19 vs ±1.0)에 있음. reward는 오히려 DQN이 약간 높음. → **README의 "DRQN 11/11 vs DQN 7/11"은 단일 평가의 과장**이며 공정·다중seed에선 재현 안 됨(§7-3 입증).
- Chain은 둘 다 12/12로 변별력 없음(§5.1.1대로).
- toyctf(소규모·CPU, 2seed): **DRQN 우위 없음** — DQN 6.0 vs DRQN 5.17/10, DRQN이 오히려 덜 일관적. 둘 다 과소학습(§5.1.3의 9/9 미달) → "부분관측에서 DRQN 유리" 주장도 미지지. (에피소드 조기종료 없어 비용 커서 소규모·CPU로 평가; DQN seed-0 혼자 92분.)

**워크플로우 메모:** 로컬에서 코드 수정 → commit/push → 서버(newport, `~/hykim_ect/CSRL`) `git pull`로 sync. 학습은 newport conda env `hykim_ect`(전역설정 없음, `source ~/miniconda3/etc/profile.d/conda.sh` + `PYTHONNOUSERSITE=1`), GPU H100. 결과는 `~/hykim_ect/results/` → scp로 로컬 회수 후 git에 커밋.
