# Episode 1–5 Adaptive Web Prototype

사용자가 위험자산·현금 비중을 5% 단위로 결정하는 순차 실험입니다. E1의
DP7까지 제출하면 같은 사용자의 E2 시나리오 3개 중 하나를 무작위 배정하며,
배정된 시나리오는 해당 에피소드가 끝날 때까지 유지됩니다. E2 완료 후에는
E1·E2 feature를 이용해 E3 손실 레벨을 배정합니다.

브라우저에는 현재 decision point까지의 정규화 가격만 반환됩니다. 60일 전체
가격, 실제 종목·기간, `market_phase`, `semantic_role`, `response_tag`는 백엔드의
비공개 시나리오 JSON에만 있습니다.

## 실행

백엔드:

```powershell
cd web/backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

프론트엔드:

```powershell
cd web/frontend
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173`에 접속합니다. 기본 DB는
`web/backend/data/experiment.db`입니다. 기존 `episode1.db`는 수정하거나
자동 이관하지 않습니다. DB 경로를 바꾸려면 `EXPERIMENT_DB_PATH` 환경변수를
사용합니다.

### Kimi-K3 분석 설정

`backend/.env.example`을 `backend/.env`로 복사한 뒤 NVIDIA API Catalog에서
발급한 서버 전용 API key를 설정합니다.

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
KIMI_MODEL=moonshotai/kimi-k3
KIMI_REASONING_EFFORT=low
KIMI_ANALYSIS_REVISION=v1
KIMI_TEMPERATURE=1.0
KIMI_MAX_TOKENS=16384
KIMI_TIMEOUT_SECONDS=120
KIMI_MAX_RETRIES=1
```

`.env`는 Git에서 제외되며 프론트엔드 번들에 포함되지 않습니다. 환경변수를
바꾼 뒤에는 backend 서버를 다시 시작합니다. 설문과 Episode 1~6를 모두 완료한
사용자에게만 `POST /api/analysis/runs`가 분석 작업을 생성하고,
`GET /api/analysis/runs/{analysis_id}`는 처리 상태와 사용자 공개용 결과만
반환합니다.

분석 캐시는 모델·reasoning effort·`KIMI_ANALYSIS_REVISION`을 조합한 설정
버전별로 분리됩니다. 따라서 `low`에서 `max`로 변경하면 기존 `low` 결과를
재사용하지 않습니다. schema 변경 없이 프롬프트나 기타 실행 의미를 바꿀 때는
`KIMI_ANALYSIS_REVISION`을 올립니다.

## API 흐름

- `POST /api/episode1/sessions`
- `POST /api/episode1/sessions/{session_id}/decisions`
- `POST /api/episode2/sessions` — 같은 사용자의 E1 완료 후에만 허용
- `POST /api/episode2/sessions/{session_id}/decisions`
- `POST /api/episode3/sessions` — 같은 사용자의 E2 완료 후에만 허용
- `POST /api/episode3/sessions/{session_id}/decisions`
- `POST /api/episode4/sessions` — 같은 사용자의 E3 완료 후에만 허용
- `POST /api/episode4/sessions/{session_id}/decisions`
- `POST /api/episode5/sessions` — 같은 사용자의 E4 완료 후에만 허용
- `POST /api/episode5/sessions/{session_id}/pre-decisions`
- `POST /api/episode5/sessions/{session_id}/post-decisions`
- 각 에피소드의 `GET /api/episodeN/sessions/{session_id}`

DP는 반드시 DP1부터 DP7까지 서버가 지정한 날짜에 순서대로 제출해야 합니다.
순서 변경, 중복 제출, 다른 시나리오 ID, 0.05 단위가 아닌 비중은 거절됩니다.

## DB 구조

- `sessions`: 사용자별 에피소드 배정과 진행 상태
- `behavior_events`: E1–E6 공통 append-only 행동 로그
- `e1_features`, `e2_features`, `e3_features`: 현재 구현된 에피소드별 feature
- `e4_features`: 현재 구현된 E4 전용 feature
- `e5_features`: 현재 구현된 E5 정보 반응 feature
- `e6_features`: 후속 구현을 위한 빈 스키마
- `profile_features`: 전체 에피소드 완료 후 통합 feature를 위한 빈 스키마

`behavior_events`에는 `episode`와 `scenario_id`가 함께 저장됩니다. UPDATE와
DELETE는 SQLite trigger가 차단합니다. Feature 테이블만 매 decision마다 전체
raw log로 재계산하여 upsert합니다.

## E2 feature 정의 (`e2_v3`)

- `recent_return_sensitivity`: DP3–DP5에서
  `delta_risk_share = alpha + beta * return_since_previous_dp + error`를 OLS로
  적합한 기울기 `beta`. DP3, DP4, DP5가 모두 제출된 뒤에만 계산하며 설명변수
  분산이 0이면 `null`입니다.
- `return_since_previous_dp = current_dp_price / previous_dp_price - 1`.
  DP1은 이전 DP가 없으므로 `null`입니다. `trailing_return_5d`도 비교 분석을
  위한 raw market-state feature로 계속 저장합니다.
- `gain_period_risk_escalation = risk_share_DP5 - risk_share_DP2`
- `uptrend_risk_exposure = Σ(risk_share_i × duration_i) / (day_DP5-day_DP2)`.
  닫힌 구간 DP2→DP3, DP3→DP4, DP4→DP5에 각 구간 시작 비중을 적용합니다.
- `e2_vs_e1_risk_shift = E2_uptrend_risk_exposure - E1_risk_exposure_auc`.
  E2가 완료된 뒤 같은 사용자의 E1 feature와 결합해 계산하며, 그전에는
  `null`입니다.
- `strong_gain_response`, `pullback_response_after_gain`,
  `renewed_rise_response`: 각각 DP5, DP6, DP7의 `delta_risk_share`.
- `uptrend_risk_increase_count`: DP3–DP5 중 `delta_risk_share > 0`인 횟수.
- `gain_period_hold_rate`: 관측된 DP3–DP5 중 변화가 0인 비율.
- `gain_adjustment_intensity`: DP3–DP5의 실제 비중 변경만 대상으로 한
  `mean(abs(delta_risk_share))`. 전부 hold이면 0입니다.
- `decision_time_median`: 해당 E2에서 관측된 전체 판단시간의 중앙값.
- `strong_gain_decision_time`, `correction_decision_time`: DP5, DP6 판단시간.

수익률에 대한 포트폴리오 비중 변화의 회귀 기울기를 민감도로 사용하는 방식은
최근 수익률과 자산배분 변화의 관계를 회귀로 추정하는 실증 금융 문헌의 구조를
따릅니다. 참고: [Review of Financial Studies](https://academic.oup.com/rfs/article/36/10/4233/7126484),
[NBER Working Paper 31317](https://www.nber.org/system/files/working_papers/w31317/w31317.pdf).

## E3 routing (`e3_routing_v1`)

- `routing_score = 0.7 × E1.risk_exposure_auc + 0.3 × E2.uptrend_risk_exposure`
- `[0,.2)→L1`, `[.2,.4)→L2`, `[.4,.6)→L3`, `[.6,.8)→L4`, `[.8,1]→L5`
- E1에서 한 번도 진입하지 않았으면 점수와 context gap보다 우선하여 L1
- `abs(E1 exposure - E2 exposure) > 0.30`이면 L3
- L1은 30%로 시작하고 10% floor, DP1–DP7에서 10–100% 자유 조정
- L2는 20%로 시작하고 10% floor, DP1–DP7에서 10–100% 자유 조정
- L3–L5는 E2 DP7 비중을 기본값으로 표시하되, 가격 공개 전 entry setup에서
  사용자가 0–100%로 다시 확정해야 E3가 시작됨

`assigned_level`, `routing_score`, `routing_version`, `scenario_max_drawdown`,
`entry_risk_share`, `allocation_floor`는 서버의 session context에 저장하며 API에는
레벨과 점수를 노출하지 않습니다. 내부 `E3_Lx_0x` ID도 API에서는 레벨을 제거한
`E3_01`–`E3_03`으로 변환합니다. E3 raw event에는 `allocation_floor`,
`floor_reached`, `initial_preallocated_risk_share`를 저장합니다.

0.7/0.3, context gap 0.30, 의미 있는 감축 10%p는 보편적인 규제·학술 cutoff가
아니라 이 MVP의 versioned heuristic입니다. 공식 자료도 risk tolerance가 개인의
목표·상황과 연결되며 단일한 정답이 없다고 설명합니다.
[FINRA risk tolerance guide](https://www.finra.org/investors/insights/know-your-risk-tolerance),
[AER loss/risk-taking study](https://www.aeaweb.org/articles?id=10.1257%2Faer.20140386).

## E3 feature 정의 (`e3_v3`)

- `loss_period_risk_change = risk_DP5 - risk_DP1`
- `drawdown_severity = -drawdown_from_peak`
- `drawdown_sensitivity = -(risk_DP5-risk_DP1) / (severity_DP5-severity_DP1)`;
  분모가 0 이하이면 `null`
- `first_meaningful_reduction_drawdown`: DP1 대비 누적 10%p 이상 처음 감소한
  DP2–DP5의 양수 drawdown severity. severity가 0이면 제외
- `loss_period_risk_exposure`: DP1→DP5 구간 시작 비중의 시간가중 평균
- `max_loss_period_reduction = min(risk_DP2..DP5) - risk_DP1`
- `recovery_reentry = risk_DP7 - risk_DP5`
- `drawdown_period_risk_increase_count`: DP2–DP5 중 `delta > 0` 횟수
- `drawdown_reduction_consistency`: DP2–DP5 중 `delta < 0` 횟수 / 4
- `reference_point_crossing_response`: 이전 DP 다음 날부터 현재 DP일까지의 daily
  path에서 100 하향 crossing이 처음 발견되면 그 직후 DP의 delta를 사용
- `trough_response`, `early_recovery_response`, `late_recovery_response`:
  각각 DP5, DP6, DP7 delta
- `post_loss_risk_persistence = risk_DP7 - risk_DP1`
- `recovery_reentry_ratio = (risk_DP7-risk_DP5)/(risk_DP1-risk_DP5)`;
  `risk_DP1 <= risk_DP5`이면 `null`
- `retention_score = clip(loss_period_risk_exposure / risk_DP1, 0, 1)`
- `reduction_score = 1 - clip((risk_DP1-min_loss_share) /`
  `(risk_DP1-allocation_floor), 0, 1)`
- `threshold_score`: DP5까지 유효 위험노출이 관측된 뒤 10%p 이상 감소가 없으면
  `1.0`; 감소가 있으면 `clip(abs(first_meaningful_reduction_drawdown /`
  `scenario_max_drawdown), 0, 1)`. DP5 이전, MDD 부재, DP1이 floor이면 `null`.
- `recovery_score`: DP7 이전에는 `null`. DP1 > DP5이면
  `clip(recovery_reentry_ratio, 0, 1)`, DP1 <= DP5이면 재진입이 필요하지 않았으므로
  `1.0`. 이 경우 raw `recovery_reentry_ratio`는 기존처럼 `null` 유지.
- `severity_factor = clip(abs(scenario_max_drawdown)/0.30, 0, 1)`
- `behavior_resilience_score = .40×retention + .30×reduction +`
  `.20×threshold + .10×recovery`
- `e3_loss_resilience_score = behavior_resilience_score × severity_factor`

분모가 0이거나 의미 있는 감축·회복이 없어 구성요소를 정의할 수 없으면 해당
score와 합성 score는 `null`로 둡니다. 임의의 대체점수는 넣지 않습니다.

손실 후 위험감수는 paper loss와 realized loss 등에 따라 방향이 달라질 수 있어,
E3 feature는 투자 유형을 즉시 판정하지 않고 행동값만 저장합니다.
[American Economic Review](https://www.aeaweb.org/articles?id=10.1257%2Faer.20140386),
[NBER household finance review](https://www.nber.org/papers/w22066.pdf).

## E4 routing (`e4_routing_v1`)

- E3 resilience가 계산 가능하면 `e4_routing_score = 0.4 × e3_routing_score +`
  `0.6 × e3_loss_resilience_score`,
  불가능하면 `e3_routing_score`를 사용하고 `e4_routing_fallback = 1`로 저장합니다.
- 기본 band는 `[0,.2)→V1`, `[.2,.4)→V2`, `[.4,.6)→V3`,
  `[.6,.8)→V4`, `[.8,1]→V5`입니다.
- 두 E3 점수가 모두 있을 때 `abs(e3_routing_score -
  e3_loss_resilience_score) >= .35`이면 V3으로 지정합니다.
- 마지막 단계에서 E3 L1/L2의 floor 도달 또는 L3-L5의 0% full exit가
  관측되면 fallback/충돌 여부와 무관하게 V2를 상한으로 적용합니다.
- E4 entry allocation은 E3 DP7의 최종 위험비중입니다.

E4 시나리오의 일간 20D 변동성은 해당 60일 내부의 20개 일간수익률에 대한
표본표준편차에 `sqrt(252)`를 곱해 Day 21부터 계산합니다. q25/q75와 percentile도
같은 시나리오 내부 값만 사용합니다. DP1과 DP2 사이에는 이전 DP의 20D 값이
없으므로 DP2의 `previous_dp_volatility_20d`, `delta_volatility_20d`,
`volatility_direction`은 `null`입니다.

## E4 feature 정의 (`e4_v1`)

- `volatility_sensitivity`: 계산 가능한 DP3-DP7에서
  `delta_risk_share = alpha + beta * delta_volatility_20d + error`의 OLS beta.
- `high_vol_risk_exposure`, `low_vol_risk_exposure`: 시나리오 q75 이상/q25 이하
  Day 21-D59 구간에 대해 DP 이후 비중을 piecewise constant로 적용한 기간가중 평균.
- `high_vs_low_vol_risk_shift = high exposure - low exposure`.
- rising/falling DP의 평균 변화는 각각
  `volatility_increase_response_mean`, `volatility_decrease_response_mean`입니다.
- `peak_volatility_response`: DP2-DP7 중 20D 변동성이 가장 높은 DP의 비중 변화.
- `volatility_adjustment_intensity`: rising/falling DP 중 실제 비중 변경만 분모에
  포함한 `mean(abs(delta_risk_share))`이며, 모든 해당 DP가 hold이면 0입니다.
- `volatility_compression_reentry`: 가장 큰 음의 변동성 변화가 발생한 DP의
  비중 변화입니다.
- `volatility_shift_decision_time`: 절댓값이 가장 큰 변동성 변화 DP의 판단시간입니다.

## E5 정보 충돌 실험 (`e5_v2`)

E5는 각 DP에서 같은 시장 snapshot을 유지한 채 PRE allocation을 먼저 받고, 그 뒤
서로 반대 polarity의 카드 두 장을 공개한 다음 POST allocation을 받습니다. PRE와
POST는 공통 append-only `behavior_events`에 각각 `pre_information`,
`post_information` event로 저장됩니다.

- 한 세션은 `NE`, `NC`, `EC`를 정확히 한 번씩 무작위 순열로 사용합니다.
- 세션 전체에 balanced polarity cycle A/B 중 하나를 한 번만 선택합니다. 어느
  cycle이든 news, expert, community가 positive와 negative로 각각 한 번씩
  등장합니다. source/sentiment별 template과 left/right 순서는 DP별로 독립적으로
  무작위화하며 모든 배정은 `e5_decision_assignments`와 session context에 저장합니다.
- 클라이언트에는 source 표시명, title, content, position만 반환합니다. sentiment,
  strength, pattern, template ID와 내부 pair metadata는 서버에만 남습니다.
- `external_information_sensitivity = mean(abs(information_delta_DP1..DP3))`
- adjustment/hold rate는 세 POST가 모두 완료된 뒤 각각 변경/hold 횟수를 3으로
  나눠 계산합니다.
- 정렬 방향은 POST 증가 시 positive source, 감소 시 negative source이며 hold이면
  `aligned_source=null`입니다.
- source별 alignment score와 함께 해당 source로 정렬된
  `abs(information_delta)`의 합을 alignment magnitude로 저장합니다.
- `information_counter_adjustment_count`는 PRE 변화와 POST 정보 변화가 모두 0이
  아니고 부호가 반대인 경우의 수입니다.

20개 일간수익률이 확보되지 않은 Day 16 또는 Day 20 snapshot에서는 시나리오 내부
`rolling_volatility_20d`를 임의 보정하지 않고 `null`로 기록합니다.

## E6 공통 Anchor/Calibration (`e6_v1`)

E6는 `E6_01`~`E6_03` 중 하나를 동일 확률로 한 번만 배정하고 session의
`scenario_id`와 `e6_assignment_version`에 보존합니다. 재접속할 때 기존 session을
복원하므로 scenario를 다시 추첨하지 않습니다. 모든 사용자는 이전 episode의 최종
비중을 상속하지 않고 위험자산 0%에서 DP1을 시작합니다.

- 시나리오별 가격과 DP days/semantic role은 서버의 `E6_*.json` 설정으로 관리합니다.
- E6 시작 전 E3 routing score, E3 behavior/loss resilience 및 profile version을
  session snapshot으로 저장합니다. E4 volatility tolerance는 anchor 비교에
  사용하지 않습니다.
- 전체 exposure AUC는 DP allocation을 piecewise constant로 적용하고 Day 1~60의
  59개 interval로 나눕니다.
- E6 behavior resilience는 E3와 같은 retention/reduction/threshold/recovery
  component 및 0.40/0.30/0.20/0.10 가중치를 공유하되 severity factor는 적용하지
  않습니다.
- `anchor_recovery_decision_time`은 early recovery DP5와 recovered-state DP6의
  판단시간 중앙값입니다.
- E6가 완료되고 두 consistency 값이 모두 유효하면 계산된
  `cross_context_consistency`를 `profile_features`에도 반영합니다.
- decision time은 브라우저 값이 아니라 서버가 session에 저장한 DP/phase 시작
  시각으로 계산합니다. 새로고침과 session 복원은 기존 시작 시각을 유지합니다.
- E6의 전체 60일 가격은 `backend/scenarios/E6_*.json`에만 존재하며 frontend에는
  현재 DP까지 API가 공개한 가격만 전달됩니다.

## 검증

```powershell
python -m unittest discover -s web/backend/tests -p "test_*.py" -v
cd web/frontend
npm run build
```

테스트에는 E1의 가상 행동 시퀀스, E2 계산식, E1 완료 전 E2 접근 차단,
공통 로그 저장, DP 순서·중복·5% 단위 검증, 미래 가격 비공개, raw log 불변성,
E1–E6 feature 테이블 골격 확인이 포함됩니다.
