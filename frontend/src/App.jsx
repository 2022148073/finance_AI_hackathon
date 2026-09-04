import { useEffect, useMemo, useRef, useState } from "react";
import AccessGate from "./AccessGate.jsx";
import Survey from "./Survey.jsx";

const API_BASE = window.__API_BASE_URL__ ?? "http://127.0.0.1:8000";
const USER_ID_KEY = "experiment_user_id";
const PARTICIPANT_ID_KEY = "experiment_participant_id";
const ANALYSIS_POLL_INTERVAL_MS = 1500;
// Kimi-K3 always reasons before returning structured output. Two sequential
// calls may take several minutes, so keep a finite but sufficiently wide UI
// deadline while the backend enforces a separate timeout per model call.
const ANALYSIS_POLL_TIMEOUT_MS = 25 * 60 * 1000;

function getUserId() {
  const existing =
    window.localStorage.getItem(USER_ID_KEY) ??
    window.localStorage.getItem("episode1_user_id");
  if (existing) window.localStorage.setItem(USER_ID_KEY, existing);
  if (existing) return existing;
  const created = `web_${crypto.randomUUID().replaceAll("-", "")}`;
  window.localStorage.setItem(USER_ID_KEY, created);
  return created;
}

function getParticipantId() {
  const existing = window.localStorage.getItem(PARTICIPANT_ID_KEY);
  if (existing) return existing;
  const created = getUserId();
  window.localStorage.setItem(PARTICIPANT_ID_KEY, created);
  return created;
}

function setActiveAssessmentId(assessmentId) {
  window.localStorage.setItem(USER_ID_KEY, assessmentId);
}

function PriceChart({ points, asset }) {
  const width = 900;
  const height = 360;
  const padding = { top: 24, right: 24, bottom: 42, left: 58 };
  const prices = points.map((point) => point.normalized_price);
  const low = Math.min(...prices, 100);
  const high = Math.max(...prices, 100);
  const spread = Math.max(high - low, 2);
  const minY = low - spread * 0.18;
  const maxY = high + spread * 0.18;
  const x = (day) =>
    padding.left + ((day - 1) / 59) * (width - padding.left - padding.right);
  const y = (price) =>
    padding.top +
    ((maxY - price) / (maxY - minY)) *
      (height - padding.top - padding.bottom);
  const line = points
    .map((point) => `${x(point.day)},${y(point.normalized_price)}`)
    .join(" ");
  const last = points.at(-1);

  return (
    <div className="chart-wrap" aria-label={`${asset} normalized price chart`}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line
          className="reference-line"
          x1={padding.left}
          x2={width - padding.right}
          y1={y(100)}
          y2={y(100)}
        />
        <line
          className="axis-line"
          x1={padding.left}
          x2={padding.left}
          y1={padding.top}
          y2={height - padding.bottom}
        />
        <line
          className="axis-line"
          x1={padding.left}
          x2={width - padding.right}
          y1={height - padding.bottom}
          y2={height - padding.bottom}
        />
        {points.length > 1 && <polyline className="price-line" points={line} />}
        {last && (
          <circle
            className="latest-point"
            cx={x(last.day)}
            cy={y(last.normalized_price)}
            r="5"
          />
        )}
        <text className="axis-label" x={padding.left} y={height - 14}>
          Day 1
        </text>
        <text
          className="axis-label"
          textAnchor="end"
          x={width - padding.right}
          y={height - 14}
        >
          Day 60
        </text>
        <text className="axis-label" x="8" y={y(100) + 4}>
          100
        </text>
        {last && (
          <text
            className="latest-label"
            textAnchor="end"
            x={Math.min(width - padding.right, x(last.day) + 54)}
            y={Math.max(18, y(last.normalized_price) - 12)}
          >
            {last.normalized_price.toFixed(2)}
          </text>
        )}
      </svg>
    </div>
  );
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      window.dispatchEvent(new Event("flowbit:access-required"));
    }
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : body.detail?.message ?? "요청을 처리하지 못했습니다.";
    throw new Error(detail);
  }
  return body;
}

function AnalysisResult({ run, onRetry, onRestart, restarting, restartError }) {
  const [showRestartConfirmation, setShowRestartConfirmation] = useState(false);
  if (!run || run.status === "queued" || run.status === "processing") {
    return (
      <main className="center-message analysis-state">
        <section className="completed-card analysis-status-card" aria-live="polite">
          <div className="analysis-spinner" aria-hidden="true" />
          <p className="eyebrow">ANALYZING</p>
          <h2>투자 행동을 분석하고 있습니다.</h2>
          <p>{run?.message ?? "분석을 준비하고 있습니다."}</p>
        </section>
      </main>
    );
  }

  if (run.status === "failed") {
    return (
      <main className="center-message analysis-state">
        <section className="completed-card analysis-status-card">
          <p className="eyebrow">ANALYSIS PAUSED</p>
          <h2>분석을 완료하지 못했습니다.</h2>
          <p>{run.message}</p>
          <button className="submit-button" type="button" onClick={onRetry}>
            다시 시도
          </button>
        </section>
      </main>
    );
  }

  const result = run.result;
  const analysis = result.analysis;
  return (
    <main className="app-shell analysis-shell">
      <header className="page-header analysis-header">
        <div>
          <p className="eyebrow">FINAL ANALYSIS</p>
          <h1>투자 성향 분석 결과</h1>
          <p className="subtitle">설문 응답과 실제 시장 선택을 구분해 비교했습니다.</p>
        </div>
      </header>

      <section className="profile-comparison-grid">
        <article className="result-card">
          <span>설문 기반 성향</span>
          <strong>{result.stated_profile}</strong>
        </article>
        <article className="result-card emphasized">
          <span>행동 기반 성향</span>
          <strong>{result.revealed_profile ?? "분석 근거 부족"}</strong>
        </article>
      </section>

      <section className="analysis-card">
        <p className="eyebrow">STATED PREFERENCE</p>
        <h2>설문 응답 요약</h2>
        <p className="analysis-description">{analysis.stated_preference_summary}</p>
      </section>

      <section className="analysis-card">
        <p className="eyebrow">REVEALED PREFERENCE</p>
        <h2>행동 응답 요약</h2>
        <p className="analysis-description">{analysis.revealed_preference_summary}</p>
      </section>

      <section className="analysis-card">
        <p className="eyebrow">COMPARISON</p>
        <h2>설문과 행동의 차이</h2>
        <p className="analysis-description">{analysis.stated_revealed_gap}</p>
        <ul className="gap-list">
          {analysis.key_behavioral_evidence.map((evidence, index) => (
            <li key={`${index}-${evidence}`}>{evidence}</li>
          ))}
        </ul>
      </section>

      <section className="analysis-card final-analysis-card">
        <p className="eyebrow">FINAL INTERPRETATION</p>
        <h2>종합 해석</h2>
        <p className="analysis-confidence">해석 신뢰도 {analysis.confidence}</p>
        <p className="analysis-description">{analysis.final_analysis}</p>
      </section>

      <section className="analysis-card">
        <p className="eyebrow">AI GUIDANCE</p>
        <h2>AI 맞춤 제안</h2>
        <p className="analysis-description">{analysis.personalized_guidance}</p>
      </section>

      <section className="restart-measurement-section">
        {!showRestartConfirmation ? (
          <button
            className="secondary-button"
            type="button"
            onClick={() => setShowRestartConfirmation(true)}
          >
            다시 측정하기
          </button>
        ) : (
          <div className="restart-confirmation" role="alertdialog" aria-live="polite">
            <h2>새로운 측정을 시작할까요?</h2>
            <p>
              기존 설문, 행동 기록과 분석 결과는 삭제되지 않아요. 새로운 응답은
              별도의 측정 회차와 분석 결과로 저장돼요.
            </p>
            {restartError && <p className="inline-error">{restartError}</p>}
            <div className="restart-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={restarting}
                onClick={() => setShowRestartConfirmation(false)}
              >
                취소
              </button>
              <button
                className="submit-button"
                type="button"
                disabled={restarting}
                onClick={onRestart}
              >
                {restarting ? "새 측정을 준비하고 있어요…" : "확인하고 다시 시작"}
              </button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default function App() {
  const [accessState, setAccessState] = useState("checking");
  const [accessError, setAccessError] = useState("");
  const [session, setSession] = useState(null);
  const [questionnaire, setQuestionnaire] = useState(null);
  const [surveySaved, setSurveySaved] = useState(false);
  const [riskPercent, setRiskPercent] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [analysisRun, setAnalysisRun] = useState(null);
  const analysisRequestToken = useRef(0);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/access/session`, { credentials: "include" })
      .then((response) => {
        if (!active) return;
        setAccessState(response.ok ? "granted" : "required");
      })
      .catch(() => {
        if (!active) return;
        setAccessError("접근 상태를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.");
        setAccessState("required");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function requireAccess() {
      analysisRequestToken.current += 1;
      setAccessState("required");
      setSession(null);
      setQuestionnaire(null);
      setSurveySaved(false);
      setAnalysisRun(null);
      setLoading(false);
    }
    window.addEventListener("flowbit:access-required", requireAccess);
    return () => window.removeEventListener("flowbit:access-required", requireAccess);
  }, []);

  async function startEpisode(number) {
    return api(`/api/episode${number}/sessions`, {
      method: "POST",
      body: JSON.stringify({ user_id: getUserId() }),
    });
  }

  async function advanceToActiveEpisode(data) {
    let current = data;
    while (
      current.episode_status === "completed" &&
      Number(current.episode.slice(1)) < 6
    ) {
      current = await startEpisode(Number(current.episode.slice(1)) + 1);
    }
    return current;
  }

  async function beginBehavioralEpisodes() {
    setLoading(true);
    setError("");
    try {
      const data = await startEpisode(1);
      const current = await advanceToActiveEpisode(data);
      setSession(current);
      setSurveySaved(false);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitSurvey(answers) {
    setSubmitting(true);
    setError("");
    try {
      await api("/api/survey/submissions", {
        method: "POST",
        body: JSON.stringify({ user_id: getUserId(), answers }),
      });
      setQuestionnaire(null);
      setSurveySaved(true);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setSubmitting(false);
    }
  }

  useEffect(() => {
    if (accessState !== "granted") return undefined;
    let active = true;
    setLoading(true);
    api("/api/survey/sessions", {
      method: "POST",
      body: JSON.stringify({ user_id: getUserId() }),
    })
      .then(async (data) => {
        if (!active) return;
        if (!data.survey_completed) {
          setQuestionnaire(data.questionnaire);
          return;
        }
        const episode = await startEpisode(1);
        const current = await advanceToActiveEpisode(episode);
        if (active) setSession(current);
      })
      .catch((reason) => {
        if (active) setError(reason.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [accessState]);

  useEffect(() => {
    if (!session) return;
    setRiskPercent(Math.round(session.current_risk_share * 100));
  }, [session?.next_decision?.decision_point, session?.interaction_phase]);

  const latestPoint = session?.price_series?.at(-1);
  const cashPercent = 100 - riskPercent;
  const progress = useMemo(() => {
    if (!session) return 0;
    return (session.progress.submitted / session.progress.total) * 100;
  }, [session]);

  async function submitDecision() {
    if (!session?.next_decision || submitting) return;
    setSubmitting(true);
    setError("");
    const point = session.next_decision;
    try {
      const episodeNumber = session.episode.slice(1);
      let updated;
      if (session.episode === "E5") {
        const isPre = session.interaction_phase === "pre_information";
        updated = await api(
          `/api/episode5/sessions/${session.session_id}/${isPre ? "pre-decisions" : "post-decisions"}`,
          {
            method: "POST",
            body: JSON.stringify({
              scenario_id: session.scenario_id,
              decision_point: point.decision_point,
              day: point.day,
              [isPre ? "risk_share_pre_info" : "risk_share_post_info"]:
                riskPercent / 100,
            }),
          },
        );
      } else {
        updated = await api(
          `/api/episode${episodeNumber}/sessions/${session.session_id}/decisions`,
          {
            method: "POST",
            body: JSON.stringify({
              scenario_id: session.scenario_id,
              decision_point: point.decision_point,
              day: point.day,
              risk_share_after: riskPercent / 100,
            }),
          },
        );
      }
      if (
        updated.episode_status === "completed" &&
        Number(updated.episode.slice(1)) < 6
      ) {
        updated = await startEpisode(Number(updated.episode.slice(1)) + 1);
      }
      setSession(updated);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function confirmEpisode3Entry() {
    if (!session?.entry_setup_required || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const updated = await api(
        `/api/episode3/sessions/${session.session_id}/entry`,
        {
          method: "POST",
          body: JSON.stringify({ risk_share: riskPercent / 100 }),
        },
      );
      setSession(updated);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function startAnalysis() {
    const token = analysisRequestToken.current + 1;
    analysisRequestToken.current = token;
    setError("");
    setAnalysisRun({ status: "queued", message: "분석을 준비하고 있습니다." });
    const deadline = Date.now() + ANALYSIS_POLL_TIMEOUT_MS;
    try {
      let run = await api("/api/analysis/runs", {
        method: "POST",
        body: JSON.stringify({ user_id: getUserId() }),
      });
      if (analysisRequestToken.current !== token) return;
      setAnalysisRun(run);
      while (run.status === "queued" || run.status === "processing") {
        if (Date.now() >= deadline) {
          throw new Error("분석 대기 시간이 초과되었습니다. 상태를 다시 확인해 주세요.");
        }
        await new Promise((resolve) => setTimeout(resolve, ANALYSIS_POLL_INTERVAL_MS));
        if (analysisRequestToken.current !== token) return;
        run = await api(
          `/api/analysis/runs/${run.analysis_id}?user_id=${encodeURIComponent(getUserId())}`,
        );
        if (analysisRequestToken.current !== token) return;
        setAnalysisRun(run);
      }
    } catch (reason) {
      if (analysisRequestToken.current !== token) return;
      setAnalysisRun({
        status: "failed",
        message: reason.message || "분석을 완료하지 못했습니다.",
      });
    }
  }

  async function restartMeasurement() {
    if (submitting || loading) return;
    setSubmitting(true);
    setError("");
    const previousAssessmentId = getUserId();
    try {
      const attempt = await api("/api/assessment-attempts", {
        method: "POST",
        body: JSON.stringify({
          participant_id: getParticipantId(),
          previous_assessment_id: previousAssessmentId,
        }),
      });
      setActiveAssessmentId(attempt.assessment_id);
      analysisRequestToken.current += 1;
      setAnalysisRun(null);
      setSession(null);
      setQuestionnaire(null);
      setSurveySaved(false);
      setRiskPercent(0);
      setLoading(true);

      const survey = await api("/api/survey/sessions", {
        method: "POST",
        body: JSON.stringify({ user_id: attempt.assessment_id }),
      });
      setQuestionnaire(survey.questionnaire);
    } catch (reason) {
      setError(reason.message || "새 측정을 시작하지 못했습니다.");
    } finally {
      setLoading(false);
      setSubmitting(false);
    }
  }

  useEffect(() => {
    if (
      session?.episode === "E6" &&
      session.episode_status === "completed" &&
      analysisRun === null
    ) {
      startAnalysis();
    }
  }, [session?.episode, session?.episode_status, analysisRun]);

  useEffect(
    () => () => {
      analysisRequestToken.current += 1;
    },
    [],
  );

  if (accessState === "checking") {
    return <main className="center-message">접근 상태를 확인하고 있습니다…</main>;
  }
  if (accessState === "required") {
    return (
      <AccessGate
        apiBase={API_BASE}
        initialError={accessError}
        onGranted={() => {
          setAccessError("");
          setError("");
          setAccessState("granted");
        }}
      />
    );
  }
  if (loading) {
    return <main className="center-message">시나리오를 준비하고 있습니다…</main>;
  }
  if (questionnaire) {
    return (
      <Survey
        questionnaire={questionnaire}
        disabled={submitting}
        error={error}
        onSubmit={submitSurvey}
      />
    );
  }
  if (surveySaved) {
    return (
      <main className="center-message survey-confirmation">
        <section className="completed-card">
          <p className="eyebrow">SAVED</p>
          <h2>설문 응답이 저장되었습니다.</h2>
          <p>다음 단계에서 시장 상황에 따른 선택을 진행합니다.</p>
          {error && <p className="inline-error">{error}</p>}
          <button
            className="submit-button"
            type="button"
            disabled={loading}
            onClick={beginBehavioralEpisodes}
          >
            다음 단계 시작
          </button>
        </section>
      </main>
    );
  }
  if (!session) {
    return <main className="center-message error">{error}</main>;
  }
  if (session.episode === "E6" && session.episode_status === "completed") {
    return (
      <AnalysisResult
        run={analysisRun}
        onRetry={startAnalysis}
        onRestart={restartMeasurement}
        restarting={submitting}
        restartError={error}
      />
    );
  }

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">EPISODE {session.episode.slice(1)}</p>
          <h1>투자 비중 결정</h1>
          <p className="subtitle">
            현재까지 공개된 가격을 보고 위험자산 비중을 선택해 주세요.
          </p>
        </div>
        <div className="progress-box">
          <span>
            {session.progress.submitted} / {session.progress.total} 결정 완료
          </span>
          <div className="progress-track">
            <div className="progress-value" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </header>

      {!session.entry_setup_required && <section className="market-card">
        <div className="market-card-header">
          <div>
            <span className="asset-label">{session.asset}</span>
            <span className="normalized-label">Normalized · Day 1 = 100</span>
          </div>
          <div className="current-price">
            <strong>{latestPoint?.normalized_price.toFixed(2)}</strong>
            <span>{latestPoint?.label}</span>
          </div>
        </div>
        <PriceChart points={session.price_series} asset={session.asset} />
      </section>}

      {session.entry_setup_required ? (
        <section className="allocation-card">
          <div className="decision-heading">
            <div>
              <p className="eyebrow">ENTRY SETUP</p>
              <h2>시작 위험자산 비중</h2>
            </div>
            <span className="step-badge">5% 단위</span>
          </div>
          <p className="subtitle">
            다음 시장 구간이 시작되기 전 초기 투자 비중을 확정해 주세요.
          </p>
          <input
            className="allocation-slider"
            type="range"
            min="0"
            max="100"
            step="5"
            value={riskPercent}
            onChange={(event) => setRiskPercent(Number(event.target.value))}
            aria-label="Episode 3 시작 위험자산 비중"
          />
          <div className="allocation-grid">
            <div className="allocation-value risk">
              <span>위험자산</span>
              <strong>{riskPercent}%</strong>
            </div>
            <div className="allocation-value cash">
              <span>현금</span>
              <strong>{cashPercent}%</strong>
            </div>
          </div>
          {error && <p className="inline-error">{error}</p>}
          <button
            className="submit-button"
            type="button"
            disabled={submitting}
            onClick={confirmEpisode3Entry}
          >
            {submitting ? "저장 중…" : "이 비중으로 시작"}
          </button>
        </section>
      ) : session.episode_status === "completed" ? (
        <section className="completed-card">
          <p className="eyebrow">COMPLETED</p>
          <h2>Episode {session.episode.slice(1)} 의사결정이 완료되었습니다.</h2>
          <p>응답이 안전하게 기록되었습니다.</p>
        </section>
      ) : (
        <section className="allocation-card">
          <div className="decision-heading">
            <div>
              <p className="eyebrow">
                DECISION {session.next_decision.sequence} OF {session.progress.total}
              </p>
              <h2>
                {`Day ${session.next_decision.day}`} {session.episode === "E5"
                  ? session.interaction_phase === "post_information"
                    ? "정보 확인 후 투자 비중"
                    : "정보 확인 전 투자 비중"
                  : "투자 비중"}
              </h2>
            </div>
            <span className="step-badge">5% 단위</span>
          </div>

          {session.episode === "E5" &&
            session.interaction_phase === "post_information" && (
              <div className="stimulus-grid" aria-label="외부 정보 카드">
                {session.stimulus_cards.map((card) => (
                  <article className="stimulus-card" key={card.position}>
                    <span>{card.source_label}</span>
                    <h3>{card.title}</h3>
                    <p>{card.content}</p>
                  </article>
                ))}
              </div>
            )}

          <input
            className="allocation-slider"
            type="range"
            min={Math.round(
              (session.allocation_constraints?.minimum_next_risk_share ?? 0) * 100,
            )}
            max="100"
            step="5"
            value={riskPercent}
            onChange={(event) => setRiskPercent(Number(event.target.value))}
            aria-label="위험자산 비중"
          />

          <div className="allocation-grid">
            <div className="allocation-value risk">
              <span>위험자산</span>
              <strong>{riskPercent}%</strong>
            </div>
            <div className="allocation-value cash">
              <span>현금</span>
              <strong>{cashPercent}%</strong>
            </div>
          </div>

          {session.allocation_constraints?.allocation_floor > 0 && (
            <p className="constraint-note">
              이번 시나리오의 위험자산 최소 비중은 {Math.round(
                session.allocation_constraints.allocation_floor * 100,
              )}%입니다.
            </p>
          )}

          {error && <p className="inline-error">{error}</p>}
          <button
            className="submit-button"
            type="button"
            disabled={submitting}
            onClick={submitDecision}
          >
            {submitting
              ? "저장 중…"
              : session.episode === "E5"
                ? session.interaction_phase === "post_information"
                  ? "정보 확인 후 비중 확정"
                  : "현재 비중 확정 후 정보 확인"
                : "이 비중으로 결정"}
          </button>
        </section>
      )}
    </main>
  );
}
