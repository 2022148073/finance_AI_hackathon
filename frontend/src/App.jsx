import { useEffect, useMemo, useState } from "react";

const API_BASE = window.__API_BASE_URL__ ?? "http://127.0.0.1:8000";

function getUserId() {
  const key = "experiment_user_id";
  const existing =
    window.localStorage.getItem(key) ??
    window.localStorage.getItem("episode1_user_id");
  if (existing) window.localStorage.setItem(key, existing);
  if (existing) return existing;
  const created = `web_${crypto.randomUUID().replaceAll("-", "")}`;
  window.localStorage.setItem(key, created);
  return created;
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
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : body.detail?.message ?? "요청을 처리하지 못했습니다.";
    throw new Error(detail);
  }
  return body;
}

export default function App() {
  const [session, setSession] = useState(null);
  const [riskPercent, setRiskPercent] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

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

  useEffect(() => {
    let active = true;
    startEpisode(1)
      .then(async (data) => {
        const current = await advanceToActiveEpisode(data);
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
  }, []);

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

  if (loading) {
    return <main className="center-message">시나리오를 준비하고 있습니다…</main>;
  }
  if (!session) {
    return <main className="center-message error">{error}</main>;
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
