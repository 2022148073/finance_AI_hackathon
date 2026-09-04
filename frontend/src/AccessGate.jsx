import { useState } from "react";


export default function AccessGate({ apiBase, initialError = "", onGranted }) {
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState(initialError);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!accessCode || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/api/access/verify`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ access_code: accessCode }),
      });
      if (!response.ok) {
        const message =
          response.status === 429
            ? "잠시 후 다시 시도해 주세요."
            : "접근 코드를 확인해 주세요.";
        throw new Error(message);
      }
      setAccessCode("");
      onGranted();
    } catch (reason) {
      setError(reason.message || "접근 코드를 확인해 주세요.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="center-message access-gate-shell">
      <section className="completed-card access-gate-card">
        <p className="eyebrow">INVITATION ONLY</p>
        <div className="access-brand-lockup">
          <img className="access-brand-logo" src="/flowbit-logo.png" alt="" />
          <h1>FlowBit</h1>
        </div>
        <p className="access-gate-description">
          대회 심사용 데모 서비스입니다.
          <br />
          <span className="access-instruction-line">
            안내받은 접근 코드를 입력해 주세요.
          </span>
        </p>
        <form onSubmit={submit}>
          <label className="access-code-label" htmlFor="flowbit-access-code">
            Access Code
          </label>
          <input
            id="flowbit-access-code"
            className="access-code-input"
            type="password"
            autoComplete="one-time-code"
            value={accessCode}
            disabled={submitting}
            onChange={(event) => setAccessCode(event.target.value)}
          />
          {error && <p className="inline-error" role="alert">{error}</p>}
          <button
            className="submit-button access-submit-button"
            type="submit"
            disabled={!accessCode || submitting}
          >
            {submitting ? "확인하고 있어요…" : "입장하기"}
          </button>
        </form>
      </section>
    </main>
  );
}
