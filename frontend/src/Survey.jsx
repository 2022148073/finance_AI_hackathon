import { useMemo, useState } from "react";

export default function Survey({ questionnaire, disabled, error, onSubmit }) {
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const questions = questionnaire.questions;
  const question = questions[index];
  const answer = answers[question.id];
  const multiple = question.type === "multiple";
  const answered = multiple
    ? Array.isArray(answer) && answer.length > 0
    : typeof answer === "string";
  const progress = useMemo(
    () => ((index + 1) / questions.length) * 100,
    [index, questions.length],
  );

  function choose(optionId) {
    if (!multiple) {
      setAnswers((current) => ({ ...current, [question.id]: optionId }));
      return;
    }
    const selected = Array.isArray(answer) ? answer : [];
    const next = selected.includes(optionId)
      ? selected.filter((item) => item !== optionId)
      : [...selected, optionId];
    setAnswers((current) => ({ ...current, [question.id]: next }));
  }

  async function next() {
    if (!answered || disabled) return;
    if (index < questions.length - 1) {
      setIndex((current) => current + 1);
      return;
    }
    await onSubmit(answers);
  }

  return (
    <main className="app-shell survey-shell">
      <header className="page-header survey-header">
        <div>
          <p className="eyebrow">STEP 1</p>
          <h1>{questionnaire.title}</h1>
          <p className="subtitle">
            현재 상황과 가장 가까운 항목을 선택해 주세요.
          </p>
        </div>
        <div className="progress-box">
          <span>
            진행률 {index + 1} / {questions.length}
          </span>
          <div className="progress-track">
            <div className="progress-value" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </header>

      <section className="survey-card">
        <p className="survey-question-number">Q{question.number}</p>
        <h2>{question.prompt}</h2>
        {question.help_text && (
          <p className="survey-help">{question.help_text}</p>
        )}

        <fieldset className="survey-options">
          <legend className="visually-hidden">{question.prompt}</legend>
          {question.options.map((option) => {
            const selected = multiple
              ? (answer ?? []).includes(option.id)
              : answer === option.id;
            return (
              <label
                className={`survey-option${selected ? " selected" : ""}`}
                key={option.id}
              >
                <input
                  type={multiple ? "checkbox" : "radio"}
                  name={question.id}
                  value={option.id}
                  checked={selected}
                  onChange={() => choose(option.id)}
                />
                <span>
                  <strong>{option.label}</strong>
                  {option.description && <small>{option.description}</small>}
                </span>
              </label>
            );
          })}
        </fieldset>

        {error && <p className="inline-error">{error}</p>}
        <div className="survey-actions">
          <button
            className="secondary-button"
            type="button"
            disabled={index === 0 || disabled}
            onClick={() => setIndex((current) => current - 1)}
          >
            이전
          </button>
          <button
            className="submit-button survey-next"
            type="button"
            disabled={!answered || disabled}
            onClick={next}
          >
            {disabled
              ? "저장 중..."
              : index === questions.length - 1
                ? "응답 저장"
                : "다음"}
          </button>
        </div>
      </section>
    </main>
  );
}
