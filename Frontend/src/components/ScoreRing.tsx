type ScoreRingProps = {
  label: string;
  value?: number | null;
};

function scoreTone(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "neutral";
  if (value >= 80) return "excellent";
  if (value >= 60) return "good";
  if (value >= 40) return "warning";
  return "poor";
}

export function ScoreRing({ label, value }: ScoreRingProps) {
  const hasValue = typeof value === "number" && !Number.isNaN(value);
  const score = Math.max(0, Math.min(100, hasValue ? value : 0));
  const circumference = 2 * Math.PI * 54;
  const dashOffset = circumference - (score / 100) * circumference;

  return (
    <article className={`score-ring-card glass-card ${scoreTone(value)}`}>
      <div className="score-ring-wrap">
        <svg className="score-ring" viewBox="0 0 128 128" role="img" aria-label={label}>
          <circle className="ring-track" cx="64" cy="64" r="54" />
          <circle
            className="ring-value"
            cx="64"
            cy="64"
            r="54"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
          />
        </svg>
        <div className="score-ring-number">
          <strong>{hasValue ? Math.round(score) : "--"}</strong>
          <span>/100</span>
        </div>
      </div>
      <div>
        <p>{label}</p>
        <h2>{hasValue ? scoreTone(value).replace(/^\w/, (letter) => letter.toUpperCase()) : "Awaiting frames"}</h2>
      </div>
    </article>
  );
}
