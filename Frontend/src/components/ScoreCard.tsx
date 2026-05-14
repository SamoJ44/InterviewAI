import type { LucideIcon } from "lucide-react";

type ScoreCardProps = {
  label: string;
  value?: number | null;
  icon: LucideIcon;
  frozen?: boolean;
};

function statusForScore(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "No data yet";
  if (value >= 80) return "Excellent";
  if (value >= 60) return "Good";
  if (value >= 40) return "Needs work";
  return "Poor";
}

function toneForScore(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "neutral";
  if (value >= 80) return "excellent";
  if (value >= 60) return "good";
  if (value >= 40) return "warning";
  return "poor";
}

export function ScoreCard({ label, value, icon: Icon, frozen = false }: ScoreCardProps) {
  const hasValue = typeof value === "number" && !Number.isNaN(value);
  const rounded = hasValue ? Math.round(value) : null;

  return (
    <article className={`score-card ${toneForScore(value)}`}>
      <div className="score-card-top">
        <span className="score-icon">
          <Icon size={18} />
        </span>
        <span>{label}</span>
      </div>
      <strong>{rounded ?? "No data yet"}</strong>
      <div className="score-card-footer">
        <span>{statusForScore(value)}</span>
        {frozen && <em>Last value</em>}
      </div>
      <div className="mini-progress" aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(100, rounded ?? 0))}%` }} />
      </div>
    </article>
  );
}
