import { Meh, SmilePlus } from "lucide-react";
import type { EmotionResponse } from "../types/backend";

type ExpressionCardProps = {
  emotion?: EmotionResponse | null;
  frozen?: boolean;
};

const INTERVIEW_READY_LABELS = new Set(["neutral", "happy"]);
const TENSE_LABELS = new Set(["sad", "angry", "fear", "disgust", "surprise"]);

function formatPercent(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "Unavailable";
}

function formatScore(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? Math.round(value) : null;
}

function labelText(label?: string | null) {
  if (!label) return "Unavailable";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function expressionState(emotion?: EmotionResponse | null) {
  if (!emotion) {
    return {
      tone: "unavailable",
      badge: "Unavailable",
      message: "Expression analysis unavailable.",
    };
  }

  if (emotion.status === "model_not_loaded") {
    return {
      tone: "unavailable",
      badge: "Unavailable",
      message: "Expression model not loaded.",
    };
  }

  if (emotion.status === "service_unavailable") {
    return {
      tone: "unavailable",
      badge: "Unavailable",
      message: "Expression service unavailable.",
    };
  }

  if (emotion.status === "face_not_detected") {
    return {
      tone: "unavailable",
      badge: "No face",
      message: "Face not visible for expression analysis.",
    };
  }

  if (emotion.status === "low_confidence") {
    return {
      tone: "low-confidence",
      badge: "Low confidence",
      message: "Prediction confidence is low.",
    };
  }

  if (emotion.status === "prediction_error") {
    return {
      tone: "unavailable",
      badge: "Unavailable",
      message: "Expression prediction failed.",
    };
  }

  const normalizedLabel = emotion.label?.toLowerCase() ?? "";
  if (emotion.available && INTERVIEW_READY_LABELS.has(normalizedLabel)) {
    return {
      tone: "ready",
      badge: "Interview-ready",
      message: "Expression looks interview-ready.",
    };
  }

  if (emotion.available && TENSE_LABELS.has(normalizedLabel)) {
    return {
      tone: "tense",
      badge: "Tense",
      message: "Expression looks tense. Try to relax your face.",
    };
  }

  return {
    tone: "uncertain",
    badge: "Uncertain",
    message: "Expression analysis unavailable.",
  };
}

export function ExpressionCard({ emotion, frozen = false }: ExpressionCardProps) {
  const state = expressionState(emotion);
  const score = formatScore(emotion?.score);
  const scoreWidth = Math.max(0, Math.min(100, score ?? 0));
  const Icon = state.tone === "ready" ? SmilePlus : Meh;

  return (
    <article className={`expression-card glass-card ${state.tone}`}>
      <div className="expression-card-header">
        <span className="score-icon">
          <Icon size={18} />
        </span>
        <div>
          <p>Expression Score</p>
          <h3>Facial Expression</h3>
        </div>
        <span className="expression-badge">{state.badge}</span>
      </div>

      <div className="expression-score-row">
        <strong>
          {score ?? (emotion?.status === "face_not_detected" ? "No face" : "Unavailable")}
          {score !== null && <small>/100</small>}
        </strong>
        <span>{state.message}</span>
      </div>

      {score !== null && (
        <div className="expression-progress mini-progress" aria-hidden="true">
          <span style={{ width: `${scoreWidth}%` }} />
        </div>
      )}

      <div className="expression-meta">
        <div>
          <span>Current</span>
          <strong>{labelText(emotion?.label)}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>{formatPercent(emotion?.confidence)}</strong>
        </div>
      </div>

      {frozen && <em className="expression-frozen">Last expression result</em>}
    </article>
  );
}
