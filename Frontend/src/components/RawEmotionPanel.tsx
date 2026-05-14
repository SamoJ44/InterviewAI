import { Activity } from "lucide-react";
import type { EmotionProbabilities, RawEmotionResponse } from "../types/backend";

type RawEmotionPanelProps = {
  rawEmotion?: RawEmotionResponse | null;
  frozen?: boolean;
};

const EMOTION_CLASSES: Array<keyof EmotionProbabilities> = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
];

function labelText(label?: string | null) {
  if (!label) return "Unavailable";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatPercent(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "Unavailable";
}

function probabilityValue(probabilities: EmotionProbabilities | undefined, label: keyof EmotionProbabilities) {
  const value = probabilities?.[label];
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

export function RawEmotionPanel({ rawEmotion, frozen = false }: RawEmotionPanelProps) {
  const hasPrediction = rawEmotion?.available === true;
  const isLowConfidence = rawEmotion?.status === "low_confidence";
  const statusText = rawEmotion?.status ?? "unavailable";

  return (
    <article className="raw-emotion-card glass-card">
      <div className="raw-emotion-header">
        <span className="score-icon">
          <Activity size={18} />
        </span>
        <div>
          <p>Latest frame</p>
          <h3>Fresh Emotion Prediction</h3>
          <span>Raw model output from the latest frame.</span>
        </div>
        <span className={`raw-emotion-badge ${isLowConfidence ? "low-confidence" : hasPrediction ? "ok" : "unavailable"}`}>
          {isLowConfidence ? "Low confidence" : hasPrediction ? "Raw" : statusText}
        </span>
      </div>

      {hasPrediction ? (
        <>
          <div className="raw-emotion-summary">
            <div>
              <span>Predicted</span>
              <strong>{labelText(rawEmotion?.label)}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{formatPercent(rawEmotion?.confidence)}</strong>
            </div>
          </div>

          <div className="raw-emotion-bars">
            {EMOTION_CLASSES.map((label) => {
              const probability = probabilityValue(rawEmotion?.probabilities, label);
              return (
                <div className="raw-emotion-row" key={label}>
                  <div>
                    <span>{label}</span>
                    <strong>{formatPercent(probability)}</strong>
                  </div>
                  <div className="mini-progress" aria-hidden="true">
                    <span style={{ width: `${Math.round(probability * 100)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <div className="raw-emotion-unavailable">
          <strong>{statusText}</strong>
          <span>Raw emotion prediction is unavailable for the latest frame.</span>
        </div>
      )}

      {frozen && <em className="expression-frozen">Latest frame</em>}
    </article>
  );
}
