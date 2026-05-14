import { Activity, CheckCircle2, Timer, TriangleAlert } from "lucide-react";
import type { Calibration } from "../types/backend";

type CalibrationPanelProps = {
  calibration?: Calibration | null;
};

function calibrationTitle(status?: string) {
  if (status === "preparing") return "Get ready";
  if (status === "calibrating") return "Hold still";
  if (status === "ready") return "Calibration ready";
  if (status === "failed") return "Calibration failed";
  return "Calibration";
}

function calibrationIcon(status?: string) {
  if (status === "ready") return <CheckCircle2 size={18} />;
  if (status === "failed") return <TriangleAlert size={18} />;
  if (status === "preparing") return <Timer size={18} />;
  return <Activity size={18} />;
}

export function CalibrationPanel({ calibration }: CalibrationPanelProps) {
  const progress = typeof calibration?.progress === "number" ? calibration.progress : 0;
  const normalized = progress <= 1 ? progress * 100 : progress;

  return (
    <section className="glass-card panel-block">
      <div className="panel-heading">
        <span className="panel-icon">{calibrationIcon(calibration?.status)}</span>
        <div>
          <p>Calibration</p>
          <h2>{calibrationTitle(calibration?.status)}</h2>
        </div>
      </div>
      <p className="panel-copy">{calibration?.message ?? "Calibration starts after the first frames are processed."}</p>
      <div className="progress-row">
        <div className="wide-progress">
          <span style={{ width: `${Math.max(0, Math.min(100, normalized))}%` }} />
        </div>
        <strong>{Math.round(normalized)}%</strong>
      </div>
      <div className="metric-row">
        <span>Samples</span>
        <strong>{calibration?.samples ?? 0}</strong>
      </div>
      {typeof calibration?.countdown_remaining === "number" && (
        <div className="metric-row">
          <span>Countdown</span>
          <strong>{Math.max(0, Math.round(calibration.countdown_remaining))}s</strong>
        </div>
      )}
    </section>
  );
}
