import { CirclePause, CirclePlay, Octagon, RotateCw } from "lucide-react";
import type { SessionStatus } from "../types/backend";

type SessionControlsProps = {
  status?: SessionStatus;
  busy: boolean;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onEnd: () => void;
};

export function SessionControls({
  status,
  busy,
  onStart,
  onPause,
  onResume,
  onEnd,
}: SessionControlsProps) {
  const effectiveStatus = status ?? "idle";
  const canStart = effectiveStatus === "idle" || effectiveStatus === "ended";
  const canPause = effectiveStatus === "preparing" || effectiveStatus === "calibrating" || effectiveStatus === "active";
  const canResume = effectiveStatus === "paused";
  const canEnd = canPause || canResume;

  return (
    <div className="session-control-panel">
      <div className="session-control-status">Session status: {effectiveStatus}</div>
      <div className="session-controls">
        {canStart && (
          <button className="action-button primary" disabled={busy} onClick={onStart} type="button">
            <CirclePlay size={18} />
            {effectiveStatus === "ended" ? "Start New Session" : "Start Session"}
          </button>
        )}
        {canPause && (
          <button className="action-button warning" disabled={busy} onClick={onPause} type="button">
            <CirclePause size={18} />
            Pause
          </button>
        )}
        {canResume && (
          <button className="action-button success" disabled={busy} onClick={onResume} type="button">
            <RotateCw size={18} />
            Resume
          </button>
        )}
        {canEnd && (
          <button className="action-button danger" disabled={busy} onClick={onEnd} type="button">
            <Octagon size={18} />
            End Session
          </button>
        )}
      </div>
    </div>
  );
}
