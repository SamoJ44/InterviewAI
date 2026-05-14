import { Eye, Hand, ScanFace, UserRoundCheck } from "lucide-react";
import type { Flags } from "../types/backend";
import { StatusBadge } from "./StatusBadge";

type DetectionFlagsProps = {
  flags?: Flags | null;
};

function eyeKind(status?: string) {
  if (status === "OPEN") return "success";
  if (status === "PARTIAL") return "warning";
  if (status === "CLOSED" || status === "NO_FACE") return "danger";
  return "info";
}

export function DetectionFlags({ flags }: DetectionFlagsProps) {
  return (
    <section className="glass-card panel-block">
      <div className="panel-heading">
        <span className="panel-icon">
          <ScanFace size={18} />
        </span>
        <div>
          <p>Detection</p>
          <h2>Live signals</h2>
        </div>
      </div>
      <div className="flag-grid">
        <StatusBadge label="Face" value={flags?.face_detected} kind={flags?.face_detected ? "success" : "danger"} />
        <StatusBadge label="Pose" value={flags?.pose_detected} kind={flags?.pose_detected ? "success" : "danger"} />
        <StatusBadge label="Hands" value={flags?.hands_detected} kind={flags?.hands_detected ? "success" : "neutral"} />
        <StatusBadge
          label="Self-touch"
          value={flags?.self_touch_active}
          kind={flags?.self_touch_active ? "danger" : "success"}
        />
      </div>
      <div className="eye-status-row">
        <Eye size={18} />
        <span>Eye status</span>
        <StatusBadge value={flags?.eye_status ?? "Unknown"} kind={eyeKind(flags?.eye_status)} />
      </div>
      <div className="signal-icons" aria-hidden="true">
        <UserRoundCheck />
        <Hand />
        <Eye />
      </div>
    </section>
  );
}
