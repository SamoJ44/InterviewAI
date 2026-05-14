import type { SessionStatus } from "../types/backend";

type StatusKind = SessionStatus | "success" | "warning" | "danger" | "info" | "neutral";

type StatusBadgeProps = {
  label?: string;
  value?: string | boolean | number | null;
  kind?: StatusKind;
};

function normalizeBoolean(value: string | boolean | number | null | undefined) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (value === null || value === undefined || value === "") return "Unknown";
  return String(value);
}

export function kindForStatus(status: SessionStatus): StatusKind {
  if (status === "active") return "success";
  if (status === "preparing" || status === "calibrating" || status === "paused") return "warning";
  if (status === "ended") return "info";
  return "neutral";
}

export function StatusBadge({ label, value, kind = "neutral" }: StatusBadgeProps) {
  const resolvedKind = ["idle", "preparing", "calibrating", "active", "paused", "ended"].includes(kind)
    ? kindForStatus(kind as SessionStatus)
    : kind;

  return (
    <span className={`status-badge ${resolvedKind}`}>
      {label && <span>{label}</span>}
      <strong>{normalizeBoolean(value)}</strong>
    </span>
  );
}
