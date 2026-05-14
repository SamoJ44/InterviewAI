import { Bell, CircleAlert, Sparkles } from "lucide-react";
import type { TimelineItem } from "../types/backend";

type EventsPanelProps = {
  events?: TimelineItem[];
  alerts?: TimelineItem[];
};

function itemSeverity(item: TimelineItem) {
  if (item.severity === "critical") return "danger";
  if (item.severity === "warning") return "warning";
  return "info";
}

export function EventsPanel({ events = [], alerts = [] }: EventsPanelProps) {
  const items = [...alerts, ...events].slice(-8).reverse();

  return (
    <section className="glass-card panel-block events-panel">
      <div className="panel-heading">
        <span className="panel-icon">
          <Bell size={18} />
        </span>
        <div>
          <p>Events</p>
          <h2>Alerts timeline</h2>
        </div>
      </div>
      <div className="timeline-list">
        {items.length === 0 ? (
          <div className="empty-state">
            <Sparkles size={18} />
            No important events yet.
          </div>
        ) : (
          items.map((item, index) => (
            <article className={`timeline-item ${itemSeverity(item)}`} key={item.id ?? `${item.title}-${index}`}>
              <CircleAlert size={16} />
              <div>
                <div className="timeline-meta">
                  <span>{item.time ?? item.timestamp ?? "now"}</span>
                  <strong>{item.severity ?? "info"}</strong>
                </div>
                <h3>{item.title ?? item.category ?? "Interview event"}</h3>
                <p>{item.description ?? "Signal change detected."}</p>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
