import { ClipboardCheck, Clock, TrendingUp } from "lucide-react";
import type { Session, SessionRecommendations, SessionSummary, TimelineItem } from "../types/backend";

type FinalSummaryProps = {
  session: Session | null;
  summary?: SessionSummary | null;
  recommendations?: SessionRecommendations | null;
  events?: TimelineItem[];
  alerts?: TimelineItem[];
};

function formatScore(value?: number | null) {
  return typeof value === "number" && !Number.isNaN(value) ? Math.round(value) : "Not enough data";
}

function formatPercent(value?: number | null) {
  return typeof value === "number" && !Number.isNaN(value) ? `${Math.round(value * 100)}%` : "Unavailable";
}

function formatLabel(label?: string | null) {
  if (!label) return "Unavailable";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatDistribution(summary?: SessionSummary | null) {
  const distribution = summary?.emotion_summary.distribution;
  if (!distribution) return "Unavailable";

  const ranked = Object.entries(distribution)
    .filter(([, count]) => count > 0)
    .sort(([, leftCount], [, rightCount]) => rightCount - leftCount)
    .slice(0, 3);

  if (ranked.length === 0) return "Unavailable";
  return ranked.map(([label, count]) => `${formatLabel(label)} ${count}`).join(" / ");
}

function formatRecommendationCategory(category: string) {
  return category
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function RecommendationsSection({ recommendations }: { recommendations?: SessionRecommendations | null }) {
  return (
    <section className="recommendations-section">
      <div className="panel-heading compact">
        <div>
          <p>Personalized Recommendations</p>
          <h2>Coaching focus</h2>
        </div>
      </div>

      {!recommendations ? (
        <p className="panel-copy">Personalized recommendations are currently unavailable.</p>
      ) : (
        <div className="recommendations-grid">
          <div className="recommendation-block wide">
            <span>Overall Assessment</span>
            <p>{recommendations.overall_assessment}</p>
          </div>

          <div className="recommendation-block">
            <span>Strengths</span>
            {recommendations.strengths.length > 0 ? (
              <ul>
                {recommendations.strengths.map((strength) => (
                  <li key={strength}>{strength}</li>
                ))}
              </ul>
            ) : (
              <p>No specific strengths were returned.</p>
            )}
          </div>

          <div className="recommendation-block">
            <span>Areas to Improve</span>
            {recommendations.areas_to_improve.length > 0 ? (
              <ul>
                {recommendations.areas_to_improve.map((area) => (
                  <li key={`${area.category}-${area.message}`}>
                    <strong>{formatRecommendationCategory(area.category)} - {area.priority}</strong>
                    {area.message}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No major improvement area was identified from this session.</p>
            )}
          </div>

          <div className="recommendation-block wide">
            <span>Next Session Focus</span>
            {recommendations.next_session_focus.length > 0 ? (
              <ul>
                {recommendations.next_session_focus.map((focus) => (
                  <li key={focus}>{focus}</li>
                ))}
              </ul>
            ) : (
              <p>No next-session focus items were returned.</p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

export function FinalSummary({ session, summary, recommendations, events = [], alerts = [] }: FinalSummaryProps) {
  if (session?.status !== "ended") return null;

  const importantEvents = [...alerts, ...events].slice(-3).reverse();
  const averages = summary?.averages;
  const emotionSummary = summary?.emotion_summary;
  const hasAnalyzedFrames = (summary?.frame_count ?? 0) > 0;

  return (
    <section className="final-summary glass-card">
      <div className="panel-heading">
        <span className="panel-icon">
          <ClipboardCheck size={18} />
        </span>
        <div>
          <p>Final summary</p>
          <h2>Session ended</h2>
        </div>
      </div>

      {!summary ? (
        <p className="panel-copy">Ending session and building summary...</p>
      ) : !hasAnalyzedFrames ? (
        <p className="panel-copy">Not enough session data to build a summary.</p>
      ) : (
        <>
          <div className="summary-grid">
            <div>
              <TrendingUp size={18} />
              <span>Session Final Score</span>
              <strong>{formatScore(averages?.final)}</strong>
            </div>
            <div>
              <span>Average Eye Contact</span>
              <strong>{formatScore(averages?.eye_contact)}</strong>
            </div>
            <div>
              <span>Average Posture</span>
              <strong>{formatScore(averages?.posture)}</strong>
            </div>
            <div>
              <span>Average Stability</span>
              <strong>{formatScore(averages?.stability)}</strong>
            </div>
            <div>
              <span>Average Self-Touch</span>
              <strong>{formatScore(averages?.self_touch)}</strong>
            </div>
            <div>
              <span>Average Expression</span>
              <strong>{formatScore(averages?.expression)}</strong>
            </div>
            <div>
              <span>Dominant Facial Expression / Emotion</span>
              <strong>{formatLabel(emotionSummary?.dominant_emotion)}</strong>
            </div>
            <div>
              <span>Average Emotion Confidence</span>
              <strong>{formatPercent(emotionSummary?.average_confidence)}</strong>
            </div>
            <div>
              <span>Positive Emotion Probability</span>
              <strong>{formatPercent(emotionSummary?.average_positive_prob)}</strong>
            </div>
            <div>
              <span>Emotion Distribution</span>
              <strong>{formatDistribution(summary)}</strong>
            </div>
            <div>
              <span>Analyzed Frames</span>
              <strong>{summary.frame_count}</strong>
            </div>
            <div>
              <Clock size={18} />
              <span>Paused</span>
              <strong>{Math.round(session?.total_paused_seconds ?? 0)}s</strong>
            </div>
          </div>

          {importantEvents.length > 0 && (
            <div className="summary-events">
              {importantEvents.map((event, index) => (
                <p key={event.id ?? index}>{event.title ?? event.description ?? "Interview event"}</p>
              ))}
            </div>
          )}

          <RecommendationsSection recommendations={recommendations} />
        </>
      )}
    </section>
  );
}
