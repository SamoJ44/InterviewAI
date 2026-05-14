import { AlertTriangle, CheckCircle2, Clock, Eye, HandHeart, Info, ShieldCheck, UserRound } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeFrame,
  endSession,
  getBackendUrl,
  pauseSession,
  resumeSession,
  startSession,
} from "../api/client";
import type {
  AnalyzeFrameResponse,
  Calibration,
  Flags,
  Scores,
  Session,
  SessionRecommendations,
  SessionStatus,
  SessionSummary,
  TimelineItem,
} from "../types/backend";
import { CalibrationPanel } from "./CalibrationPanel";
import { CameraPanel } from "./CameraPanel";
import { DetectionFlags } from "./DetectionFlags";
import { EventsPanel } from "./EventsPanel";
import { ExpressionCard } from "./ExpressionCard";
import { FinalSummary } from "./FinalSummary";
import { Layout } from "./Layout";
import { RawEmotionPanel } from "./RawEmotionPanel";
import { ScoreCard } from "./ScoreCard";
import { ScoreRing } from "./ScoreRing";
import { SessionControls } from "./SessionControls";
import { StatusBadge } from "./StatusBadge";

const ANALYSIS_INTERVAL_MS = 800;
const DEBUG_STABILITY = import.meta.env.DEV;

function deriveStatusFromResponse(response: AnalyzeFrameResponse, current: SessionStatus): SessionStatus {
  if (response.session?.status) return response.session.status;
  if (response.calibration?.status === "preparing") return "preparing";
  if (response.calibration?.active || response.flags?.calibration_active) return "calibrating";
  if (current === "preparing" || current === "calibrating") return "active";
  return current;
}

function cameraState(status: SessionStatus, cameraActive: boolean, busy: boolean) {
  if (busy && !cameraActive) return "connecting" as const;
  if (status === "paused") return "paused" as const;
  if (status === "ended") return "ended" as const;
  if (cameraActive) return "live" as const;
  return "off" as const;
}

function formatDuration(session: Session | null) {
  if (!session?.started_at) return "Unavailable";
  const start = new Date(session.started_at).getTime();
  const end = session.ended_at ? new Date(session.ended_at).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return "Unavailable";
  const totalSeconds = Math.max(0, Math.round((end - start) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function shouldShowCalibration(calibration: Calibration | null) {
  if (!calibration) return false;
  return (
    calibration.status === "preparing" ||
    calibration.status === "calibrating" ||
    calibration.status === "failed" ||
    calibration.ready === false
  );
}

function calibrationInstruction(calibration: Calibration | null, status: string) {
  if (status === "preparing") return "Sit upright, relax your shoulders, and face the camera.";
  if (status === "calibrating") return "Hold still while we learn your neutral posture.";
  if (status === "failed") return "Calibration failed. Make sure your face and upper body are visible.";
  return calibration?.message ?? "Scores will appear after calibration.";
}

function getCalibrationStatus(calibration: Calibration | null) {
  return calibration?.status ?? (calibration?.ready ? "ready" : "not_ready");
}

function scoreStatus(label: string, value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return `${label}: No data yet`;
  if (value >= 80) {
    if (label === "Stability") return `${label}: ${Math.round(value)} - Stable`;
    if (label === "Self Touch") return `${label}: ${Math.round(value)} - Controlled`;
    return `${label}: ${Math.round(value)} - Excellent`;
  }
  if (value >= 60) return `${label}: ${Math.round(value)} - Good`;
  return `${label}: ${Math.round(value)} - Needs attention`;
}

function buildCoachingTips(flags: Flags | null, scores: Scores | null) {
  const tips: Array<{ tone: "danger" | "warning" | "success"; title: string; detail: string }> = [];

  if (flags?.face_detected === false) {
    tips.push({ tone: "danger", title: "Face not detected", detail: "Center yourself in the camera." });
  }
  if (flags?.pose_detected === false) {
    tips.push({ tone: "danger", title: "Pose not detected", detail: "Make sure your upper body is visible." });
  }
  if (flags?.hands_detected === false && flags?.pose_detected !== false) {
    tips.push({ tone: "warning", title: "Hands not detected", detail: "Self-touch tracking may be limited." });
  }
  if (flags?.eye_status === "CLOSED" || flags?.eye_status === "NO_FACE") {
    tips.push({ tone: "warning", title: "Eye contact needs attention", detail: "Try to look closer to the camera." });
  }
  if (flags?.self_touch_active) {
    tips.push({ tone: "warning", title: "Self-touch detected", detail: "Try to keep hands relaxed." });
  }
  if (typeof scores?.eye_contact === "number" && scores.eye_contact < 50) {
    tips.push({ tone: "warning", title: "Low eye contact", detail: "Look toward the camera more often." });
  }
  if (typeof scores?.posture === "number" && scores.posture < 60) {
    tips.push({ tone: "warning", title: "Posture dip", detail: "Sit upright and relax your shoulders." });
  }
  if (typeof scores?.stability === "number" && scores.stability < 60) {
    tips.push({ tone: "warning", title: "Movement is high", detail: "Try to reduce unnecessary movement." });
  }

  if (tips.length === 0 && scores) {
    tips.push({ tone: "success", title: "Everything looks stable", detail: "Keep going." });
  }
  if (tips.length === 0) {
    tips.push({ tone: "success", title: "Ready when you are", detail: "Start a session to begin live coaching." });
  }

  return tips.slice(0, 3);
}

function importantTimelineItems(events: TimelineItem[], alerts: TimelineItem[]) {
  const alertSet = new Set(alerts);
  return [...alerts, ...events]
    .filter((item) => alertSet.has(item) || item.severity === "warning" || item.severity === "critical")
    .slice(-3)
    .reverse();
}

export function LiveSession() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const statusRef = useRef<SessionStatus>("idle");
  const sessionIdRef = useRef<string | null>(null);

  const [session, setSession] = useState<Session | null>(null);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [scores, setScores] = useState<Scores | null>(null);
  const [sessionScores, setSessionScores] = useState<Partial<Scores> | null>(null);
  const [flags, setFlags] = useState<Flags | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [events, setEvents] = useState<TimelineItem[]>([]);
  const [alerts, setAlerts] = useState<TimelineItem[]>([]);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [sessionRecommendations, setSessionRecommendations] = useState<SessionRecommendations | null>(null);
  const [lastResponse, setLastResponse] = useState<AnalyzeFrameResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [lastFrameAt, setLastFrameAt] = useState<string | null>(null);
  const [hasOfficialScores, setHasOfficialScores] = useState(false);

  const updateStatus = useCallback((nextStatus: SessionStatus) => {
    statusRef.current = nextStatus;
    setSessionStatus(nextStatus);
  }, []);

  const stopFrameLoop = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopFrameLoop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, [stopFrameLoop]);

  const waitForAnalysisIdle = useCallback(async () => {
    for (let attempt = 0; attempt < 30; attempt += 1) {
      if (!inFlightRef.current) return;
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }
  }, []);

  const captureFrameBlob = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;

    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, width, height);

    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.82);
    });
  }, []);

  const sendCurrentFrame = useCallback(async () => {
    const sessionId = sessionIdRef.current;
    const status = statusRef.current;
    if (!sessionId || status === "idle" || status === "paused" || status === "ended" || inFlightRef.current) {
      return;
    }

    const frameBlob = await captureFrameBlob();
    if (!frameBlob) return;

    inFlightRef.current = true;
    try {
      const response = await analyzeFrame({ frameBlob, sessionId, drawOverlay: false });
      if (response.scores) setScores(response.scores);
      setSessionScores(response.session_scores ?? null);
      setFlags(response.flags ?? null);
      setCalibration(response.calibration ?? null);
      setEvents(response.events ?? []);
      setAlerts(response.alerts ?? []);
      setLastResponse(response);
      setError(null);
      setLastFrameAt(new Date().toLocaleTimeString());

      if (response.session) {
        setSession(response.session);
      }
      const nextStatus = deriveStatusFromResponse(response, status);
      const responseCalibrationStatus = getCalibrationStatus(response.calibration ?? null);
      const responseCalibrationReady =
        response.calibration?.ready === true || responseCalibrationStatus === "ready";
      if (nextStatus === "active" && responseCalibrationReady && response.scores) {
        setHasOfficialScores(true);
      }
      updateStatus(nextStatus);

      if (DEBUG_STABILITY && response.scores) {
        console.debug("[stability]", {
          backend: response.scores.stability,
          displayed: response.scores.stability,
          sessionStatus: nextStatus,
          debug: response.stability_debug,
        });
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analyze frame failed.");
    } finally {
      inFlightRef.current = false;
    }
  }, [captureFrameBlob, updateStatus]);

  const startFrameLoop = useCallback(() => {
    stopFrameLoop();
    void sendCurrentFrame();
    timerRef.current = window.setInterval(() => {
      void sendCurrentFrame();
    }, ANALYSIS_INTERVAL_MS);
  }, [sendCurrentFrame, stopFrameLoop]);

  const attachCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API is unavailable in this browser.");
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
    setCameraActive(true);
  }, []);

  const handleStart = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const nextSession = await startSession();
      sessionIdRef.current = nextSession.id;
      setSession(nextSession);
      setScores(null);
      setSessionScores(null);
      setHasOfficialScores(false);
      setSessionSummary(null);
      setSessionRecommendations(null);
      setEvents([]);
      setAlerts([]);
      setLastResponse(null);
      setLastFrameAt(null);
      updateStatus(nextSession.status ?? "preparing");
      await attachCamera();
      startFrameLoop();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Session start failed.");
      updateStatus("idle");
    } finally {
      setBusy(false);
    }
  }, [attachCamera, startFrameLoop, updateStatus]);

  const handlePause = useCallback(async () => {
    if (!sessionIdRef.current) return;
    setBusy(true);
    setError(null);
    try {
      const nextSession = await pauseSession(sessionIdRef.current);
      setSession(nextSession);
      updateStatus("paused");
      stopFrameLoop();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Session pause failed.");
    } finally {
      setBusy(false);
    }
  }, [stopFrameLoop, updateStatus]);

  const handleResume = useCallback(async () => {
    if (!sessionIdRef.current) return;
    setBusy(true);
    setError(null);
    try {
      const nextSession = await resumeSession(sessionIdRef.current);
      setSession(nextSession);
      updateStatus(nextSession.status === "paused" ? "active" : nextSession.status ?? "active");
      startFrameLoop();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Session resume failed.");
    } finally {
      setBusy(false);
    }
  }, [startFrameLoop, updateStatus]);

  const handleEnd = useCallback(async () => {
    if (!sessionIdRef.current) return;
    setBusy(true);
    setError(null);
    stopFrameLoop();
    try {
      await waitForAnalysisIdle();
      const endedSession = await endSession(sessionIdRef.current);
      setSession(endedSession.session);
      setSessionSummary(endedSession.summary);
      setSessionRecommendations(endedSession.recommendations);
      updateStatus("ended");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Session end failed.");
      updateStatus("ended");
    } finally {
      stopCamera();
      setBusy(false);
    }
  }, [stopCamera, stopFrameLoop, updateStatus, waitForAnalysisIdle]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const effectiveSessionStatus = session?.status ?? sessionStatus ?? "idle";
  const calibrationStatus = getCalibrationStatus(calibration);
  const isCalibrationReady = calibration?.ready === true || calibrationStatus === "ready";
  const isSessionOfficiallyActive = effectiveSessionStatus === "active" && isCalibrationReady;
  const isEnded = effectiveSessionStatus === "ended";
  const hasEndedSummary = isEnded && sessionSummary !== null;
  const shouldShowScores = hasOfficialScores || hasEndedSummary;
  const frozen = (effectiveSessionStatus === "paused" || effectiveSessionStatus === "ended") && shouldShowScores;
  const scoreFreezeCopy =
    effectiveSessionStatus === "paused" && shouldShowScores
      ? "Session paused - showing last scores"
      : effectiveSessionStatus === "ended" && shouldShowScores
        ? "Session ended - showing last scores"
        : null;
  const sessionFinal = isEnded ? sessionSummary?.averages.final : sessionScores?.final;
  const cameraMode = cameraState(effectiveSessionStatus, cameraActive, busy);
  const visibleScores = shouldShowScores ? scores : null;
  const visibleEmotion = shouldShowScores ? lastResponse?.emotion ?? null : null;
  const visibleRawEmotion = shouldShowScores ? lastResponse?.raw_emotion ?? null : null;
  const coachingTips = useMemo(() => buildCoachingTips(flags, visibleScores), [flags, visibleScores]);
  const importantItems = useMemo(() => importantTimelineItems(events, alerts), [events, alerts]);
  const showCalibration = shouldShowCalibration(calibration);

  const scoreCards = useMemo(
    () => [
      { label: "Eye Contact", value: visibleScores?.eye_contact, icon: Eye },
      { label: "Posture", value: visibleScores?.posture, icon: UserRound },
      { label: "Stability", value: visibleScores?.stability, icon: ShieldCheck },
      { label: "Self Touch", value: visibleScores?.self_touch, icon: HandHeart },
    ],
    [visibleScores],
  );

  return (
    <Layout status={effectiveSessionStatus}>
      <section className="live-session-shell">
        <div className="camera-column">
          <div className="camera-stage">
            <CameraPanel
              videoRef={videoRef}
              canvasRef={canvasRef}
              cameraActive={cameraActive}
              cameraState={cameraMode}
              sessionStatus={effectiveSessionStatus}
              lastFrameAt={lastFrameAt}
            />
            <div className="camera-session-strip">
              <StatusBadge label="Session" value={effectiveSessionStatus} kind={effectiveSessionStatus} />
              {calibration?.ready && !showCalibration && <StatusBadge value="Calibration ready" kind="success" />}
            </div>
            <div className="camera-controls-overlay">
              <SessionControls
                status={effectiveSessionStatus}
                busy={busy}
                onStart={handleStart}
                onPause={handlePause}
                onResume={handleResume}
                onEnd={handleEnd}
              />
            </div>
          </div>
        </div>

        <aside className="session-info-panel live-coaching-panel" aria-label="Live coaching panel">
          {!shouldShowScores && (
            <section className="info-section">
              <div className="section-heading compact">
                <div>
                  <p>Setup</p>
                  <h2>{isEnded ? "Session ended" : "Calibration status"}</h2>
                </div>
              </div>
              {isEnded ? (
                <section className="calibration-gate-card glass-card failed">
                  <AlertTriangle size={20} />
                  <div>
                    <h3>Session ended before calibration completed</h3>
                    <p>No valid session score is available.</p>
                  </div>
                </section>
              ) : (
                <>
                  <CalibrationPanel calibration={calibration} />
                  <section className={`calibration-gate-card glass-card ${calibrationStatus}`}>
                    {calibrationStatus === "failed" ? <AlertTriangle size={20} /> : <Info size={20} />}
                    <div>
                      <h3>Scores will appear after calibration.</h3>
                      <p>{calibrationInstruction(calibration, calibrationStatus)}</p>
                      {effectiveSessionStatus === "paused" && <strong>Session paused during calibration.</strong>}
                    </div>
                  </section>
                </>
              )}
            </section>
          )}

          {shouldShowScores && (
            <section className="info-section">
              <div className="section-heading compact">
                <div>
                  <p>{isEnded ? "Summary" : "Live coaching"}</p>
                  <h2>{isEnded ? "Session summary" : "Live score summary"}</h2>
                </div>
                {scoreFreezeCopy && <span>{scoreFreezeCopy}</span>}
              </div>
              <div className="summary-stack">
                <ScoreRing label={isEnded ? "Session Final Score" : "Live Final Score"} value={isEnded ? sessionFinal : visibleScores?.final} />
                <section className="glass-card session-final-card">
                  <p>Session final score</p>
                  <strong>{typeof sessionFinal === "number" ? Math.round(sessionFinal) : "Not enough data"}</strong>
                  <span>
                    {typeof sessionFinal === "number"
                      ? "Based on completed session samples."
                      : "No valid score samples were captured."}
                  </span>
                </section>
              </div>
            </section>
          )}

          {effectiveSessionStatus === "paused" && shouldShowScores && (
            <section className="coaching-card paused glass-card">
              <Info size={18} />
              <div>
                <h3>Session paused</h3>
                <p>Scores are frozen until you resume.</p>
              </div>
            </section>
          )}

          {isEnded && shouldShowScores ? (
            <FinalSummary
              session={session}
              summary={sessionSummary}
              recommendations={sessionRecommendations}
              events={events}
              alerts={alerts}
            />
          ) : !isEnded && shouldShowScores ? (
            <>
              <section className="info-section">
                <ExpressionCard emotion={visibleEmotion} frozen={frozen} />
                <RawEmotionPanel rawEmotion={visibleRawEmotion} frozen={frozen} />
              </section>

              <section className="info-section">
                <div className="section-heading compact">
                  <div>
                    <p>Behavior</p>
                    <h2>Core signals</h2>
                  </div>
                </div>
                <div className="score-grid compact-score-grid">
                  {scoreCards.map((card) => (
                    <ScoreCard key={card.label} label={card.label} value={card.value} icon={card.icon} frozen={frozen} />
                  ))}
                </div>
                <div className="score-status-list">
                  {scoreCards.map((card) => (
                    <span key={`${card.label}-status`}>{scoreStatus(card.label, card.value)}</span>
                  ))}
                </div>
              </section>

              {showCalibration ? (
                <section className="info-section">
                  <CalibrationPanel calibration={calibration} />
                </section>
              ) : (
                calibration?.ready && (
                  <div className="compact-ready-badge">
                    <CheckCircle2 size={16} />
                    Calibration ready
                  </div>
                )
              )}

              <section className="info-section">
                <div className="section-heading compact">
                  <div>
                    <p>Coaching</p>
                    <h2>Live tips</h2>
                  </div>
                </div>
                <div className="coaching-list">
                  {coachingTips.map((tip) => (
                    <article className={`coaching-card glass-card ${tip.tone}`} key={tip.title}>
                      {tip.tone === "success" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                      <div>
                        <h3>{tip.title}</h3>
                        <p>{tip.detail}</p>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="info-section">
                <EventsPanel events={[]} alerts={importantItems} />
                {events.length + alerts.length > importantItems.length && (
                  <p className="view-all-note">More events are available in Advanced details.</p>
                )}
              </section>
            </>
          ) : (
            <>
              <section className="info-section">
                <div className="section-heading compact">
                  <div>
                    <p>Coaching</p>
                    <h2>Setup checks</h2>
                  </div>
                </div>
                <div className="coaching-list">
                  {coachingTips.map((tip) => (
                    <article className={`coaching-card glass-card ${tip.tone}`} key={tip.title}>
                      {tip.tone === "success" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                      <div>
                        <h3>{tip.title}</h3>
                        <p>{tip.detail}</p>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </>
          )}

          <details className="advanced-details glass-card">
            <summary>Advanced details</summary>
            <div className="advanced-details-body">
              <div className="session-meta-grid">
                <div className="glass-card meta-card">
                  <Info size={18} />
                  <span>Status</span>
                  <StatusBadge value={sessionStatus} kind={sessionStatus} />
                </div>
                <div className="glass-card meta-card">
                  <span>Session ID</span>
                  <strong>{session?.id ?? sessionIdRef.current ?? "Not started"}</strong>
                </div>
                <div className="glass-card meta-card">
                  <Clock size={18} />
                  <span>Last frame</span>
                  <strong>{lastFrameAt ?? "No frames yet"}</strong>
                </div>
                <div className="glass-card meta-card">
                  <span>Duration</span>
                  <strong>{formatDuration(session)}</strong>
                </div>
                <div className="glass-card meta-card">
                  <span>Frame metadata</span>
                  <strong>
                    {lastResponse?.frame_meta?.width && lastResponse?.frame_meta?.height
                      ? `${lastResponse.frame_meta.width} x ${lastResponse.frame_meta.height}`
                      : "Awaiting frame"}
                  </strong>
                </div>
                <div className="glass-card meta-card">
                  <span>Reliability</span>
                  <strong>{lastResponse?.score_details?.final?.reliability ?? "Unavailable"}</strong>
                </div>
                <div className="glass-card meta-card">
                  <span>Calibration samples</span>
                  <strong>{calibration?.samples ?? 0}</strong>
                </div>
                <div className="glass-card meta-card">
                  <span>Backend</span>
                  <strong>{getBackendUrl()}</strong>
                </div>
              </div>
              <DetectionFlags flags={flags} />
              <EventsPanel events={events} alerts={alerts} />
              {error && (
                <div className="error-banner" role="alert">
                  <strong>Backend, camera, or API error</strong>
                  <span>{error}</span>
                </div>
              )}
              <details className="debug-drawer glass-card">
                <summary>Raw backend payload</summary>
                <pre>{lastResponse ? JSON.stringify(lastResponse, null, 2) : "No analysis response yet."}</pre>
              </details>
            </div>
          </details>

          {error && (
            <section className="info-section">
              <div className="error-banner" role="alert">
                <strong>Action needed</strong>
                <span>{error}</span>
              </div>
            </section>
          )}
        </aside>
      </section>
    </Layout>
  );
}
