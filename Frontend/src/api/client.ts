import type { AnalyzeFrameResponse, EndSessionResponse, Session, SessionResponse } from "../types/backend";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/$/, "");

type AnalyzeFrameInput = {
  frameBlob: Blob;
  sessionId?: string | null;
  drawOverlay?: boolean;
};

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${BACKEND_URL}${path}`, {
      ...options,
      headers: {
        ...(options?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...options?.headers,
      },
    });
  } catch {
    throw new Error("Backend offline or unreachable. Confirm FastAPI is running on port 8000.");
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}${body ? `: ${body}` : ""}`);
  }

  return response.json() as Promise<T>;
}

function normalizeSession(response: SessionResponse): Session {
  if ("session" in response) {
    return response.session;
  }

  return {
    id: response.id,
    status: response.status ?? "active",
    started_at: "started_at" in response ? response.started_at : null,
    ended_at: "ended_at" in response ? response.ended_at : null,
    paused: "paused" in response ? response.paused : false,
    pause_count: "pause_count" in response ? response.pause_count : 0,
    total_paused_seconds: "total_paused_seconds" in response ? response.total_paused_seconds : 0,
  };
}

export function getBackendUrl() {
  return BACKEND_URL;
}

export async function startSession(): Promise<Session> {
  return normalizeSession(await requestJson<SessionResponse>("/session/start", { method: "POST" }));
}

export async function pauseSession(sessionId: string): Promise<Session> {
  return normalizeSession(
    await requestJson<SessionResponse>("/session/pause", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  );
}

export async function resumeSession(sessionId: string): Promise<Session> {
  return normalizeSession(
    await requestJson<SessionResponse>("/session/resume", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  );
}

export async function endSession(sessionId: string): Promise<EndSessionResponse> {
  return requestJson<EndSessionResponse>("/session/end", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function getSessionStatus(sessionId: string): Promise<Session> {
  return normalizeSession(await requestJson<SessionResponse>(`/session/status/${encodeURIComponent(sessionId)}`));
}

export async function analyzeFrame({
  frameBlob,
  sessionId,
  drawOverlay = false,
}: AnalyzeFrameInput): Promise<AnalyzeFrameResponse> {
  const formData = new FormData();
  formData.append("frame", frameBlob, "frame.jpg");
  if (sessionId) {
    formData.append("session_id", sessionId);
  }
  formData.append("draw_overlay", String(drawOverlay));

  return requestJson<AnalyzeFrameResponse>("/analyze-frame", {
    method: "POST",
    body: formData,
  });
}
