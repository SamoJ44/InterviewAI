import { Camera, CameraOff, Pause, Radio } from "lucide-react";
import type { RefObject } from "react";
import type { SessionStatus } from "../types/backend";

type CameraPanelProps = {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  cameraActive: boolean;
  cameraState: "off" | "connecting" | "live" | "paused" | "ended";
  sessionStatus: SessionStatus;
  lastFrameAt?: string | null;
};

function cameraLabel(state: CameraPanelProps["cameraState"]) {
  if (state === "connecting") return "Connecting";
  if (state === "live") return "Live";
  if (state === "paused") return "Paused";
  if (state === "ended") return "Ended";
  return "Camera off";
}

function cameraIcon(state: CameraPanelProps["cameraState"]) {
  if (state === "live") return <Radio size={16} />;
  if (state === "paused") return <Pause size={16} />;
  if (state === "off") return <CameraOff size={16} />;
  return <Camera size={16} />;
}

export function CameraPanel({
  videoRef,
  canvasRef,
  cameraActive,
  cameraState,
  sessionStatus,
  lastFrameAt,
}: CameraPanelProps) {
  return (
    <section className="camera-card glass-card">
      <div className="camera-frame">
        <video ref={videoRef} className="camera-preview" autoPlay muted playsInline />
        {!cameraActive && (
          <div className="camera-empty">
            <Camera size={42} />
            <span>Camera preview will appear here</span>
          </div>
        )}
        <div className={`camera-pill ${cameraState}`}>
          {cameraIcon(cameraState)}
          {cameraLabel(cameraState)}
        </div>
        <div className="camera-meta">
          <span>{sessionStatus}</span>
          <span>{lastFrameAt ? `Last frame ${lastFrameAt}` : "Awaiting analysis"}</span>
        </div>
      </div>
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </section>
  );
}
