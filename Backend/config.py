from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - backend can still run with defaults.
    load_dotenv = None

BACKEND_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BACKEND_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


FACE_TOUCH_SCALE = 0.21
BODY_TOUCH_SCALE = 0.20
FACE_Z_THRESHOLD = 0.12
FACE_SIDE_Z_THRESHOLD = 0.16
BODY_Z_THRESHOLD = 0.20

PRE_CALIBRATION_COUNTDOWN_SECONDS = 3.0
CALIBRATION_DURATION_SECONDS = 5.0
SMOOTH_ALPHA = 0.8

# Stability tuning.
MOVEMENT_WINDOW_SIZE = 5  # Controls how quickly stability reacts to recent movement.
STABILITY_MOTION_DEADZONE = 0.03  # Ignores tiny jitter and camera/landmark micro-movement.
STABILITY_MOTION_MAX = 0.45  # Movement level that causes near-zero stability.
STABILITY_HORIZONTAL_SWAY_WEIGHT = 1.25
STABILITY_SWAY_WINDOW_SIZE = 8
STABILITY_DIRECTION_CHANGE_PENALTY = 12
STABILITY_STILLNESS_THRESHOLD = 0.025
STABILITY_STILLNESS_RECOVERY_FRAMES = 5
STABILITY_RECOVERY_BOOST = 8.0
STABILITY_DROP_SMOOTH_ALPHA = 0.55
STABILITY_RECOVERY_SMOOTH_ALPHA = 0.35
POSE_VISIBILITY_THRESHOLD = 0.5

CENTRAL_FACE_INDICES = [1, 152, 10]
SIDE_FACE_INDICES = [234, 454, 127, 356, 172, 136, 150, 176, 400, 379, 365, 397]
KEY_HAND_POINTS = [8, 4, 12, 0]

SELF_TOUCH_ACTIVE_FRAMES = 4
SELF_TOUCH_FRAME_PENALTY = 5

GOOD_SCORE_THRESHOLD = 70
BAD_SCORE_THRESHOLD = 40

EMOTION_MIN_CONFIDENCE = 0.45
EMOTION_SMOOTH_ALPHA = 0.75
ENABLE_EMOTION_SCORING = _env_bool("ENABLE_EMOTION_SCORING", True)
ENABLE_REMOTE_EMOTION_SERVICE = _env_bool("ENABLE_REMOTE_EMOTION_SERVICE", True)
EMOTION_SERVICE_URL = os.getenv("EMOTION_SERVICE_URL", "http://127.0.0.1:8765").strip()
EMOTION_SERVICE_TIMEOUT_SECONDS = _env_float("EMOTION_SERVICE_TIMEOUT_SECONDS", 5.0)

FINAL_EXCELLENT_THRESHOLD = 80
FINAL_GOOD_THRESHOLD = 60

FINAL_WEIGHT_EYE = 0.30
FINAL_WEIGHT_POSTURE = 0.22
FINAL_WEIGHT_STABILITY = 0.22
FINAL_WEIGHT_SELF_TOUCH = 0.16

CALIBRATION_TEXT = "Calibration: Sit still and look forward"
