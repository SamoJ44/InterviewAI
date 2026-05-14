from __future__ import annotations

import os
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from emotion_detector import EMOTION_LABELS
from pipeline import PipelineState, Processors, create_processors, init_state, process_frame
from recommendation_service import build_recommendation_input, generate_recommendations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_processors_context: AbstractContextManager[Processors] | None = None
_processors: Processors | None = None
_session_states: dict[str, PipelineState] = {}
_sessions: dict[str, dict[str, Any]] = {}
_session_histories: dict[str, dict[str, Any]] = {}
_session_lock = Lock()
_inference_lock = Lock()
DEBUG_STABILITY = os.getenv("INTERVIEWAI_DEBUG_STABILITY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise HTTPException(
        status_code=400,
        detail="Invalid draw_overlay value. Use true/false.",
    )


def _get_session_state(session_id: str) -> PipelineState:
    with _session_lock:
        state = _session_states.get(session_id)
        if state is None:
            state = init_state()
            _session_states[session_id] = state
        return state


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_score_history() -> dict[str, list[float]]:
    return {
        "eye_contact": [],
        "posture": [],
        "stability": [],
        "self_touch": [],
        "expression": [],
        "final": [],
    }


def _new_emotion_history() -> dict[str, Any]:
    return {
        "label_counts": {},
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "positive_prob_sum": 0.0,
        "positive_prob_count": 0,
        "probability_sums": {label: 0.0 for label in EMOTION_LABELS},
        "probability_count": 0,
    }


def _new_session_history() -> dict[str, Any]:
    return {
        "score_frame_count": 0,
        "score_history": _new_score_history(),
        "emotion_history": _new_emotion_history(),
        "event_counts": {},
        "alert_counts": {},
    }


def _session_payload(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": session["id"],
        "status": session["status"],
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "paused": session.get("status") == "paused",
        "pause_count": session.get("pause_count", 0),
        "total_paused_seconds": session.get("total_paused_seconds", 0),
    }


def _get_existing_session(session_id: str) -> dict[str, Any]:
    with _session_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        return session


def _session_id_from_payload(payload: dict[str, Any] | None) -> str:
    session_id = (payload or {}).get("session_id") or (payload or {}).get("id")
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        raise HTTPException(status_code=400, detail="Missing required session_id.")
    return normalized_session_id


def _sync_session_status(session_id: str, calibration: dict[str, Any]) -> dict[str, Any] | None:
    with _session_lock:
        session = _sessions.get(session_id)
        if session is None or session["status"] in {"paused", "ended"}:
            return session

        calibration_status = calibration.get("status")
        if calibration_status == "preparing":
            session["status"] = "preparing"
        elif calibration_status == "calibrating":
            session["status"] = "calibrating"
        else:
            session["status"] = "active"
        return session


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _score_available(score_details: dict[str, Any], score_name: str) -> bool:
    detail = score_details.get(score_name)
    if isinstance(detail, dict) and "available" in detail:
        return detail.get("available") is True
    return True


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_score(value: float | None) -> int | None:
    return None if value is None else int(round(value))


def _round_probability(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _record_session_frame(
    session_id: str,
    scores: dict[str, Any],
    score_details: dict[str, Any],
    emotion: dict[str, Any],
    events: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
) -> None:
    with _session_lock:
        session = _sessions.get(session_id)
        if session is None or session.get("status") in {"paused", "ended"}:
            return

        history = _session_histories.setdefault(session_id, _new_session_history())
        score_history = history.setdefault("score_history", _new_score_history())
        history["score_frame_count"] = history.get("score_frame_count", 0) + 1

        for score_name in ("eye_contact", "posture", "stability", "self_touch", "final"):
            score_value = scores.get(score_name)
            if _is_number(score_value) and _score_available(score_details, score_name):
                score_history.setdefault(score_name, []).append(float(score_value))

        expression_score = scores.get("expression")
        if _is_number(expression_score) and _score_available(score_details, "expression"):
            score_history.setdefault("expression", []).append(float(expression_score))

        event_counts = history.setdefault("event_counts", {})
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if isinstance(event_type, str) and event_type.strip():
                event_type = event_type.strip()
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

        alert_counts = history.setdefault("alert_counts", {})
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            alert_key = alert.get("type") or alert.get("title") or alert.get("category")
            if isinstance(alert_key, str) and alert_key.strip():
                alert_key = alert_key.strip()
                alert_counts[alert_key] = alert_counts.get(alert_key, 0) + 1

        label = emotion.get("label")
        if not isinstance(label, str) or not label.strip():
            return

        normalized_label = label.strip()
        if normalized_label.lower() == "uncertain":
            return

        status = str(emotion.get("status") or "")
        has_prediction = emotion.get("available") is True or status in {"ok", "low_confidence"}
        if not has_prediction:
            return

        emotion_history = history.setdefault("emotion_history", _new_emotion_history())
        label_counts = emotion_history.setdefault("label_counts", {})
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1

        confidence = emotion.get("confidence")
        if _is_number(confidence):
            emotion_history["confidence_sum"] = emotion_history.get("confidence_sum", 0.0) + float(confidence)
            emotion_history["confidence_count"] = emotion_history.get("confidence_count", 0) + 1

        positive_prob = emotion.get("positive_prob")
        if _is_number(positive_prob):
            emotion_history["positive_prob_sum"] = (
                emotion_history.get("positive_prob_sum", 0.0) + float(positive_prob)
            )
            emotion_history["positive_prob_count"] = emotion_history.get("positive_prob_count", 0) + 1

        probabilities = emotion.get("probabilities")
        if isinstance(probabilities, dict):
            probability_sums = emotion_history.setdefault(
                "probability_sums",
                {emotion_label: 0.0 for emotion_label in EMOTION_LABELS},
            )
            recorded_probability = False
            for emotion_label, probability in probabilities.items():
                if _is_number(probability):
                    probability_sums[str(emotion_label)] = probability_sums.get(str(emotion_label), 0.0) + float(
                        probability,
                    )
                    recorded_probability = True
            if recorded_probability:
                emotion_history["probability_count"] = emotion_history.get("probability_count", 0) + 1


def _history_debug_counts(session_id: str, history: dict[str, Any] | None = None) -> dict[str, Any]:
    history = history if history is not None else _session_histories.get(session_id)
    score_history = (history or {}).get("score_history") or {}
    emotion_history = (history or {}).get("emotion_history") or {}
    label_counts = emotion_history.get("label_counts") or {}
    return {
        "session_id": session_id,
        "stored_score_frames": int((history or {}).get("score_frame_count", 0)),
        "stored_score_samples": {
            score_name: len(values) if isinstance(values, list) else 0
            for score_name, values in score_history.items()
        },
        "stored_emotion_samples": sum(int(count) for count in label_counts.values()),
        "stored_event_samples": sum(int(count) for count in ((history or {}).get("event_counts") or {}).values()),
    }


def _coaching_event_counts(history: dict[str, Any]) -> dict[str, int]:
    event_counts = history.get("event_counts") or {}
    alert_counts = history.get("alert_counts") or {}
    counts = {
        "gaze_away_started": int(event_counts.get("gaze_away_started", 0)),
        "gaze_away_prolonged": int(event_counts.get("gaze_away_prolonged", 0)),
        "self_touch_started": int(event_counts.get("self_touch_started", 0)),
        "posture_deviation_started": int(event_counts.get("posture_deviation_started", 0)),
        "posture_recovered": int(event_counts.get("posture_deviation_recovered", 0)),
        "face_lost": int(event_counts.get("face_lost", 0)),
        "face_reacquired": int(event_counts.get("face_reacquired", 0)),
        "self_touch_ended": int(event_counts.get("self_touch_ended", 0)),
        "tracking_face_lost": int(alert_counts.get("Face tracking lost", 0)),
        "tracking_pose_lost": int(alert_counts.get("Pose tracking lost", 0)),
        "posture_sustained": int(alert_counts.get("Sustained posture deviation", 0)),
        "self_touch_repeated": int(alert_counts.get("Repeated self-touch detected", 0)),
    }
    return {key: value for key, value in counts.items() if value > 0}


def _build_session_summary(session_id: str, history: dict[str, Any]) -> dict[str, Any]:
    score_history = history.get("score_history") or _new_score_history()
    raw_averages = {
        "eye_contact": _average(score_history.get("eye_contact", [])),
        "posture": _average(score_history.get("posture", [])),
        "stability": _average(score_history.get("stability", [])),
        "self_touch": _average(score_history.get("self_touch", [])),
        "expression": _average(score_history.get("expression", [])),
    }
    final_components = [value for value in raw_averages.values() if value is not None]
    raw_averages["final"] = sum(final_components) / len(final_components) if final_components else None

    emotion_history = history.get("emotion_history") or _new_emotion_history()
    label_counts = emotion_history.get("label_counts") or {}
    valid_emotion_frame_count = sum(int(count) for count in label_counts.values())
    dominant_emotion = None
    if valid_emotion_frame_count > 0:
        dominant_emotion = max(label_counts.items(), key=lambda item: item[1])[0]

    distribution = {label: int(label_counts.get(label, 0)) for label in EMOTION_LABELS}
    for label, count in label_counts.items():
        if label not in distribution:
            distribution[label] = int(count)

    confidence_count = emotion_history.get("confidence_count", 0)
    positive_prob_count = emotion_history.get("positive_prob_count", 0)
    probability_count = emotion_history.get("probability_count", 0)
    average_probabilities = None
    if probability_count:
        probability_sums = emotion_history.get("probability_sums") or {}
        average_probabilities = {
            str(label): round(float(total) / probability_count, 3)
            for label, total in probability_sums.items()
        }

    return {
        "frame_count": int(history.get("score_frame_count", 0)),
        "valid_expression_frame_count": len(score_history.get("expression", [])),
        "valid_emotion_frame_count": valid_emotion_frame_count,
        "averages": {score_name: _round_score(value) for score_name, value in raw_averages.items()},
        "event_counts": _coaching_event_counts(history),
        "emotion_summary": {
            "dominant_emotion": dominant_emotion,
            "average_confidence": _round_probability(
                float(emotion_history.get("confidence_sum", 0.0)) / confidence_count
                if confidence_count
                else None,
            ),
            "average_positive_prob": _round_probability(
                float(emotion_history.get("positive_prob_sum", 0.0)) / positive_prob_count
                if positive_prob_count
                else None,
            ),
            "distribution": distribution,
            "average_probabilities": average_probabilities,
        },
        "debug": _history_debug_counts(session_id, history),
    }


@app.on_event("startup")
def startup() -> None:
    global _processors_context, _processors
    _processors_context = create_processors()
    _processors = _processors_context.__enter__()


@app.on_event("shutdown")
def shutdown() -> None:
    global _processors_context, _processors
    if _processors_context is not None:
        _processors_context.__exit__(None, None, None)
    _processors_context = None
    _processors = None
    with _session_lock:
        _session_states.clear()
        _sessions.clear()
        _session_histories.clear()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "interviewai-backend",
        "mode": "api",
    }


@app.post("/session/start")
def start_session() -> dict[str, Any]:
    session_id = str(uuid4())
    session = {
        "id": session_id,
        "status": "preparing",
        "started_at": _utc_now(),
        "ended_at": None,
        "pause_count": 0,
        "total_paused_seconds": 0,
    }

    with _session_lock:
        _sessions[session_id] = session
        _session_states[session_id] = init_state()
        _session_histories[session_id] = _new_session_history()

    return {"session": _session_payload(session)}


@app.post("/session/pause")
def pause_session(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    session_id = _session_id_from_payload(payload)
    session = _get_existing_session(session_id)

    with _session_lock:
        if session["status"] == "ended":
            raise HTTPException(status_code=409, detail="Ended sessions cannot be paused.")
        if session["status"] != "paused":
            session["status"] = "paused"
            session["paused_at"] = datetime.now(timezone.utc)
            session["pause_count"] = session.get("pause_count", 0) + 1

    return {"session": _session_payload(session)}


@app.post("/session/resume")
def resume_session(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    session_id = _session_id_from_payload(payload)
    session = _get_existing_session(session_id)

    with _session_lock:
        if session["status"] == "ended":
            raise HTTPException(status_code=409, detail="Ended sessions cannot be resumed.")
        paused_at = session.pop("paused_at", None)
        if paused_at is not None:
            paused_seconds = (datetime.now(timezone.utc) - paused_at).total_seconds()
            session["total_paused_seconds"] = session.get("total_paused_seconds", 0) + round(paused_seconds, 3)
        session["status"] = "active"

    return {"session": _session_payload(session)}


@app.post("/session/end")
def end_session(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    session_id = _session_id_from_payload(payload)
    session = _get_existing_session(session_id)

    with _session_lock:
        if session["status"] != "ended":
            paused_at = session.pop("paused_at", None)
            if paused_at is not None:
                paused_seconds = (datetime.now(timezone.utc) - paused_at).total_seconds()
                session["total_paused_seconds"] = session.get("total_paused_seconds", 0) + round(paused_seconds, 3)
            session["status"] = "ended"
            session["ended_at"] = _utc_now()
        history = _session_histories.get(session_id) or _new_session_history()
        summary = _build_session_summary(session_id, history)
        recommendation_input = build_recommendation_input(
            session_id=session_id,
            session=session,
            summary=summary,
        )

    print("[session-summary]", summary["debug"])
    recommendations = generate_recommendations(recommendation_input)
    print(
        "[session-end] final payload",
        {
            "session_id": session_id,
            "recommendations_present": recommendations is not None,
        },
    )

    return {
        "session_id": session_id,
        "status": "ended",
        "session": _session_payload(session),
        "summary": summary,
        "recommendations": recommendations,
    }


@app.get("/session/status/{session_id}")
def get_session_status(session_id: str) -> dict[str, Any]:
    session = _get_existing_session(session_id)
    return {"session": _session_payload(session)}


@app.post("/analyze-frame")
def analyze_frame(
    frame: UploadFile | None = File(default=None),
    session_id: str = Form(default="default"),
    draw_overlay: str | bool | None = Form(default=False),
) -> dict[str, Any]:
    if frame is None:
        raise HTTPException(status_code=400, detail="Missing required file field 'frame'.")

    draw_overlay = _parse_bool(draw_overlay)

    content_type = (frame.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPEG or PNG image.",
        )

    file_bytes = frame.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    if image_array.size == 0:
        raise HTTPException(status_code=400, detail="Failed to parse uploaded image bytes.")

    decoded_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if decoded_frame is None:
        raise HTTPException(
            status_code=415,
            detail="Could not decode image. Supported formats are JPEG and PNG.",
        )

    normalized_session_id = (session_id or "default").strip() or "default"
    session = _sessions.get(normalized_session_id)
    if session is not None and session.get("status") in {"paused", "ended"}:
        raise HTTPException(
            status_code=409,
            detail=f"Session '{normalized_session_id}' is {session['status']}.",
        )

    state = _get_session_state(normalized_session_id)

    with _inference_lock:
        if _processors is None:
            raise HTTPException(status_code=503, detail="Processors are not initialized.")
        result = process_frame(decoded_frame, _processors, state, draw=draw_overlay)
        stability_debug = result.get("stability_debug", {})
        if DEBUG_STABILITY and stability_debug.get("debug_sequence", 0) % 5 == 0:
            print(
                "[stability]",
                {
                    "session_id": normalized_session_id,
                    "stability_score": result["stability_score"],
                    "pose_detected": result["pose_detected"],
                    "delta_time": stability_debug.get("delta_time"),
                    "nose_movement_pixels": stability_debug.get("nose_movement_pixels"),
                    "shoulder_midpoint_movement_pixels": stability_debug.get(
                        "shoulder_midpoint_movement_pixels",
                    ),
                    "shoulder_width": stability_debug.get("shoulder_width"),
                    "normalized_motion_rate": stability_debug.get("normalized_motion_rate"),
                    "horizontal_sway_rate": stability_debug.get("horizontal_sway_rate"),
                    "motion_signal": stability_debug.get("motion_signal"),
                    "average_motion_rate": stability_debug.get("average_motion_rate"),
                    "effective_motion": stability_debug.get("effective_motion"),
                    "previous_stability_score": stability_debug.get("previous_stability_score"),
                    "raw_stability_score": stability_debug.get("raw_stability_score"),
                    "smoothed_stability_score": stability_debug.get("smoothed_stability_score"),
                    "stability_smoothing_alpha": stability_debug.get("stability_smoothing_alpha"),
                    "direction_changes": stability_debug.get("direction_changes"),
                    "still_frames": stability_debug.get("still_frames"),
                    "movement_window_length": stability_debug.get("movement_window_length"),
                },
            )

    processed_frame = result["frame"]
    frame_height, frame_width = processed_frame.shape[:2]
    scores = {
        "eye_contact": result["eye_contact_score"],
        "posture": result["posture_score"],
        "stability": result["stability_score"],
        "self_touch": result["self_touch_score"],
        "expression": result["expression_score"],
        "final": result["final_score"],
    }
    _record_session_frame(
        normalized_session_id,
        scores,
        result["score_details"],
        result["emotion"],
        result["events"],
        result["alerts"],
    )
    session = _sync_session_status(normalized_session_id, result["calibration"])
    return {
        "session_id": normalized_session_id,
        "session": _session_payload(session) if session is not None else None,
        "scores": scores,
        "session_debug": _history_debug_counts(normalized_session_id),
        "emotion": result["emotion"],
        "raw_emotion": result["raw_emotion"],
        "score_details": result["score_details"],
        "stability_debug": result["stability_debug"],
        "self_touch_details": result["self_touch_details"],
        "calibration": result["calibration"],
        "flags": {
            "eye_status": result["eye_status_text"],
            "self_touch_active": result["self_touch_active"],
            "calibration_active": result["calibration_active"],
            "face_detected": result["face_detected"],
            "pose_detected": result["pose_detected"],
            "hands_detected": result["hands_detected"],
            "gaze_away_active": result["gaze_away_active"],
        },
        "frame_meta": {
            "width": frame_width,
            "height": frame_height,
        },
        "events": result["events"],
        "alerts": result["alerts"],
    }
