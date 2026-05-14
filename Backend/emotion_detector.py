from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import uuid

import cv2
import numpy as np

from config import (
    EMOTION_MIN_CONFIDENCE,
    EMOTION_SERVICE_TIMEOUT_SECONDS,
    EMOTION_SERVICE_URL,
    ENABLE_EMOTION_SCORING,
    ENABLE_REMOTE_EMOTION_SERVICE,
)

DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "best_resnet50v2_finetuned.keras"
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

POSITIVE_EXPRESSION_LABELS = {"neutral", "happy"}
TENSE_EXPRESSION_LABELS = {"sad", "angry", "fear", "disgust", "surprise"}

_model: Any | None = None
_model_load_attempted = False
_model_load_error: str | None = None
_remote_config_logged = False


def _empty_probabilities() -> dict[str, float]:
    return {label: 0.0 for label in EMOTION_LABELS}


def _build_result(
    *,
    label: str | None = None,
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    positive_prob: float = 0.0,
    tense_prob: float = 0.0,
    score: float | None = None,
    available: bool = False,
    status: str = "unavailable",
) -> dict[str, Any]:
    clamped_score = None if score is None else max(0.0, min(100.0, score))
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities or _empty_probabilities(),
        "positive_prob": positive_prob,
        "tense_prob": tense_prob,
        "score": clamped_score,
        "available": available,
        "status": status,
    }


def build_raw_emotion_result(emotion_result: dict[str, Any]) -> dict[str, Any]:
    status = str(emotion_result.get("status") or "unavailable")
    label = emotion_result.get("label")
    confidence = emotion_result.get("confidence")
    probabilities = emotion_result.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = _empty_probabilities()

    normalized_probabilities = _empty_probabilities()
    for emotion_label in EMOTION_LABELS:
        value = probabilities.get(emotion_label, 0.0)
        try:
            normalized_probabilities[emotion_label] = float(value)
        except (TypeError, ValueError):
            normalized_probabilities[emotion_label] = 0.0

    has_prediction = (
        label is not None
        and isinstance(confidence, (int, float))
        and status in {"ok", "low_confidence"}
    )
    return {
        "label": str(label) if label is not None else None,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
        "probabilities": normalized_probabilities,
        "available": has_prediction,
        "status": status,
    }


def _resolve_model_path() -> Path:
    return (Path(__file__).resolve().parent / DEFAULT_MODEL_RELATIVE_PATH).resolve()


def _service_unavailable_result() -> dict[str, Any]:
    return _build_result(status="service_unavailable")


def _emotion_service_endpoint_url() -> str:
    parsed_url = urlparse(EMOTION_SERVICE_URL)
    path = parsed_url.path.rstrip("/")
    if not path:
        path = "/predict-emotion"
    return urlunparse(parsed_url._replace(path=path, params="", query="", fragment=""))


def _log_remote_config_once() -> None:
    global _remote_config_logged
    if _remote_config_logged:
        return
    _remote_config_logged = True
    print(
        "[emotion-service] configured",
        {
            "url": EMOTION_SERVICE_URL,
            "endpoint_url": _emotion_service_endpoint_url(),
            "timeout_seconds": EMOTION_SERVICE_TIMEOUT_SECONDS,
            "remote_enabled": ENABLE_REMOTE_EMOTION_SERVICE,
        },
    )


def _load_model() -> Any | None:
    global _model, _model_load_attempted, _model_load_error

    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    _model_load_attempted = True
    model_path = _resolve_model_path()
    if not model_path.exists():
        _model_load_error = f"model_not_found: {model_path}"
        return None

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        _model_load_error = str(exc)
        _model = None

    return _model


def get_model_load_error() -> str | None:
    return _model_load_error


def _face_crop_from_landmarks(frame_bgr: np.ndarray, landmarks: Any) -> np.ndarray | None:
    frame_height, frame_width = frame_bgr.shape[:2]
    points = landmarks.landmark
    if not points:
        return None

    x_values = [point.x * frame_width for point in points]
    y_values = [point.y * frame_height for point in points]
    x_min = max(0, int(min(x_values)))
    x_max = min(frame_width, int(max(x_values)))
    y_min = max(0, int(min(y_values)))
    y_max = min(frame_height, int(max(y_values)))

    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width < 8 or box_height < 8:
        return None

    margin_x = int(box_width * 0.20)
    margin_y = int(box_height * 0.25)
    x_min = max(0, x_min - margin_x)
    x_max = min(frame_width, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(frame_height, y_max + margin_y)

    if x_max <= x_min or y_max <= y_min:
        return None
    return frame_bgr[y_min:y_max, x_min:x_max]


def _as_probabilities(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions[0], dtype=np.float32)
    if values.size < len(EMOTION_LABELS):
        padded = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
        padded[: values.size] = values
        values = padded
    values = values[: len(EMOTION_LABELS)]

    total = float(np.sum(values))
    if np.any(values < 0.0) or total <= 0.0 or abs(total - 1.0) > 0.05:
        values = values - np.max(values)
        exp_values = np.exp(values)
        values = exp_values / max(float(np.sum(exp_values)), 1e-6)
    else:
        values = values / total
    return values


def _normalize_remote_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _service_unavailable_result()

    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, dict):
        return _build_result(status="prediction_error")

    normalized_probabilities = _empty_probabilities()
    for label in EMOTION_LABELS:
        value = probabilities.get(label, 0.0)
        try:
            normalized_probabilities[label] = float(value)
        except (TypeError, ValueError):
            normalized_probabilities[label] = 0.0

    score = payload.get("score")
    try:
        score_value = None if score is None else float(score)
    except (TypeError, ValueError):
        score_value = None

    confidence = payload.get("confidence")
    try:
        confidence_value = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_value = None

    positive_prob = payload.get("positive_prob", 0.0)
    tense_prob = payload.get("tense_prob", 0.0)
    try:
        positive_prob_value = float(positive_prob)
    except (TypeError, ValueError):
        positive_prob_value = 0.0
    try:
        tense_prob_value = float(tense_prob)
    except (TypeError, ValueError):
        tense_prob_value = 0.0

    label = payload.get("label")
    if label is not None:
        label = str(label)

    status = str(payload.get("status") or "unavailable")
    has_prediction = (
        status in {"ok", "low_confidence"}
        and label is not None
        and confidence_value is not None
        and score_value is not None
    )
    return _build_result(
        label=label,
        confidence=confidence_value,
        probabilities=normalized_probabilities,
        positive_prob=positive_prob_value,
        tense_prob=tense_prob_value,
        score=score_value,
        available=has_prediction,
        status=status,
    )


def _multipart_body(
    *,
    field_name: str,
    filename: str,
    content_type: str,
    file_bytes: bytes,
    fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----InterviewAIEmotionBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ],
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ],
    )
    return b"".join(chunks), boundary


def _detect_expression_remote(face_crop: np.ndarray) -> dict[str, Any]:
    _log_remote_config_once()
    success, encoded_image = cv2.imencode(".jpg", face_crop)
    if not success:
        return _service_unavailable_result()

    endpoint_url = _emotion_service_endpoint_url()
    print("[emotion-service] call attempted", {"url": endpoint_url})

    body, boundary = _multipart_body(
        field_name="image",
        filename="face.jpg",
        content_type="image/jpeg",
        file_bytes=encoded_image.tobytes(),
        fields={"is_face_crop": "true"},
    )
    request = Request(
        endpoint_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=EMOTION_SERVICE_TIMEOUT_SECONDS) as response:
            response_body = response.read()
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(
            "[emotion-service] call failed",
            {
                "url": EMOTION_SERVICE_URL,
                "endpoint_url": endpoint_url,
                "reason": type(exc).__name__,
                "detail": str(exc)[:180],
            },
        )
        return _service_unavailable_result()

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(
            "[emotion-service] call failed",
            {
                "url": EMOTION_SERVICE_URL,
                "endpoint_url": endpoint_url,
                "reason": type(exc).__name__,
                "detail": str(exc)[:180],
            },
        )
        return _service_unavailable_result()

    result = _normalize_remote_result(payload)
    print(
        "[emotion-service] call succeeded",
        {
            "url": endpoint_url,
            "status": result["status"],
            "label": result["label"],
            "available": result["available"],
        },
    )
    return result


def detect_expression(frame_bgr: np.ndarray, face_landmarks: Any | None) -> dict[str, Any]:
    if not ENABLE_EMOTION_SCORING:
        return _build_result(status="unavailable")

    if face_landmarks is None:
        return _build_result(status="face_not_detected")

    face_crop = _face_crop_from_landmarks(frame_bgr, face_landmarks)
    if face_crop is None:
        return _build_result(status="face_not_detected")

    if ENABLE_REMOTE_EMOTION_SERVICE:
        return _detect_expression_remote(face_crop)

    model = _load_model()
    if model is None:
        return _build_result(status="model_not_loaded")

    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) < 4:
        return _build_result(status="model_not_loaded")

    img_height = input_shape[1]
    img_width = input_shape[2]
    channels = input_shape[3]
    if img_height is None or img_width is None or channels not in (1, 3):
        return _build_result(status="model_not_loaded")

    face_resized = cv2.resize(face_crop, (int(img_width), int(img_height)))
    if int(channels) == 1:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_resized = np.expand_dims(face_resized, axis=-1)
    else:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    input_data = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
    try:
        predictions = model.predict(input_data, verbose=0)
    except Exception:
        return _build_result(status="model_not_loaded")
    probabilities_array = _as_probabilities(predictions)
    probabilities = {
        label: float(probabilities_array[index])
        for index, label in enumerate(EMOTION_LABELS)
    }

    predicted_index = int(np.argmax(probabilities_array))
    label = EMOTION_LABELS[predicted_index]
    confidence = float(probabilities_array[predicted_index])
    positive_prob = sum(probabilities[label] for label in POSITIVE_EXPRESSION_LABELS)
    tense_prob = sum(probabilities[label] for label in TENSE_EXPRESSION_LABELS)
    expression_score = positive_prob * 100.0 + tense_prob * 25.0
    status = "ok" if confidence >= EMOTION_MIN_CONFIDENCE else "low_confidence"

    return _build_result(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        positive_prob=positive_prob,
        tense_prob=tense_prob,
        score=expression_score,
        available=True,
        status=status,
    )
