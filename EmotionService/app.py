from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="InterviewAI Emotion Service")

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / ".."
    / "Backend"
    / "models"
    / "best_resnet50v2_finetuned.keras"
)
EMOTION_MODEL_PATH = Path(os.getenv("EMOTION_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
EMOTION_MIN_CONFIDENCE = 0.45
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
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities or _empty_probabilities(),
        "positive_prob": positive_prob,
        "tense_prob": tense_prob,
        "score": None if score is None else int(round(max(0.0, min(100.0, score)))),
        "available": available,
        "status": status,
    }


def _model_path() -> Path:
    return EMOTION_MODEL_PATH.resolve()


def _load_model() -> Any | None:
    global _model, _model_load_attempted, _model_load_error

    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    _model_load_attempted = True
    model_path = _model_path()
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


def _decode_image(file_bytes: bytes) -> np.ndarray | None:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    if image_array.size == 0:
        return None
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _detect_face_crop(frame_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.20)
    frame_height, frame_width = frame_bgr.shape[:2]
    x_min = max(0, x - margin_x)
    y_min = max(0, y - margin_y)
    x_max = min(frame_width, x + w + margin_x)
    y_max = min(frame_height, y + h + margin_y)
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


def _predict(face_bgr: np.ndarray, model: Any) -> dict[str, Any]:
    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) < 4:
        return _build_result(status="model_not_loaded")

    img_height = input_shape[1]
    img_width = input_shape[2]
    channels = input_shape[3]
    if img_height is None or img_width is None or channels not in (1, 3):
        return _build_result(status="model_not_loaded")

    face_resized = cv2.resize(face_bgr, (int(img_width), int(img_height)))
    if int(channels) == 1:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_resized = np.expand_dims(face_resized, axis=-1)
    else:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    input_data = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
    try:
        predictions = model.predict(input_data, verbose=0)
    except Exception:
        return _build_result(status="prediction_error")

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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "interviewai-emotion-service",
        "model_path": str(_model_path()),
        "model_loaded": _model is not None,
        "model_load_error": _model_load_error,
    }


@app.post("/predict-emotion")
async def predict_emotion(
    image: UploadFile = File(...),
    is_face_crop: bool = Form(default=False),
) -> dict[str, Any]:
    file_bytes = await image.read()
    frame_bgr = _decode_image(file_bytes)
    if frame_bgr is None or frame_bgr.size == 0:
        return _build_result(status="face_not_detected")

    model = _load_model()
    if model is None:
        return _build_result(status="model_not_loaded")

    face_bgr = frame_bgr if is_face_crop else _detect_face_crop(frame_bgr)
    if face_bgr is None or face_bgr.size == 0:
        return _build_result(status="face_not_detected")

    return _predict(face_bgr, model)
