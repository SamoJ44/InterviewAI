import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "best_resnet50v2_finetuned.keras"
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
MIRROR_CAMERA_PREVIEW = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time emotion detection test")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_RELATIVE_PATH),
        help="Path to .keras model file. Relative paths are resolved from Backend/.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0).",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    backend_dir = Path(__file__).resolve().parent
    candidate = Path(model_arg).expanduser()
    if not candidate.is_absolute():
        candidate = backend_dir / candidate
    return candidate.resolve()


def load_model_or_exit(model_path: Path):
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"Error: failed to load model from {model_path}")
        print(f"Reason: {exc}")
        sys.exit(1)

    return model


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model_display_text = f"Model: {model_path.name}"

    print(f"Loading model: {model_path}")
    model = load_model_or_exit(model_path)

    input_shape = model.input_shape
    img_height = input_shape[1]
    img_width = input_shape[2]
    channels = input_shape[3]

    if img_height is None or img_width is None or channels is None:
        print(f"Error: unsupported model input shape: {input_shape}")
        sys.exit(1)

    img_height = int(img_height)
    img_width = int(img_width)
    channels = int(channels)

    print(f"Model input shape: {input_shape}")
    print(f"Expected input image size: {img_width}x{img_height}, channels={channels}")
    print(f"Class labels: {CLASS_NAMES}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"Trying camera index: {args.camera_index}")
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("Camera opened: False")
        print(f"Error: could not open camera (index {args.camera_index}).")
        sys.exit(1)

    print("Camera opened: True")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from camera.")
            break

        if MIRROR_CAMERA_PREVIEW:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            face_resized = cv2.resize(face, (img_width, img_height))

            if channels == 1:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_resized = np.expand_dims(face_resized, axis=-1)
            elif channels == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            else:
                print(f"Error: unsupported channel count in model input: {channels}")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

            face_resized = face_resized.astype("float32") / 255.0
            input_data = np.expand_dims(face_resized, axis=0)

            predictions = model.predict(input_data, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])

            if predicted_index >= len(CLASS_NAMES):
                label_text = f"class_{predicted_index}: {confidence:.2f}"
            else:
                label_text = f"{CLASS_NAMES[predicted_index]}: {confidence:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        model_font = cv2.FONT_HERSHEY_SIMPLEX
        model_font_scale = 0.65
        model_font_thickness = 2
        text_size, baseline = cv2.getTextSize(
            model_display_text,
            model_font,
            model_font_scale,
            model_font_thickness,
        )
        text_x = 12
        text_y = 30
        padding = 8
        box_top_left = (text_x - padding, text_y - text_size[1] - padding)
        box_bottom_right = (text_x + text_size[0] + padding, text_y + baseline + padding)
        cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 0, 0), -1)
        cv2.putText(
            frame,
            model_display_text,
            (text_x, text_y),
            model_font,
            model_font_scale,
            (255, 255, 255),
            model_font_thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Real-Time Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
