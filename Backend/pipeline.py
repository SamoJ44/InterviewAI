import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import cv2
import mediapipe as mp

from config import (
    CALIBRATION_DURATION_SECONDS,
    EMOTION_SMOOTH_ALPHA,
    FINAL_WEIGHT_EYE,
    FINAL_WEIGHT_POSTURE,
    FINAL_WEIGHT_SELF_TOUCH,
    FINAL_WEIGHT_STABILITY,
    MOVEMENT_WINDOW_SIZE,
    POSE_VISIBILITY_THRESHOLD,
    PRE_CALIBRATION_COUNTDOWN_SECONDS,
    SELF_TOUCH_ACTIVE_FRAMES,
    SELF_TOUCH_FRAME_PENALTY,
    SMOOTH_ALPHA,
    STABILITY_DIRECTION_CHANGE_PENALTY,
    STABILITY_DROP_SMOOTH_ALPHA,
    STABILITY_HORIZONTAL_SWAY_WEIGHT,
    STABILITY_MOTION_DEADZONE,
    STABILITY_MOTION_MAX,
    STABILITY_RECOVERY_BOOST,
    STABILITY_RECOVERY_SMOOTH_ALPHA,
    STABILITY_STILLNESS_RECOVERY_FRAMES,
    STABILITY_STILLNESS_THRESHOLD,
    STABILITY_SWAY_WINDOW_SIZE,
)
from detectors import (
    compute_eye_open_factor,
    compute_eye_score,
    compute_face_score,
    compute_posture_score,
    smooth_score,
)
from emotion_detector import build_raw_emotion_result, detect_expression
from touch_logic import compute_body_targets, detect_potential_touch, extract_face_touch_targets
from utils import _distance
from visualization import draw_face_landmarks, draw_overlay, draw_pose_landmarks

GAZE_AWAY_EYE_CONTACT_THRESHOLD = 35
POSTURE_DEVIATION_THRESHOLD = 60
GAZE_AWAY_PROLONGED_SECONDS = 3.0
POSTURE_SUSTAINED_SECONDS = 4.0
TRACKING_LOST_ALERT_SECONDS = 4.0
SELF_TOUCH_REPEAT_WINDOW_SECONDS = 20.0
SELF_TOUCH_REPEAT_COUNT = 3


@dataclass
class PipelineState:
    previous_eye_contact_score: float | None = None
    previous_expression_score: float | None = None
    pre_calibration_countdown_seconds: float = PRE_CALIBRATION_COUNTDOWN_SECONDS
    calibration_duration_seconds: float = CALIBRATION_DURATION_SECONDS
    calibration_start_time: float = field(default_factory=time.time)
    baseline_nose_y: float | None = None
    baseline_shoulder_mid_y: float | None = None
    baseline_shoulder_width: float | None = None
    baseline_nose_sum: float = 0.0
    baseline_shoulder_mid_sum: float = 0.0
    baseline_shoulder_width_sum: float = 0.0
    calibration_samples: int = 0
    previous_nose_position: tuple[float, float] | None = None
    previous_shoulder_mid_position: tuple[float, float] | None = None
    previous_motion_time: float | None = None
    previous_stability_score: float | None = None
    movement_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=MOVEMENT_WINDOW_SIZE),
    )
    sway_direction_window: deque[int] = field(
        default_factory=lambda: deque(maxlen=STABILITY_SWAY_WINDOW_SIZE),
    )
    stability_still_frames: int = 0
    stability_debug_counter: int = 0
    touch_frames: int = 0
    session_start_time: float = field(default_factory=time.time)
    event_counter: int = 0
    alert_counter: int = 0
    prev_self_touch_active: bool = False
    prev_gaze_away_active: bool = False
    prev_posture_deviation_active: bool = False
    prev_face_detected: bool = False
    prev_pose_detected: bool = False
    gaze_away_start_time: float | None = None
    posture_deviation_start_time: float | None = None
    face_lost_start_time: float | None = None
    pose_lost_start_time: float | None = None
    gaze_away_prolonged_emitted: bool = False
    posture_sustained_alert_emitted: bool = False
    face_tracking_alert_emitted: bool = False
    pose_tracking_alert_emitted: bool = False
    self_touch_start_times: deque[float] = field(default_factory=deque)
    last_alert_time_by_key: dict[str, float] = field(default_factory=dict)


@dataclass
class Processors:
    drawing_utils: Any
    drawing_styles: Any
    face_mesh_module: Any
    pose_module: Any
    face_mesh: Any
    pose: Any
    hands: Any


@dataclass(frozen=True)
class ScoreDetail:
    value: float | int | None
    available: bool
    status: str
    reason: str | None = None


def build_score_detail(
    value: float | int | None,
    available: bool,
    status: str,
    reason: str | None = None,
) -> dict[str, float | int | bool | str | None]:
    return {
        "value": value,
        "available": available,
        "status": status,
        "reason": reason,
    }


def calculate_final_score_detail(
    component_details: dict[str, dict[str, float | int | bool | str | None]],
) -> dict[str, float | int | bool | str | None]:
    weights = {
        "eye_contact": FINAL_WEIGHT_EYE,
        "posture": FINAL_WEIGHT_POSTURE,
        "stability": FINAL_WEIGHT_STABILITY,
        "self_touch": FINAL_WEIGHT_SELF_TOUCH,
    }
    weighted_score_sum = 0.0
    available_weight_sum = 0.0
    total_configured_weight = sum(weights.values())

    # A missing detector signal is not the same thing as bad behavior.
    # Unavailable components are excluded from the normalized final score.
    for score_name, weight in weights.items():
        detail = component_details[score_name]
        value = detail["value"]
        if detail["available"] is True and value is not None:
            weighted_score_sum += float(value) * weight
            available_weight_sum += weight

    if available_weight_sum <= 0.0:
        return {
            **build_score_detail(
                None,
                False,
                "unavailable",
                "No component scores are currently available.",
            ),
            "reliability": 0.0,
        }

    final_score = weighted_score_sum / available_weight_sum
    final_score = max(0.0, min(100.0, final_score))
    status = "ok" if available_weight_sum >= total_configured_weight else "partial"
    return {
        **build_score_detail(final_score, True, status),
        "reliability": available_weight_sum / total_configured_weight if status == "partial" else 1.0,
    }


def is_pose_landmark_visible(
    landmark,
    threshold: float = POSE_VISIBILITY_THRESHOLD,
) -> bool:
    return landmark is not None and getattr(landmark, "visibility", 0.0) >= threshold


def has_required_pose_landmarks(pose_landmarks) -> bool:
    if pose_landmarks is None:
        return False

    landmarks = pose_landmarks.landmark
    required_indices = (
        mp.solutions.pose.PoseLandmark.NOSE.value,
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    )
    for index in required_indices:
        if index >= len(landmarks) or not is_pose_landmark_visible(landmarks[index]):
            return False
    return True


def init_state() -> PipelineState:
    return PipelineState()


@contextmanager
def create_processors() -> Iterator[Processors]:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        yield Processors(
            drawing_utils=mp_drawing,
            drawing_styles=mp_drawing_styles,
            face_mesh_module=mp_face_mesh,
            pose_module=mp_pose,
            face_mesh=face_mesh,
            pose=pose,
            hands=hands,
        )


def _format_elapsed(elapsed_seconds: float) -> str:
    elapsed = max(0, int(round(elapsed_seconds)))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_event(
    state: PipelineState,
    now: float,
    event_type: str,
    severity: str,
    description: str,
) -> dict[str, str]:
    state.event_counter += 1
    return {
        "id": f"evt-{state.event_counter:06d}",
        "timestamp": _format_elapsed(now - state.session_start_time),
        "type": event_type,
        "severity": severity,
        "description": description,
    }


def _build_alert(
    state: PipelineState,
    now: float,
    key: str,
    severity: str,
    title: str,
    description: str,
    category: str,
    cooldown_seconds: float,
) -> dict[str, str] | None:
    last_time = state.last_alert_time_by_key.get(key)
    if last_time is not None and now - last_time < cooldown_seconds:
        return None

    state.alert_counter += 1
    state.last_alert_time_by_key[key] = now
    return {
        "id": f"alr-{state.alert_counter:06d}",
        "time": _format_elapsed(now - state.session_start_time),
        "severity": severity,
        "title": title,
        "description": description,
        "category": category,
    }


def process_frame(
    frame,
    processors: Processors,
    state: PipelineState,
    draw: bool = False,
) -> dict[str, Any]:
    state.stability_debug_counter += 1
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    face_results = processors.face_mesh.process(rgb_frame)
    pose_results = processors.pose.process(rgb_frame)
    hands_results = processors.hands.process(rgb_frame)

    rgb_frame.flags.writeable = True
    current_eye_contact_score: float | None = None
    eye_status_text = "NO_FACE"
    posture_score = 0
    stability_score = 0
    normalized_motion_rate: float | None = None
    average_motion_rate = 0.0
    effective_motion = 0.0
    raw_stability_score = 0.0
    smoothed_stability_score = 0.0
    delta_time: float | None = None
    nose_movement_pixels: float | None = None
    shoulder_midpoint_movement_pixels: float | None = None
    shoulder_width: float | None = None
    horizontal_sway_rate: float | None = None
    motion_signal: float | None = None
    direction_changes = 0
    sway_penalty = 0
    previous_stability_score = state.previous_stability_score
    stability_smoothing_alpha: float | None = None
    self_touch_score = 100
    frame_height, frame_width = frame.shape[:2]
    central_face_targets = []
    side_face_targets = []
    body_targets = []
    face_nose_point = None
    face_reference_z = None
    face_chin_point = None
    left_face_edge_point = None
    right_face_edge_point = None
    primary_face_landmarks = None

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            primary_face_landmarks = face_landmarks
            if draw:
                draw_face_landmarks(
                    frame,
                    face_landmarks,
                    processors.drawing_utils,
                    processors.drawing_styles,
                    processors.face_mesh_module,
                )

            landmarks = face_landmarks.landmark
            (
                face_nose_point,
                face_reference_z,
                face_chin_point,
                left_face_edge_point,
                right_face_edge_point,
                central_face_targets,
                side_face_targets,
            ) = extract_face_touch_targets(landmarks, frame_width, frame_height)
            eye_score = compute_eye_score(landmarks, frame_width, frame_height)
            face_score = compute_face_score(landmarks, frame_width, frame_height)
            eye_factor, eye_state = compute_eye_open_factor(
                landmarks,
                frame_width,
                frame_height,
            )
            if eye_state == "CLOSED":
                eye_score = 0.0
                face_score *= 0.3
            elif eye_state == "PARTIAL":
                eye_score *= 0.5
            else:
                eye_score *= eye_factor

            final_eye_score = (0.5 * face_score + 0.5 * eye_score) * 100.0
            current_eye_contact_score = max(0.0, min(100.0, final_eye_score))
            eye_status_text = eye_state
            break

    if pose_results.pose_landmarks:
        body_targets = compute_body_targets(
            pose_results.pose_landmarks.landmark,
            frame_width,
            frame_height,
        )

    potential_touch = False
    current_touch_details = {
        "detected": False,
        "region": None,
        "hand": None,
        "confidence": 0.0,
    }
    if (
        face_nose_point is not None
        and face_reference_z is not None
        and face_chin_point is not None
        and (central_face_targets or side_face_targets)
        and left_face_edge_point is not None
        and right_face_edge_point is not None
        and hands_results.multi_hand_landmarks
    ):
        current_touch_details = detect_potential_touch(
            hands_results,
            frame_width,
            frame_height,
            face_reference_z,
            left_face_edge_point,
            right_face_edge_point,
            central_face_targets,
            side_face_targets,
            body_targets,
            return_details=True,
        )
        potential_touch = bool(current_touch_details["detected"])

    if potential_touch:
        state.touch_frames += 1
    else:
        # Decay instead of hard reset to reduce flicker and avoid losing
        # activation progress on short single-frame misses.
        state.touch_frames = max(0, state.touch_frames - 1)

    self_touch_active = state.touch_frames >= SELF_TOUCH_ACTIVE_FRAMES
    self_touch_score = max(0, min(100, 100 - state.touch_frames * SELF_TOUCH_FRAME_PENALTY))
    self_touch_details = {
        "active": self_touch_active,
        "region": current_touch_details["region"] if potential_touch else None,
        "hand": current_touch_details["hand"] if potential_touch else None,
        "confidence": current_touch_details["confidence"] if potential_touch else 0.0,
        "touch_frames": state.touch_frames,
    }

    face_detected = bool(face_results.multi_face_landmarks)
    pose_detected = bool(pose_results.pose_landmarks)
    pose_available = has_required_pose_landmarks(pose_results.pose_landmarks)
    pose_unavailable_reason = None
    if not pose_detected:
        pose_unavailable_reason = "pose_not_detected"
    elif not pose_available:
        pose_unavailable_reason = "pose_low_visibility"

    if current_eye_contact_score is None:
        state.previous_eye_contact_score = None
        eye_contact_score = 0
    else:
        smoothed_eye_contact_score = smooth_score(
            state.previous_eye_contact_score,
            current_eye_contact_score,
            alpha=SMOOTH_ALPHA,
        )
        state.previous_eye_contact_score = smoothed_eye_contact_score
        eye_contact_score = int(round(smoothed_eye_contact_score))

    emotion_result = detect_expression(frame, primary_face_landmarks)
    raw_emotion_result = build_raw_emotion_result(emotion_result)
    emotion_score_value = emotion_result["score"]
    has_expression_score = isinstance(emotion_score_value, (int, float))
    if has_expression_score:
        smoothed_emotion_score = smooth_score(
            state.previous_expression_score,
            float(emotion_score_value),
            alpha=EMOTION_SMOOTH_ALPHA,
        )
        state.previous_expression_score = smoothed_emotion_score
        emotion_result["score"] = int(round(max(0.0, min(100.0, smoothed_emotion_score))))
    else:
        state.previous_expression_score = None

    now_for_calibration = time.time()
    elapsed_since_session_start = now_for_calibration - state.session_start_time
    preparation_duration = max(state.pre_calibration_countdown_seconds, 1e-6)
    calibration_duration = max(state.calibration_duration_seconds, 1e-6)

    if elapsed_since_session_start < state.pre_calibration_countdown_seconds:
        calibration_phase = "preparing"
        calibration_active = False
        calibration_progress = min(1.0, max(0.0, elapsed_since_session_start / preparation_duration))
        countdown_remaining = max(0.0, state.pre_calibration_countdown_seconds - elapsed_since_session_start)
    else:
        calibration_elapsed = elapsed_since_session_start - state.pre_calibration_countdown_seconds
        countdown_remaining = 0.0
        if calibration_elapsed < state.calibration_duration_seconds:
            calibration_phase = "calibrating"
            calibration_active = True
            calibration_progress = min(1.0, max(0.0, calibration_elapsed / calibration_duration))
        else:
            calibration_phase = "complete"
            calibration_active = False
            calibration_progress = 1.0

    if pose_results.pose_landmarks:
        if draw:
            draw_pose_landmarks(
                frame,
                pose_results.pose_landmarks,
                processors.drawing_utils,
                processors.drawing_styles,
                processors.pose_module,
            )
        pose_landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = pose_landmarks[processors.pose_module.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[processors.pose_module.PoseLandmark.RIGHT_SHOULDER.value]
        nose = pose_landmarks[processors.pose_module.PoseLandmark.NOSE.value]

        left_shoulder_x = left_shoulder.x * frame_width
        right_shoulder_x = right_shoulder.x * frame_width
        left_shoulder_y = left_shoulder.y * frame_height
        right_shoulder_y = right_shoulder.y * frame_height
        nose_x = nose.x * frame_width
        nose_y = nose.y * frame_height
        current_nose_position = (nose_x, nose_y)
        current_shoulder_mid_position = (
            (left_shoulder_x + right_shoulder_x) * 0.5,
            (left_shoulder_y + right_shoulder_y) * 0.5,
        )
        current_shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
        current_shoulder_width = abs(right_shoulder_x - left_shoulder_x) + 1e-6
        shoulder_width = current_shoulder_width

        if pose_available:
            movement_rate_pixels: float | None = None
            now_for_motion = time.time()
            normalization_width = (
                state.baseline_shoulder_width
                if state.baseline_shoulder_width is not None
                else current_shoulder_width
            )
            normalization_width = max(float(normalization_width), 1e-6)
            if (
                state.previous_nose_position is not None
                and state.previous_shoulder_mid_position is not None
                and state.previous_motion_time is not None
            ):
                nose_movement_pixels = _distance(current_nose_position, state.previous_nose_position)
                shoulder_midpoint_movement_pixels = _distance(
                    current_shoulder_mid_position,
                    state.previous_shoulder_mid_position,
                )
                delta_time = max(now_for_motion - state.previous_motion_time, 1e-3)
                movement_pixels = nose_movement_pixels + shoulder_midpoint_movement_pixels
                movement_rate_pixels = movement_pixels / delta_time

                nose_dx_norm = abs(nose_x - state.previous_nose_position[0]) / normalization_width
                shoulder_dx = current_shoulder_mid_position[0] - state.previous_shoulder_mid_position[0]
                shoulder_dx_norm = abs(shoulder_dx) / normalization_width
                horizontal_sway_rate = (nose_dx_norm + shoulder_dx_norm) / delta_time

                sway_step_threshold = STABILITY_MOTION_DEADZONE
                if shoulder_dx_norm > sway_step_threshold:
                    state.sway_direction_window.append(1 if shoulder_dx > 0 else -1)

            state.previous_nose_position = current_nose_position
            state.previous_shoulder_mid_position = current_shoulder_mid_position
            state.previous_motion_time = now_for_motion

            if movement_rate_pixels is not None:
                normalized_motion_rate = movement_rate_pixels / normalization_width
                weighted_horizontal_sway = (
                    (horizontal_sway_rate or 0.0) * STABILITY_HORIZONTAL_SWAY_WEIGHT
                )
                motion_signal = max(normalized_motion_rate, weighted_horizontal_sway)
                state.movement_window.append(motion_signal)

                if motion_signal < STABILITY_STILLNESS_THRESHOLD:
                    state.stability_still_frames += 1
                else:
                    state.stability_still_frames = 0

                if state.stability_still_frames >= STABILITY_STILLNESS_RECOVERY_FRAMES:
                    state.sway_direction_window.clear()
                    state.movement_window.append(0.0)
            else:
                state.stability_still_frames = 0

            directions = list(state.sway_direction_window)
            direction_changes = sum(
                1
                for previous_direction, current_direction in zip(directions, directions[1:])
                if previous_direction != current_direction
            )
            average_motion_rate = (
                sum(state.movement_window) / len(state.movement_window)
                if state.movement_window
                else 0.0
            )
            effective_motion = max(0.0, average_motion_rate - STABILITY_MOTION_DEADZONE)
            motion_span = max(STABILITY_MOTION_MAX - STABILITY_MOTION_DEADZONE, 1e-6)
            motion_penalty = min(1.0, effective_motion / motion_span)
            raw_stability_score = (1.0 - motion_penalty) * 100.0
            if (
                state.stability_still_frames >= STABILITY_STILLNESS_RECOVERY_FRAMES
                and state.previous_stability_score is not None
            ):
                raw_stability_score = max(
                    raw_stability_score,
                    state.previous_stability_score + STABILITY_RECOVERY_BOOST,
                )
                raw_stability_score = min(raw_stability_score, 100.0)

            stability_smoothing_alpha = (
                STABILITY_DROP_SMOOTH_ALPHA
                if (
                    state.previous_stability_score is not None
                    and raw_stability_score < state.previous_stability_score
                )
                else STABILITY_RECOVERY_SMOOTH_ALPHA
            )
            smoothed_stability_score = smooth_score(
                state.previous_stability_score,
                raw_stability_score,
                alpha=stability_smoothing_alpha,
            )
            sway_penalty = direction_changes * STABILITY_DIRECTION_CHANGE_PENALTY
            penalized_stability_score = max(
                0.0,
                min(100.0, smoothed_stability_score - sway_penalty),
            )
            state.previous_stability_score = penalized_stability_score
            stability_score = int(round(penalized_stability_score))
        else:
            state.previous_nose_position = None
            state.previous_shoulder_mid_position = None
            state.previous_motion_time = None
            state.previous_stability_score = None
            state.movement_window.clear()
            state.sway_direction_window.clear()
            state.stability_still_frames = 0

        if calibration_active:
            if pose_available and stability_score > 80:
                state.baseline_nose_sum += nose_y
                state.baseline_shoulder_mid_sum += current_shoulder_mid_y
                state.baseline_shoulder_width_sum += current_shoulder_width
                state.calibration_samples += 1

                state.baseline_nose_y = state.baseline_nose_sum / state.calibration_samples
                state.baseline_shoulder_mid_y = state.baseline_shoulder_mid_sum / state.calibration_samples
                state.baseline_shoulder_width = state.baseline_shoulder_width_sum / state.calibration_samples
            posture_score = 0
        else:
            if (
                state.baseline_nose_y is not None
                and state.baseline_shoulder_mid_y is not None
                and state.baseline_shoulder_width is not None
            ):
                posture_score = compute_posture_score(
                    pose_results.pose_landmarks,
                    frame_width,
                    frame_height,
                    state.baseline_nose_y,
                    state.baseline_shoulder_mid_y,
                    state.baseline_shoulder_width,
                )
            else:
                posture_score = 0
    else:
        state.previous_nose_position = None
        state.previous_shoulder_mid_position = None
        state.previous_motion_time = None
        state.previous_stability_score = None
        state.movement_window.clear()
        state.sway_direction_window.clear()
        state.stability_still_frames = 0

    calibration_ready = (
        state.calibration_samples > 0
        and state.baseline_nose_y is not None
        and state.baseline_shoulder_mid_y is not None
        and state.baseline_shoulder_width is not None
    )
    if calibration_phase == "preparing":
        calibration_status = "preparing"
    elif calibration_active:
        calibration_status = "calibrating"
    elif calibration_ready:
        calibration_status = "ready"
    else:
        calibration_status = "failed"

    if calibration_status == "preparing":
        # Preparing is not bad posture; the user is being given time to settle.
        posture_detail = build_score_detail(
            None,
            False,
            "preparing",
            "posture_preparation_active",
        )
    elif calibration_active:
        posture_detail = build_score_detail(
            None,
            False,
            "calibrating",
            "posture_calibration_active",
        )
    elif not calibration_ready:
        # Failed calibration means posture cannot be scored yet; it is not bad posture.
        posture_detail = build_score_detail(
            None,
            False,
            "calibration_failed",
            "no_valid_calibration_samples",
        )
    else:
        posture_detail = build_score_detail(posture_score, True, "ok")

    calibration_messages = {
        "preparing": "Sit straight and face the camera. Calibration starts soon.",
        "calibrating": "Hold still. Calibrating your posture baseline.",
        "ready": "Posture calibration ready.",
        "failed": "Calibration failed. Sit straight and stay visible, then restart calibration.",
    }
    calibration = {
        "active": calibration_active,
        "ready": calibration_ready,
        "progress": calibration_progress,
        "countdown_remaining": countdown_remaining,
        "samples": state.calibration_samples,
        "status": calibration_status,
        "message": calibration_messages[calibration_status],
    }

    score_details = {
        "eye_contact": build_score_detail(
            eye_contact_score if face_detected else None,
            face_detected,
            "ok" if face_detected else "no_face",
            None if face_detected else "face_not_detected",
        ),
        # Calibration is not bad posture; it means posture is not scoreable yet.
        "posture": posture_detail,
        "stability": build_score_detail(
            stability_score if pose_available else None,
            pose_available,
            "ok" if pose_available else "unavailable",
            pose_unavailable_reason,
        ),
        "self_touch": build_score_detail(self_touch_score, True, "ok"),
        "expression": build_score_detail(
            emotion_result["score"] if isinstance(emotion_result["score"], (int, float)) else None,
            isinstance(emotion_result["score"], (int, float)),
            emotion_result["status"],
            None if isinstance(emotion_result["score"], (int, float)) else emotion_result["status"],
        ),
    }
    # Future: optionally include expression score in final score with a small weight around 7-10%.
    final_score_detail = calculate_final_score_detail(score_details)
    score_details["final"] = final_score_detail
    final_score = final_score_detail["value"] if final_score_detail["value"] is not None else 0.0

    hands_detected = bool(hands_results.multi_hand_landmarks)
    gaze_away_active = (
        face_detected
        and eye_status_text == "OPEN"
        and eye_contact_score < GAZE_AWAY_EYE_CONTACT_THRESHOLD
    )
    posture_deviation_active = (
        pose_detected
        and score_details["posture"]["available"] is True
        and posture_score < POSTURE_DEVIATION_THRESHOLD
    )

    now = time.time()
    events: list[dict[str, str]] = []
    alerts: list[dict[str, str]] = []

    if self_touch_active and not state.prev_self_touch_active:
        events.append(
            _build_event(
                state,
                now,
                "self_touch_started",
                "medium",
                "Self-touch behavior started.",
            ),
        )
        state.self_touch_start_times.append(now)
        while state.self_touch_start_times and (
            now - state.self_touch_start_times[0] > SELF_TOUCH_REPEAT_WINDOW_SECONDS
        ):
            state.self_touch_start_times.popleft()

        if len(state.self_touch_start_times) >= SELF_TOUCH_REPEAT_COUNT:
            repeated_touch_alert = _build_alert(
                state,
                now,
                "self_touch_repeated",
                "high",
                "Repeated self-touch detected",
                "Multiple self-touch starts detected in a short period.",
                "self_touch",
                cooldown_seconds=SELF_TOUCH_REPEAT_WINDOW_SECONDS,
            )
            if repeated_touch_alert is not None:
                alerts.append(repeated_touch_alert)

    if not self_touch_active and state.prev_self_touch_active:
        events.append(
            _build_event(
                state,
                now,
                "self_touch_ended",
                "positive",
                "Self-touch behavior ended.",
            ),
        )

    if gaze_away_active:
        if state.gaze_away_start_time is None:
            state.gaze_away_start_time = now
            events.append(
                _build_event(
                    state,
                    now,
                    "gaze_away_started",
                    "medium",
                    "Gaze away from camera started.",
                ),
            )
        gaze_duration = now - state.gaze_away_start_time
        if gaze_duration >= GAZE_AWAY_PROLONGED_SECONDS and not state.gaze_away_prolonged_emitted:
            events.append(
                _build_event(
                    state,
                    now,
                    "gaze_away_prolonged",
                    "high",
                    "Gaze away persisted beyond the threshold.",
                ),
            )
            prolonged_gaze_alert = _build_alert(
                state,
                now,
                "gaze_away_prolonged",
                "high",
                "Prolonged gaze away",
                "Eye contact dropped for an extended period.",
                "eye_contact",
                cooldown_seconds=15.0,
            )
            if prolonged_gaze_alert is not None:
                alerts.append(prolonged_gaze_alert)
            state.gaze_away_prolonged_emitted = True
    else:
        state.gaze_away_start_time = None
        state.gaze_away_prolonged_emitted = False

    if posture_deviation_active:
        if state.posture_deviation_start_time is None:
            state.posture_deviation_start_time = now
            events.append(
                _build_event(
                    state,
                    now,
                    "posture_deviation_started",
                    "medium",
                    "Posture deviation started.",
                ),
            )
        posture_duration = now - state.posture_deviation_start_time
        if posture_duration >= POSTURE_SUSTAINED_SECONDS and not state.posture_sustained_alert_emitted:
            posture_alert = _build_alert(
                state,
                now,
                "posture_sustained",
                "high",
                "Sustained posture deviation",
                "Posture score remained below threshold for several seconds.",
                "posture",
                cooldown_seconds=15.0,
            )
            if posture_alert is not None:
                alerts.append(posture_alert)
            state.posture_sustained_alert_emitted = True
    else:
        if state.prev_posture_deviation_active:
            events.append(
                _build_event(
                    state,
                    now,
                    "posture_deviation_recovered",
                    "positive",
                    "Posture recovered above threshold.",
                ),
            )
        state.posture_deviation_start_time = None
        state.posture_sustained_alert_emitted = False

    if not face_detected:
        if state.face_lost_start_time is None:
            state.face_lost_start_time = now
            if state.prev_face_detected:
                events.append(
                    _build_event(
                        state,
                        now,
                        "face_lost",
                        "high",
                        "Face tracking was lost.",
                    ),
                )
        if (
            now - state.face_lost_start_time >= TRACKING_LOST_ALERT_SECONDS
            and not state.face_tracking_alert_emitted
        ):
            face_tracking_alert = _build_alert(
                state,
                now,
                "tracking_face_lost",
                "high",
                "Face tracking lost",
                "Face tracking has been unavailable for too long.",
                "tracking",
                cooldown_seconds=20.0,
            )
            if face_tracking_alert is not None:
                alerts.append(face_tracking_alert)
            state.face_tracking_alert_emitted = True
    else:
        if not state.prev_face_detected and state.face_lost_start_time is not None:
            events.append(
                _build_event(
                    state,
                    now,
                    "face_reacquired",
                    "positive",
                    "Face tracking was reacquired.",
                ),
            )
        state.face_lost_start_time = None
        state.face_tracking_alert_emitted = False

    if not pose_detected:
        if state.pose_lost_start_time is None:
            state.pose_lost_start_time = now
        if (
            now - state.pose_lost_start_time >= TRACKING_LOST_ALERT_SECONDS
            and not state.pose_tracking_alert_emitted
        ):
            pose_tracking_alert = _build_alert(
                state,
                now,
                "tracking_pose_lost",
                "medium",
                "Pose tracking lost",
                "Pose tracking has been unavailable for too long.",
                "tracking",
                cooldown_seconds=20.0,
            )
            if pose_tracking_alert is not None:
                alerts.append(pose_tracking_alert)
            state.pose_tracking_alert_emitted = True
    else:
        state.pose_lost_start_time = None
        state.pose_tracking_alert_emitted = False

    state.prev_self_touch_active = self_touch_active
    state.prev_gaze_away_active = gaze_away_active
    state.prev_posture_deviation_active = posture_deviation_active
    state.prev_face_detected = face_detected
    state.prev_pose_detected = pose_detected

    if draw:
        draw_overlay(
            frame,
            eye_contact_score,
            eye_status_text,
            posture_score,
            stability_score,
            self_touch_active,
            final_score,
            calibration_active,
            calibration,
        )

    return {
        "frame": frame,
        "eye_contact_score": eye_contact_score,
        "eye_status_text": eye_status_text,
        "posture_score": posture_score,
        "stability_score": stability_score,
        "expression_score": emotion_result["score"] if isinstance(emotion_result["score"], (int, float)) else None,
        "emotion": emotion_result,
        "raw_emotion": raw_emotion_result,
        "stability_debug": {
            "debug_sequence": state.stability_debug_counter,
            "delta_time": delta_time,
            "nose_movement_pixels": nose_movement_pixels,
            "shoulder_midpoint_movement_pixels": shoulder_midpoint_movement_pixels,
            "shoulder_width": shoulder_width,
            "normalized_motion_rate": normalized_motion_rate,
            "horizontal_sway_rate": horizontal_sway_rate,
            "motion_signal": motion_signal,
            "average_motion_rate": average_motion_rate,
            "effective_motion": effective_motion,
            "raw_stability_score": raw_stability_score,
            "smoothed_stability_score": smoothed_stability_score,
            "penalized_stability_score": state.previous_stability_score,
            "previous_stability_score": previous_stability_score,
            "stability_smoothing_alpha": stability_smoothing_alpha,
            "direction_changes": direction_changes,
            "sway_penalty": sway_penalty,
            "still_frames": state.stability_still_frames,
            "movement_window_length": len(state.movement_window),
            "movement_window": list(state.movement_window),
            "pose_detected": pose_detected,
            "pose_available": pose_available,
            "calibration_active": calibration_active,
        },
        "self_touch_score": self_touch_score,
        "self_touch_active": self_touch_active,
        "self_touch_details": self_touch_details,
        "final_score": final_score,
        "score_details": score_details,
        "calibration": calibration,
        "calibration_active": calibration_active,
        "face_detected": face_detected,
        "pose_detected": pose_detected,
        "pose_available": pose_available,
        "hands_detected": hands_detected,
        "gaze_away_active": gaze_away_active,
        "events": events,
        "alerts": alerts,
    }
