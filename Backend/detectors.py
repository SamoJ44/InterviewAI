import mediapipe as mp

from utils import _distance, _landmark_to_pixel


def _compute_eye_ear(landmarks, indices: list[int], frame_width: int, frame_height: int) -> float:
    p1 = _landmark_to_pixel(landmarks, indices[0], frame_width, frame_height)
    p2 = _landmark_to_pixel(landmarks, indices[1], frame_width, frame_height)
    p3 = _landmark_to_pixel(landmarks, indices[2], frame_width, frame_height)
    p4 = _landmark_to_pixel(landmarks, indices[3], frame_width, frame_height)
    p5 = _landmark_to_pixel(landmarks, indices[4], frame_width, frame_height)
    p6 = _landmark_to_pixel(landmarks, indices[5], frame_width, frame_height)

    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = max(2.0 * _distance(p1, p4), 1e-6)
    return vertical / horizontal


def compute_eye_open_factor(landmarks, frame_width: int, frame_height: int) -> tuple[float, str]:
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    left_ear = _compute_eye_ear(landmarks, left_eye_indices, frame_width, frame_height)
    right_ear = _compute_eye_ear(landmarks, right_eye_indices, frame_width, frame_height)
    avg_ear = (left_ear + right_ear) * 0.5

    if avg_ear < 0.15:
        return 0.0, "CLOSED"
    if avg_ear < 0.2:
        return 0.5, "PARTIAL"
    return 1.0, "OPEN"


def compute_eye_score(landmarks, frame_width: int, frame_height: int) -> float:
    left_iris_x, left_iris_y = _landmark_to_pixel(landmarks, 468, frame_width, frame_height)
    right_iris_x, right_iris_y = _landmark_to_pixel(landmarks, 473, frame_width, frame_height)

    left_outer_x, left_outer_y = _landmark_to_pixel(landmarks, 33, frame_width, frame_height)
    left_inner_x, left_inner_y = _landmark_to_pixel(landmarks, 133, frame_width, frame_height)
    right_inner_x, right_inner_y = _landmark_to_pixel(landmarks, 362, frame_width, frame_height)
    right_outer_x, right_outer_y = _landmark_to_pixel(landmarks, 263, frame_width, frame_height)

    left_upper_y = _landmark_to_pixel(landmarks, 159, frame_width, frame_height)[1]
    left_lower_y = _landmark_to_pixel(landmarks, 145, frame_width, frame_height)[1]
    right_upper_y = _landmark_to_pixel(landmarks, 386, frame_width, frame_height)[1]
    right_lower_y = _landmark_to_pixel(landmarks, 374, frame_width, frame_height)[1]

    left_center_x = (left_outer_x + left_inner_x) * 0.5
    right_center_x = (right_inner_x + right_outer_x) * 0.5

    left_center_y = (left_upper_y + left_lower_y) * 0.5
    right_center_y = (right_upper_y + right_lower_y) * 0.5

    left_eye_width = max(_distance((left_outer_x, left_outer_y), (left_inner_x, left_inner_y)), 1e-6)
    right_eye_width = max(_distance((right_inner_x, right_inner_y), (right_outer_x, right_outer_y)), 1e-6)

    left_eye_height = max(abs(left_upper_y - left_lower_y), 1e-6)
    right_eye_height = max(abs(right_upper_y - right_lower_y), 1e-6)

    left_eye_offset = abs(left_iris_x - left_center_x) / left_eye_width
    right_eye_offset = abs(right_iris_x - right_center_x) / right_eye_width
    eye_offset = (left_eye_offset + right_eye_offset) * 0.5

    left_vertical_offset = abs(left_iris_y - left_center_y) / left_eye_height
    right_vertical_offset = abs(right_iris_y - right_center_y) / right_eye_height
    vertical_offset = (left_vertical_offset + right_vertical_offset) * 0.5
    if vertical_offset > 0.30:
        return 0.0

    if eye_offset > 0.20:
        return 0.0
    return max(0.0, 1.0 - eye_offset * 5.0)


def compute_face_score(landmarks, frame_width: int, frame_height: int) -> float:
    left_outer_x, _ = _landmark_to_pixel(landmarks, 33, frame_width, frame_height)
    left_inner_x, _ = _landmark_to_pixel(landmarks, 133, frame_width, frame_height)
    right_inner_x, _ = _landmark_to_pixel(landmarks, 362, frame_width, frame_height)
    right_outer_x, _ = _landmark_to_pixel(landmarks, 263, frame_width, frame_height)
    nose_x, _ = _landmark_to_pixel(landmarks, 1, frame_width, frame_height)

    left_eye_center_x = (left_outer_x + left_inner_x) * 0.5
    right_eye_center_x = (right_inner_x + right_outer_x) * 0.5

    face_center_x = (left_eye_center_x + right_eye_center_x) * 0.5
    face_offset = abs(nose_x - face_center_x) / max(float(frame_width), 1e-6)

    return max(0.0, min(1.0, 1.0 - face_offset * 6.0))


def smooth_score(previous_score: float | None, current_score: float, alpha: float = 0.8) -> float:
    if previous_score is None:
        return current_score
    return alpha * previous_score + (1.0 - alpha) * current_score


def compute_posture_score(
    pose_landmarks,
    frame_width: int,
    frame_height: int,
    baseline_nose_y: float,
    baseline_shoulder_mid_y: float,
    baseline_shoulder_width: float,
) -> int:
    landmarks = pose_landmarks.landmark
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]

    left_shoulder_x = left_shoulder.x * frame_width
    right_shoulder_x = right_shoulder.x * frame_width
    left_shoulder_y = left_shoulder.y * frame_height
    right_shoulder_y = right_shoulder.y * frame_height
    nose_y = nose.y * frame_height
    current_shoulder_mid_y = (left_shoulder_y + right_shoulder_y) * 0.5
    current_shoulder_width = abs(right_shoulder_x - left_shoulder_x) + 1e-6
    baseline_shoulder_width = max(baseline_shoulder_width, 1e-6)

    head_drop = ((nose_y - baseline_nose_y) / baseline_shoulder_width) * 4.0
    shoulder_tilt = abs(current_shoulder_mid_y - baseline_shoulder_mid_y) / baseline_shoulder_width
    openness_ratio = current_shoulder_width / baseline_shoulder_width

    head_deviation = abs(head_drop)
    if head_deviation < 0.02:
        head_score = 1.0
    else:
        head_score = max(0.0, 1.0 - head_deviation * 3.0)
    head_score = max(0.0, min(1.0, head_score))
    shoulder_score = max(0.0, 1.0 - shoulder_tilt * 5.0)
    open_score = min(1.0, openness_ratio)
    posture_score = (
        0.4 * head_score
        + 0.4 * shoulder_score
        + 0.2 * open_score
    ) * 100.0
    posture_score = int(round(posture_score))

    return posture_score
