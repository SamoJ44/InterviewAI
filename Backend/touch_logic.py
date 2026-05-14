import mediapipe as mp

from config import (
    BODY_TOUCH_SCALE,
    BODY_Z_THRESHOLD,
    CENTRAL_FACE_INDICES,
    FACE_SIDE_Z_THRESHOLD,
    FACE_TOUCH_SCALE,
    FACE_Z_THRESHOLD,
    KEY_HAND_POINTS,
    SIDE_FACE_INDICES,
)
from utils import _distance, _landmark_to_pixel


def _empty_touch_details() -> dict[str, bool | str | float | None]:
    return {
        "detected": False,
        "region": None,
        "hand": None,
        "confidence": 0.0,
    }


def _get_hand_label(hands_results, hand_index: int) -> str | None:
    handedness = getattr(hands_results, "multi_handedness", None)
    if not handedness or hand_index >= len(handedness):
        # TODO: Infer hand side from landmark geometry if MediaPipe handedness is unavailable.
        return None

    classifications = handedness[hand_index].classification
    if not classifications:
        return None

    label = classifications[0].label.lower()
    if label in {"left", "right"}:
        return label
    return None


def _touch_confidence(distance: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / threshold)))


def _touch_confidence_with_z(
    distance: float,
    distance_threshold: float,
    z_distance: float | None,
    z_threshold: float | None,
) -> float:
    distance_confidence = _touch_confidence(distance, distance_threshold)
    if z_distance is None or z_threshold is None or z_threshold <= 0.0:
        # TODO: Some MediaPipe depth values come from different models/coordinate
        # spaces. Keep 2D detection available for debugging, but lower confidence.
        return distance_confidence * 0.5

    z_confidence = _touch_confidence(z_distance, z_threshold)
    return max(0.0, min(1.0, (0.6 * distance_confidence) + (0.4 * z_confidence)))


def _landmark_to_pixel_with_z(
    landmarks,
    index: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    x, y = _landmark_to_pixel(landmarks, index, frame_width, frame_height)
    return x, y, landmarks[index].z


def extract_face_touch_targets(landmarks, frame_width: int, frame_height: int):
    face_nose_point = _landmark_to_pixel(landmarks, 1, frame_width, frame_height)
    face_reference_z = landmarks[1].z
    face_chin_point = _landmark_to_pixel(landmarks, 152, frame_width, frame_height)
    left_face_edge_point = _landmark_to_pixel(landmarks, 234, frame_width, frame_height)
    right_face_edge_point = _landmark_to_pixel(landmarks, 454, frame_width, frame_height)
    central_face_targets = [
        _landmark_to_pixel(landmarks, index, frame_width, frame_height)
        for index in CENTRAL_FACE_INDICES
    ]
    side_face_targets = [
        _landmark_to_pixel_with_z(landmarks, index, frame_width, frame_height)
        for index in SIDE_FACE_INDICES
    ]

    return (
        face_nose_point,
        face_reference_z,
        face_chin_point,
        left_face_edge_point,
        right_face_edge_point,
        central_face_targets,
        side_face_targets,
    )


def compute_body_targets(pose_landmarks, frame_width: int, frame_height: int):
    left_shoulder_touch = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder_touch = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_shoulder_x_touch = left_shoulder_touch.x * frame_width
    right_shoulder_x_touch = right_shoulder_touch.x * frame_width
    left_shoulder_y_touch = left_shoulder_touch.y * frame_height
    right_shoulder_y_touch = right_shoulder_touch.y * frame_height

    neck_x = (left_shoulder_x_touch + right_shoulder_x_touch) * 0.5
    neck_y = (left_shoulder_y_touch + right_shoulder_y_touch) * 0.5
    body_z = (left_shoulder_touch.z + right_shoulder_touch.z) * 0.5

    shoulder_width_for_body = abs(right_shoulder_x_touch - left_shoulder_x_touch) + 1e-6
    chest_y = neck_y + (0.1 * shoulder_width_for_body)
    neck_offset_x = 0.15 * shoulder_width_for_body
    chest_offset_x = 0.20 * shoulder_width_for_body
    body_targets = [
        (neck_x, neck_y, body_z),
        (neck_x - neck_offset_x, neck_y, body_z),
        (neck_x + neck_offset_x, neck_y, body_z),
        (neck_x, chest_y, body_z),
        (neck_x - chest_offset_x, chest_y, body_z),
        (neck_x + chest_offset_x, chest_y, body_z),
    ]
    return body_targets


def detect_potential_touch(
    hands_results,
    frame_width: int,
    frame_height: int,
    face_reference_z: float,
    left_face_edge_point: tuple[float, float],
    right_face_edge_point: tuple[float, float],
    central_face_targets: list[tuple[float, float]],
    side_face_targets: list[tuple[float, ...]],
    body_targets: list[tuple[float, ...]],
    return_details: bool = False,
) -> bool | dict[str, bool | str | float | None]:
    actual_face_width = _distance(left_face_edge_point, right_face_edge_point)
    face_threshold = actual_face_width * FACE_TOUCH_SCALE
    body_threshold = actual_face_width * BODY_TOUCH_SCALE
    best_touch = _empty_touch_details()

    def update_best_touch(
        region: str,
        hand: str | None,
        distance: float,
        threshold: float,
        z_distance: float | None = None,
        z_threshold: float | None = None,
    ) -> None:
        confidence = _touch_confidence_with_z(distance, threshold, z_distance, z_threshold)
        if confidence > best_touch["confidence"]:
            best_touch.update(
                {
                    "detected": True,
                    "region": region,
                    "hand": hand,
                    "confidence": confidence,
                },
            )

    for hand_index, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        hand_points = hand_landmarks.landmark
        hand_label = _get_hand_label(hands_results, hand_index)
        for point_index in KEY_HAND_POINTS:
            hand_z = hand_points[point_index].z
            hand_point = _landmark_to_pixel(
                hand_points,
                point_index,
                frame_width,
                frame_height,
            )
            if abs(hand_z - face_reference_z) <= FACE_Z_THRESHOLD:
                for target_point in central_face_targets:
                    distance = _distance(hand_point, target_point)
                    if distance <= face_threshold:
                        update_best_touch(
                            "central_face",
                            hand_label,
                            distance,
                            face_threshold,
                            abs(hand_z - face_reference_z),
                            FACE_Z_THRESHOLD,
                        )
            for target_point in side_face_targets:
                distance = _distance(hand_point, target_point)
                target_z = target_point[2] if len(target_point) > 2 else None
                z_distance = abs(hand_z - target_z) if target_z is not None else None
                if (
                    distance <= face_threshold
                    and z_distance is not None
                    and z_distance <= FACE_SIDE_Z_THRESHOLD
                ):
                    update_best_touch(
                        "side_face",
                        hand_label,
                        distance,
                        face_threshold,
                        z_distance,
                        FACE_SIDE_Z_THRESHOLD,
                    )
            for target_point in body_targets:
                distance = _distance(hand_point, target_point)
                target_z = target_point[2] if len(target_point) > 2 else None
                z_distance = abs(hand_z - target_z) if target_z is not None else None
                if (
                    distance <= body_threshold
                    and z_distance is not None
                    and z_distance <= BODY_Z_THRESHOLD
                ):
                    update_best_touch(
                        "body",
                        hand_label,
                        distance,
                        body_threshold,
                        z_distance,
                        BODY_Z_THRESHOLD,
                    )

    if return_details:
        return best_touch
    return bool(best_touch["detected"])
