import math


def _landmark_to_pixel(landmarks, index: int, frame_width: int, frame_height: int) -> tuple[float, float]:
    point = landmarks[index]
    return point.x * frame_width, point.y * frame_height


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
