from __future__ import annotations

import inspect
import math
import time
from pathlib import Path
from types import SimpleNamespace

import mediapipe as mp
import numpy as np

import api
from pipeline import (
    Processors,
    build_score_detail,
    calculate_final_score_detail,
    init_state,
    process_frame,
)


def _blank_frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


class _FakeProcessor:
    def __init__(self, result):
        self._result = result

    def process(self, _frame):
        return self._result


def _fake_processors(face_landmarks=None, pose_landmarks=None, hand_landmarks=None) -> Processors:
    return Processors(
        drawing_utils=None,
        drawing_styles=None,
        face_mesh_module=None,
        pose_module=mp.solutions.pose,
        face_mesh=_FakeProcessor(SimpleNamespace(multi_face_landmarks=face_landmarks)),
        pose=_FakeProcessor(SimpleNamespace(pose_landmarks=pose_landmarks)),
        hands=_FakeProcessor(SimpleNamespace(multi_hand_landmarks=hand_landmarks)),
    )


def _api_like_response(result):
    return {
        "scores": {
            "eye_contact": result["eye_contact_score"],
            "posture": result["posture_score"],
            "stability": result["stability_score"],
            "self_touch": result["self_touch_score"],
            "final": result["final_score"],
        },
        "score_details": result["score_details"],
        "flags": {
            "eye_status": result["eye_status_text"],
            "self_touch_active": result["self_touch_active"],
            "calibration_active": result["calibration_active"],
            "face_detected": result["face_detected"],
            "pose_detected": result["pose_detected"],
            "gaze_away_active": result["gaze_away_active"],
        },
    }


def _assert_close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_no_face_detected():
    result = process_frame(_blank_frame(), _fake_processors(), init_state())
    response = _api_like_response(result)

    assert response["score_details"]["eye_contact"]["available"] is False
    assert response["score_details"]["eye_contact"]["status"] == "no_face"
    assert response["flags"]["eye_status"] == "NO_FACE"
    assert response["scores"]["eye_contact"] == 0


def test_no_pose_detected():
    result = process_frame(_blank_frame(), _fake_processors(), init_state())
    response = _api_like_response(result)

    assert response["score_details"]["posture"]["available"] is False
    assert response["score_details"]["stability"]["available"] is False
    assert response["scores"]["posture"] == 0
    assert response["scores"]["stability"] == 0


def test_calibration_preparing():
    result = process_frame(_blank_frame(), _fake_processors(), init_state())
    posture = result["score_details"]["posture"]
    final = result["score_details"]["final"]

    assert result["calibration"]["status"] == "preparing"
    assert result["calibration"]["active"] is False
    assert posture["status"] == "preparing"
    assert posture["available"] is False
    assert posture["value"] is None
    assert posture["reason"] == "posture_preparation_active"
    assert final["reliability"] < 1.0


def test_calibration_active():
    state = init_state()
    state.session_start_time = time.time() - state.pre_calibration_countdown_seconds - 0.1
    result = process_frame(_blank_frame(), _fake_processors(), state)
    posture = result["score_details"]["posture"]

    assert result["calibration"]["status"] == "calibrating"
    assert result["calibration"]["active"] is True
    assert posture["status"] == "calibrating"
    assert posture["available"] is False
    assert posture["reason"] == "posture_calibration_active"


def test_calibration_failed():
    state = init_state()
    state.session_start_time = (
        time.time()
        - state.pre_calibration_countdown_seconds
        - state.calibration_duration_seconds
        - 1.0
    )
    result = process_frame(_blank_frame(), _fake_processors(), state)
    posture = result["score_details"]["posture"]

    assert result["calibration"]["active"] is False
    assert result["calibration"]["ready"] is False
    assert result["calibration"]["status"] == "failed"
    assert posture["status"] == "calibration_failed"
    assert posture["reason"] == "no_valid_calibration_samples"


def test_final_score_availability():
    all_available = {
        "eye_contact": build_score_detail(80, True, "ok"),
        "posture": build_score_detail(60, True, "ok"),
        "stability": build_score_detail(100, True, "ok"),
        "self_touch": build_score_detail(90, True, "ok"),
    }
    final = calculate_final_score_detail(all_available)
    _assert_close(final["reliability"], 1.0, "all-available reliability")
    _assert_close(final["value"], 81.5, "all-available final")

    partial = {
        "eye_contact": build_score_detail(80, True, "ok"),
        "posture": build_score_detail(None, False, "calibrating"),
        "stability": build_score_detail(None, False, "unavailable"),
        "self_touch": build_score_detail(90, True, "ok"),
    }
    final = calculate_final_score_detail(partial)
    expected = ((80 * 0.35) + (90 * 0.15)) / 0.50
    assert final["status"] == "partial"
    _assert_close(final["reliability"], 0.50, "partial reliability")
    _assert_close(final["value"], expected, "partial normalized final")


def test_draw_overlay_pass_through():
    source = inspect.getsource(api.analyze_frame)
    assert "draw_overlay = _parse_bool(draw_overlay)" in source
    assert "draw=draw_overlay" in source


def main() -> None:
    tests = [
        test_no_face_detected,
        test_no_pose_detected,
        test_calibration_preparing,
        test_calibration_active,
        test_calibration_failed,
        test_final_score_availability,
        test_draw_overlay_pass_through,
    ]

    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print(f"All scoring diagnostics passed from {Path(__file__).name}.")


if __name__ == "__main__":
    main()
