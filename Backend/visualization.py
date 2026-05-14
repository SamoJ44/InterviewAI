import cv2

from config import (
    BAD_SCORE_THRESHOLD,
    FINAL_EXCELLENT_THRESHOLD,
    FINAL_GOOD_THRESHOLD,
    GOOD_SCORE_THRESHOLD,
)


def draw_face_landmarks(frame, face_landmarks, mp_drawing, mp_drawing_styles, mp_face_mesh) -> None:
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )


def draw_pose_landmarks(frame, pose_landmarks, mp_drawing, mp_drawing_styles, mp_pose) -> None:
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )


def draw_overlay(
    frame,
    eye_contact_score: int,
    eye_status_text: str,
    posture_score: int,
    stability_score: int,
    self_touch_active: bool,
    final_score: float,
    calibration_active: bool,
    calibration: dict | None = None,
) -> None:
    if eye_contact_score > GOOD_SCORE_THRESHOLD:
        status_text = "Eye Contact: YES"
        status_color = (0, 255, 0)
    elif eye_contact_score < BAD_SCORE_THRESHOLD:
        status_text = "Eye Contact: NO"
        status_color = (0, 0, 255)
    else:
        status_text = "Eye Contact: NEUTRAL"
        status_color = (0, 255, 255)

    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(
        frame,
        f"Score: {eye_contact_score}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
    eye_status_color = (0, 0, 255) if eye_status_text == "CLOSED" else (0, 255, 0)
    cv2.putText(
        frame,
        f"Eye Status: {eye_status_text}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        eye_status_color,
        2,
    )

    if posture_score > GOOD_SCORE_THRESHOLD:
        posture_text = "Posture: Good"
        posture_color = (0, 255, 0)
    elif posture_score < BAD_SCORE_THRESHOLD:
        posture_text = "Posture: Bad"
        posture_color = (0, 0, 255)
    else:
        posture_text = "Posture: Neutral"
        posture_color = (0, 255, 255)
    cv2.putText(
        frame,
        posture_text,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        posture_color,
        2,
    )
    cv2.putText(
        frame,
        f"Posture Score: {posture_score}%",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        posture_color,
        2,
    )

    if stability_score > GOOD_SCORE_THRESHOLD:
        stability_text = "Stability: Good"
        stability_color = (0, 255, 0)
    elif stability_score < BAD_SCORE_THRESHOLD:
        stability_text = "Stability: Bad"
        stability_color = (0, 0, 255)
    else:
        stability_text = "Stability: Neutral"
        stability_color = (0, 255, 255)
    cv2.putText(
        frame,
        stability_text,
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        stability_color,
        2,
    )
    cv2.putText(
        frame,
        f"Stability Score: {stability_score}%",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        stability_color,
        2,
    )

    final_score_display = int(round(final_score))
    if final_score > FINAL_EXCELLENT_THRESHOLD:
        final_label = "Interview: Excellent"
        final_color = (0, 255, 0)
    elif final_score >= FINAL_GOOD_THRESHOLD:
        final_label = "Interview: Good"
        final_color = (0, 255, 255)
    else:
        final_label = "Interview: Needs Improvement"
        final_color = (0, 0, 255)
    cv2.putText(
        frame,
        final_label,
        (20, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        final_color,
        2,
    )
    cv2.putText(
        frame,
        f"Final Score: {final_score_display}%",
        (20, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        final_color,
        2,
    )

    if self_touch_active:
        self_touch_text = "Self-Touch: YES"
        self_touch_color = (0, 0, 255)
    else:
        self_touch_text = "Self-Touch: NO"
        self_touch_color = (0, 255, 0)
    cv2.putText(
        frame,
        self_touch_text,
        (20, 355),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        self_touch_color,
        2,
    )

    if calibration:
        calibration_status = calibration.get("status")
        if calibration_status == "preparing":
            countdown_remaining = int(round(float(calibration.get("countdown_remaining", 0.0))))
            calibration_text = f"Get ready: {countdown_remaining}s"
        elif calibration_status == "calibrating":
            progress = int(round(float(calibration.get("progress", 0.0)) * 100.0))
            calibration_text = f"Calibrating posture: {progress}%"
        elif calibration_status == "ready":
            calibration_text = "Calibration ready"
        elif calibration_status == "failed":
            calibration_text = "Calibration failed"
        else:
            calibration_text = None
    else:
        calibration_text = "Calibrating posture" if calibration_active else None

    if calibration_text:
        cv2.putText(
            frame,
            calibration_text,
            (20, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
        )
