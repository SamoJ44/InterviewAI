import cv2

from pipeline import create_processors, init_state, process_frame


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    state = init_state()
    with create_processors() as processors:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to read frame from webcam.")
                break

            result = process_frame(frame, processors, state, draw=True)
            cv2.imshow("FaceMesh + Pose", result["frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
