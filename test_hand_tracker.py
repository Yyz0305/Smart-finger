import cv2
import numpy as np
from hand_tracker import (
    HandTracker, HandLandmark, HandData,
    detect_finger_states, get_finger_names, count_fingers_open
)


def make_fake_landmarks(is_right: bool):
    class FakeLM:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    lms = [FakeLM(0.5, 0.5)] * 21
    for i in range(21):
        lms[i] = FakeLM(0.5 + i * 0.01, 0.3 + i * 0.02, 0.0)
    return lms


def test_finger_states():
    print("=== Finger State Detection Test ===")
    fake_right = HandData(
        landmarks=make_fake_landmarks(True),
        handedness=[],
        label="Right",
    )
    states = detect_finger_states(fake_right)
    print(f"Finger states: {get_finger_names(states)}")
    print(f"Fingers open: {count_fingers_open(states)}")
    assert len(states) == 5
    print("PASSED\n")


def test_hand_left_right():
    print("=== Left/Right Classification Test ===")
    right = HandData(landmarks=[], handedness=[], label="Right")
    left = HandData(landmarks=[], handedness=[], label="Left")
    assert right.is_right and not right.is_left
    assert left.is_left and not left.is_right
    print("PASSED\n")


def test_draw_no_crash():
    print("=== Drawing Test (no crash) ===")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Test with None results
    from types import SimpleNamespace
    empty_result = SimpleNamespace(hand_landmarks=[])
    tracker = HandTracker(max_hands=2)
    result = tracker.draw_landmarks(frame, empty_result)
    assert result is frame
    print("draw_landmarks (empty) - OK")
    tracker.close()
    print("PASSED\n")


def run_live_preview():
    print("=== Live Webcam Preview ===")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("SKIPPED: 未检测到摄像头")
        return

    tracker = HandTracker(max_hands=2)
    print("按 ESC 退出实时预览...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands, results = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, results, draw_ids=True)

        for hand in hands:
            tracker.draw_bbox(frame, hand, (0, 255, 0))
            states = detect_finger_states(hand)
            label = f"{hand.label}: {get_finger_names(states)}"
            idx = hands.index(hand)
            cv2.putText(frame, label, (10, 30 + 30 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Smart Finger - Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Live preview ended.\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Finger - Hand Tracker Test Suite")
    print("=" * 50)

    test_hand_left_right()
    test_finger_states()
    test_draw_no_crash()

    print("=" * 50)
    print("  All unit tests PASSED")
    print("=" * 50)

    run_live_preview()
