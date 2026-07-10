import cv2
import numpy as np
import os
import time
from datetime import datetime
from dataclasses import dataclass

from PIL import ImageGrab

from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult,
    HandLandmarksConnections, RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
from mediapipe.tasks.python.vision import drawing_utils as mp_draw
from mediapipe.tasks.python.vision import drawing_styles as mp_styles


# 21 个手部关键点索引
class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


FINGER_TIPS = [HandLandmark.THUMB_TIP, HandLandmark.INDEX_TIP,
               HandLandmark.MIDDLE_TIP, HandLandmark.RING_TIP, HandLandmark.PINKY_TIP]
FINGER_PIPS = [HandLandmark.THUMB_IP, HandLandmark.INDEX_PIP,
               HandLandmark.MIDDLE_PIP, HandLandmark.RING_PIP, HandLandmark.PINKY_PIP]


@dataclass
class HandData:
    landmarks: list
    handedness: list
    label: str

    @property
    def is_right(self) -> bool:
        return self.label == "Right"

    @property
    def is_left(self) -> bool:
        return self.label == "Left"

    def landmark(self, idx: int):
        return self.landmarks[idx]

    def landmark_xy(self, idx: int):
        lm = self.landmarks[idx]
        return lm.x, lm.y


class HandTracker:
    def __init__(self, model_path=None, max_hands=2,
                 detection_confidence=0.7, tracking_confidence=0.5):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"手部模型未找到: {model_path}\n"
                "请下载: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        self.max_hands = max_hands
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)
        self._timestamp = 0

    def process(self, frame: np.ndarray):
        self._timestamp += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(ImageFormat.SRGB, rgb)
        results = self.hand_landmarker.detect_for_video(mp_image, self._timestamp)

        hands: list[HandData] = []
        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):
                hd = results.handedness[i] if i < len(results.handedness) else []
                label = hd[0].category_name if hd else "Unknown"
                hands.append(HandData(
                    landmarks=landmarks,
                    handedness=hd,
                    label=label,
                ))

        return hands, results

    def draw_landmarks(self, frame: np.ndarray, results, draw_ids=True):
        if not results or not results.hand_landmarks:
            return frame

        connections = HandLandmarksConnections.HAND_CONNECTIONS
        for landmarks in results.hand_landmarks:
            for conn in connections:
                start = landmarks[conn.start]
                end = landmarks[conn.end]
                h, w, _ = frame.shape
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for idx, lm in enumerate(landmarks):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (255, 0, 0) if idx == 0 else (0, 255, 255)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                if draw_ids:
                    cv2.putText(frame, str(idx), (cx - 6, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        return frame

    def draw_bbox(self, frame: np.ndarray, hand: HandData, color=(0, 255, 0)):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand.landmarks]
        ys = [lm.y * h for lm in hand.landmarks]
        x_min, x_max = int(min(xs)) - 10, int(max(xs)) + 10
        y_min, y_max = int(min(ys)) - 10, int(max(ys)) + 10
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, hand.label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def draw_info(self, frame: np.ndarray, hand: HandData):
        h, w, _ = frame.shape
        wrist = hand.landmarks[HandLandmark.WRIST]
        cx, cy = int(wrist.x * w), int(wrist.y * h)
        score = hand.handedness[0].score if hand.handedness else 0
        cv2.putText(frame, f"{hand.label} ({score:.2f})",
                    (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
        return frame

    def close(self):
        self.hand_landmarker.close()


def detect_finger_states(hand: HandData) -> list[bool]:
    states = [False] * 5
    wrist = hand.landmarks[HandLandmark.WRIST]

    for i in range(5):
        tip = hand.landmarks[FINGER_TIPS[i]]
        pip = hand.landmarks[FINGER_PIPS[i]]
        d_tip = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
        d_pip = ((pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2) ** 0.5
        states[i] = d_tip > d_pip

    return states


def get_finger_names(states: list[bool]) -> str:
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    parts = [f"{n}({'open' if s else 'closed'})" for n, s in zip(names, states)]
    return ", ".join(parts)


def count_fingers_open(states: list[bool]) -> int:
    return sum(states)


def detect_three_fingers_down(hand: HandData) -> bool:
    states = detect_finger_states(hand)
    if states[0] or states[4]:
        return False
    if not (states[1] and states[2] and states[3]):
        return False
    wrist = hand.landmarks[HandLandmark.WRIST]
    mid_mcp = hand.landmarks[HandLandmark.MIDDLE_MCP]
    hand_down = mid_mcp.y > wrist.y
    idx_tip = hand.landmarks[HandLandmark.INDEX_TIP]
    mid_tip = hand.landmarks[HandLandmark.MIDDLE_TIP]
    ring_tip = hand.landmarks[HandLandmark.RING_TIP]
    tips_down = (idx_tip.y > mid_mcp.y and mid_tip.y > mid_mcp.y and ring_tip.y > mid_mcp.y)
    return hand_down and tips_down


def _tip_spread_ratio(hand: HandData) -> float:
    wrist = hand.landmarks[HandLandmark.WRIST]
    mid_mcp = hand.landmarks[HandLandmark.MIDDLE_MCP]
    palm_size = ((wrist.x - mid_mcp.x) ** 2 + (wrist.y - mid_mcp.y) ** 2) ** 0.5
    if palm_size < 0.01:
        return 999.0
    tips = [(hand.landmarks[i].x, hand.landmarks[i].y) for i in FINGER_TIPS]
    max_pairwise = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            d = ((tips[i][0] - tips[j][0]) ** 2 + (tips[i][1] - tips[j][1]) ** 2) ** 0.5
            if d > max_pairwise:
                max_pairwise = d
    return max_pairwise / palm_size


_SCREENSHOT_DIR = None
_LAST_SCREENSHOT_TIME = 0
_SCREENSHOT_COOLDOWN = 1.5


def take_screenshot():
    global _SCREENSHOT_DIR, _LAST_SCREENSHOT_TIME
    now = time.time()
    if now - _LAST_SCREENSHOT_TIME < _SCREENSHOT_COOLDOWN:
        return None
    _LAST_SCREENSHOT_TIME = now

    if _SCREENSHOT_DIR is None:
        _SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(_SCREENSHOT_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(_SCREENSHOT_DIR, f"screenshot_{ts}.png")
    img = ImageGrab.grab()
    img.save(path)
    return path


_LAST_DESKTOP_TIME = 0
_DESKTOP_COOLDOWN = 2.0


def show_desktop():
    global _LAST_DESKTOP_TIME
    now = time.time()
    if now - _LAST_DESKTOP_TIME < _DESKTOP_COOLDOWN:
        return False
    _LAST_DESKTOP_TIME = now
    import subprocess
    subprocess.run(
        ["powershell", "-command", "(New-Object -ComObject Shell.Application).MinimizeAll()"],
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return True


class HandGestureState:
    def __init__(self, window=10):
        self.spread_buf = []
        self.window = window
        self.motion = "STEADY"
        self.posture = "UNKNOWN"
        self.was_open = False
        self.pinch_ready = False
        self.was_down = False

    def update(self, spread: float, hand: HandData):
        self.spread_buf.append(spread)
        if len(self.spread_buf) > self.window:
            self.spread_buf.pop(0)

        vel = 0.0
        if len(self.spread_buf) >= 4:
            recent = self.spread_buf[-4:]
            vel = (recent[-1] - recent[0]) / 3.0

        if vel < -0.025:
            self.motion = "CLOSING"
        elif vel > 0.025:
            self.motion = "OPENING"
        else:
            self.motion = "STEADY"

        cur = self.spread_buf[-1] if self.spread_buf else spread
        if cur > 0.5:
            self.posture = "OPEN"
        elif cur < 0.2:
            self.posture = "CLOSED"
        else:
            self.posture = "PARTIAL"

        if self.posture == "OPEN":
            self.was_open = True

        trigger = False
        if (self.was_open and self.posture == "CLOSED"
                and self.motion == "CLOSING" and not self.pinch_ready):
            self.pinch_ready = True
            trigger = True
            self.was_open = False

        if self.posture != "CLOSED":
            self.pinch_ready = False

        return trigger, vel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    tracker = HandTracker(max_hands=2)
    gesture_states = {}
    screenshot_path = None
    screenshot_timer = 0

    print("手部追踪已启动 — 按 ESC 退出")
    print("手势1: 五指合并(轨迹检测) → 全屏截图")
    print("手势2: 三指向下(食中无, 拇尾收) → 返回桌面")
    frame_count = 0
    desktop_timer = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands, results = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, results, draw_ids=False)

        for hand in hands:
            tracker.draw_bbox(frame, hand)
            states = detect_finger_states(hand)
            label = f"{hand.label}: {get_finger_names(states)}"
            idx = hands.index(hand)
            cv2.putText(frame, label, (10, 60 + 30 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            key = hand.label
            if key not in gesture_states:
                gesture_states[key] = HandGestureState()
            gs = gesture_states[key]

            spread = _tip_spread_ratio(hand)
            trigger, vel = gs.update(spread, hand)

            frame_count += 1
            if frame_count % 15 == 0:
                print(f"[{hand.label}] spread={spread:.3f} mot={gs.motion} pos={gs.posture} vel={vel:.3f} was_open={gs.was_open}")

            cv2.putText(frame, f"spr:{spread:.2f} vel:{vel:.2f}", (10, 90 + 30 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            cv2.putText(frame, f"{gs.motion} {gs.posture}", (10, 110 + 30 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if gs.motion == "STEADY" else (0, 0, 255), 1)

            if trigger:
                print(f"  → 五指合并(轨迹) 触发截图! spread={spread:.3f}")
                path = take_screenshot()
                if path:
                    screenshot_path = path
                    screenshot_timer = 60
                    print(f"[截图] {hand.label} {path}")

            three_down = detect_three_fingers_down(hand)
            if three_down:
                cv2.putText(frame, "三指向下", (300, 60 + 30 * idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if three_down and not gs.was_down:
                if desktop_timer <= 0 and show_desktop():
                    desktop_timer = 90
                    print(f"  → 三指向下 返回桌面 [{hand.label}]")
            gs.was_down = three_down

        for key in list(gesture_states.keys()):
            if not any(h.label == key for h in hands):
                del gesture_states[key]

        if screenshot_timer > 0:
            screenshot_timer -= 1
            h, w, _ = frame.shape
            cv2.putText(frame, "SCREENSHOT!", (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if screenshot_path:
                fn = os.path.basename(screenshot_path)
                cv2.putText(frame, fn, (w // 2 - 120, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if desktop_timer > 0:
            desktop_timer -= 1
            h, w, _ = frame.shape
            cv2.putText(frame, "SHOW DESKTOP!", (w // 2 - 110, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        cv2.imshow("Smart Finger - Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
