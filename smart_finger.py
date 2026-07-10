"""
Smart Finger v2 — 每根手指独立轨迹分析与手势识别

架构:
  HandTracker → 21关键点
  FingerTrail → 每根手指: 伸展/角度/弯曲的轨迹追踪 + 速度/加速度
  MotionAnalyzer → 综合5指分析 → 手势触发
  GestureNet → MediaPipe神经网络手势识别(9类)
"""

import cv2
import os
import time
import math
import json
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from PIL import ImageGrab

import numpy as np
import joblib

import torch
import torch.nn as nn

from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    GestureRecognizer, GestureRecognizerOptions,
    HandLandmarksConnections, RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat

# ─── 自定义 MLP (与训练器一致) ─────────────────────────────
class GestureMLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x): return self.net(x)


class CustomGestureClassifier:
    """加载训练好的模型进行实时手势分类 (含干扰过滤)"""

    def __init__(self, model_dir="gesture_models"):
        self.model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.model_type = None
        self.class_mapping = {}
        self.pred_history = deque(maxlen=5)
        self.confidence_threshold = 0.55

        # 尝试加载模型
        self._try_load()

    def _try_load(self):
        mapping_path = os.path.join(self.model_dir, "class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                self.class_mapping = json.load(f)
            self.id_to_class = {int(k): v for k, v in self.class_mapping.get("id_to_class", {}).items()}
        else:
            self.id_to_class = {}
            for i, name in enumerate(["five_pinch", "three_down", "open_palm", "fist",
                                       "point_up", "ok", "victory", "thumb_up",
                                       "thumb_down", "rest"]):
                self.id_to_class[i] = name

        # 优先加载 MLP, 其次 RF
        mlp_path = os.path.join(self.model_dir, "gesture_mlp.pth")
        rf_path = os.path.join(self.model_dir, "gesture_rf.joblib")

        if os.path.exists(mlp_path):
            try:
                ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
                input_dim = ckpt.get("input_dim", 63)
                num_classes = ckpt.get("num_classes", len(self.id_to_class))
                self.model = GestureMLP(input_dim=input_dim, num_classes=num_classes)
                self.model.load_state_dict(ckpt["model_state"])
                self.model.eval()
                self.scaler_mean = np.array(ckpt.get("scaler_mean", []), dtype=np.float32)
                self.scaler_scale = np.array(ckpt.get("scaler_scale", []), dtype=np.float32)
                self.model_type = "mlp"
                print(f"[CustomModel] 加载 MLP ({input_dim}→{num_classes})")
                return
            except Exception as e:
                print(f"[CustomModel] MLP加载失败: {e}")

        if os.path.exists(rf_path):
            try:
                self.model = joblib.load(rf_path)
                self.model_type = "rf"
                print(f"[CustomModel] 加载 RF")
            except Exception as e:
                print(f"[CustomModel] RF加载失败: {e}")

    def is_ready(self):
        return self.model is not None

    def extract(self, landmarks):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return arr.flatten()

    def predict(self, landmarks):
        """返回 (label_name, confidence, raw_pred)"""
        feat = self.extract(landmarks)

        if self.model_type == "rf":
            proba = self.model.predict_proba([feat])[0]
            pred_id = int(np.argmax(proba))
            conf = float(proba[pred_id])
        elif self.model_type == "mlp":
            if len(self.scaler_mean) > 0:
                feat = (feat - self.scaler_mean) / (self.scaler_scale + 1e-8)
            with torch.no_grad():
                out = self.model(torch.FloatTensor(feat[None, :]))
                prob = torch.softmax(out, 1)[0]
                pred_id = int(prob.argmax())
                conf = float(prob[pred_id])
        else:
            return "unknown", 0.0, -1

        # ─── 干扰过滤 ───
        # 1. 低置信度 → rest
        if conf < self.confidence_threshold:
            return "rest", conf, pred_id

        # 2. 时序平滑: 最近N帧必须多数一致
        self.pred_history.append(pred_id)
        if len(self.pred_history) >= 3:
            counts = np.bincount(self.pred_history, minlength=len(self.id_to_class))
            majority = int(np.argmax(counts))
            majority_ratio = counts[majority] / len(self.pred_history)
            if majority_ratio < 0.6:
                return "rest", conf, pred_id
            if majority != pred_id:
                pred_id = majority
                conf = float(proba[majority]) if self.model_type == "rf" else 0.5

        label = self.id_to_class.get(pred_id, "unknown")
        return label, conf, pred_id


# ─── 常量 ────────────────────────────────────────────────────
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# 每根手指的关键点索引 [TIP, DIP(或IP), PIP, MCP(或CMC)]
FINGER_LANDMARKS = [
    [4, 3, 2, 1],   # Thumb: TIP, IP, MCP, CMC
    [8, 7, 6, 5],   # Index
    [12, 11, 10, 9], # Middle
    [16, 15, 14, 13],# Ring
    [20, 19, 18, 17],# Pinky
]

SCREENSHOT_DIR = None


@dataclass
class FingerMetrics:
    """单根手指当前帧的指标"""
    idx: int
    extended: bool = False
    extension_ratio: float = 0.0     # 指尖到手腕 / PIP到手腕
    angle_deg: float = 0.0           # 手指指向角度(度)
    curl_deg: float = 0.0            # PIP关节弯曲角度
    tip_x: float = 0.0
    tip_y: float = 0.0
    mcp_x: float = 0.0
    mcp_y: float = 0.0


@dataclass
class HandMotionState:
    """每只手的完整轨迹分析状态"""
    finger_metrics: list = field(default_factory=lambda: [None] * 5)
    finger_buf: list = field(default_factory=lambda: [deque(maxlen=12) for _ in range(5)])
    spread_buf: deque = field(default_factory=lambda: deque(maxlen=15))
    spread_ratio: float = 1.0
    palm_angle: float = 0.0
    gesture: str = ""
    gesture_score: float = 0.0
    pinch_done: bool = False
    three_down_active: bool = False

    def reset(self):
        self.finger_buf = [deque(maxlen=12) for _ in range(5)]
        self.spread_buf.clear()
        self.pinch_done = False


def compute_finger_metrics(lms, finger_idx: int) -> FingerMetrics:
    """从landmarks计算单根手指的全部指标"""
    idxs = FINGER_LANDMARKS[finger_idx]
    tip = lms[idxs[0]]
    dip = lms[idxs[1]]
    pip = lms[idxs[2]]
    mcp = lms[idxs[3]]
    wrist = lms[0]

    m = FingerMetrics(idx=finger_idx)
    m.tip_x, m.tip_y = tip.x, tip.y
    m.mcp_x, m.mcp_y = mcp.x, mcp.y

    # 1. 伸展比: 指尖到手腕距离 / PIP到手腕距离
    d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    m.extension_ratio = d_tip / d_pip if d_pip > 0.001 else 1.0
    m.extended = m.extension_ratio > 1.15

    # 2. 手指指向角度 (MCP→TIP向量)
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    m.angle_deg = math.degrees(math.atan2(dy, dx))

    # 3. 弯曲角度 (MCP→PIP 与 PIP→TIP 的夹角)
    v1x = pip.x - mcp.x
    v1y = pip.y - mcp.y
    v2x = tip.x - pip.x
    v2y = tip.y - pip.y
    dot = v1x * v2x + v1y * v2y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 * n2 > 0.001:
        cos_a = max(-1, min(1, dot / (n1 * n2)))
        m.curl_deg = math.degrees(math.acos(cos_a))
    else:
        m.curl_deg = 0.0

    return m


def compute_trend(buf: deque, key="extended", window=6) -> tuple:
    """分析最近N帧的趋势: (方向, 变化量)"""
    n = min(len(buf), window)
    if n < 3:
        return "stable", 0.0
    recent = list(buf)[-n:]

    if key == "extended":
        vals = [float(getattr(f, key)) for f in recent]
    else:
        vals = [getattr(f, key) for f in recent]

    delta = vals[-1] - vals[0]
    if abs(delta) < 0.03:
        return "stable", delta
    return ("rising" if delta > 0 else "falling"), delta


def detect_pinch_by_fingers(states: HandMotionState, lms) -> bool:
    """五指合并检测：基于5指轨迹"""
    tips = [(lms[i].x, lms[i].y) for i in [4, 8, 12, 16, 20]]
    cx = sum(x for x, _ in tips) / 5
    cy = sum(y for _, y in tips) / 5
    avg_dist = sum(math.hypot(x - cx, y - cy) for x, y in tips) / 5

    palm = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
    if palm < 0.01:
        return False
    norm_avg = avg_dist / palm

    states.spread_buf.append(norm_avg)
    if len(states.spread_buf) < 6:
        return False

    recent = list(states.spread_buf)
    vel = (recent[-1] - recent[0]) / max(1, len(recent) - 1)

    if recent[0] > 0.40 and norm_avg < 0.18 and vel < -0.015 and not states.pinch_done:
        states.pinch_done = True
        return True
    if norm_avg > 0.30:
        states.pinch_done = False
    return False


def detect_three_down_by_fingers(states: HandMotionState, lms) -> bool:
    """三指向下检测：基于5指独立轨迹"""
    fm = [compute_finger_metrics(lms, i) for i in range(5)]

    # thumb(0)和pinky(4)必须闭合
    if fm[0].extended or fm[4].extended:
        return False
    # index(1), middle(2), ring(3)必须伸展
    if not (fm[1].extended and fm[2].extended and fm[3].extended):
        return False

    wrist, mid_mcp = lms[0], lms[9]
    hand_down = mid_mcp.y > wrist.y

    # 三根手指指向下: 指尖在MCP下方且角度在向下范围
    angles_down = all(
        60 < fm[i].angle_deg < 120 for i in [1, 2, 3]
    )

    return hand_down and angles_down


def analyze_hand_motion(states: HandMotionState, lms) -> dict:
    """分析每根手指的轨迹 → 输出综合运动状态"""
    # 更新5指指标
    fm = [compute_finger_metrics(lms, i) for i in range(5)]
    states.finger_metrics = fm

    for i in range(5):
        states.finger_buf[i].append(fm[i])

    # 整体spread
    palm = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
    if palm > 0.01:
        tips = [(lms[i].x, lms[i].y) for i in [4, 8, 12, 16, 20]]
        max_d = max(math.hypot(tips[i][0] - tips[j][0], tips[i][1] - tips[j][1])
                    for i in range(5) for j in range(i + 1, 5))
        states.spread_ratio = max_d / palm

    # 每根手指的趋势
    trends = {}
    for i in range(5):
        trends[i] = {
            "ext": compute_trend(states.finger_buf[i], "extended"),
            "angle": compute_trend(states.finger_buf[i], "angle_deg"),
            "curl": compute_trend(states.finger_buf[i], "curl_deg"),
        }

    # 手掌朝向
    dx = lms[9].x - lms[0].x
    dy = lms[9].y - lms[0].y
    states.palm_angle = math.degrees(math.atan2(dy, dx))

    return {
        "fingers": fm,
        "trends": trends,
        "palm_angle": states.palm_angle,
        "spread": states.spread_ratio,
    }


def draw_finger_analysis(frame, lms, fm_list, idx, label, analysis, gesture_text):
    """绘制手部 + 每根手指分析"""
    h, w = frame.shape[:2]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        s, e = lms[conn.start], lms[conn.end]
        cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0, 180, 100), 2)

    colors = [(180, 80, 80), (80, 180, 255), (100, 255, 100), (200, 200, 80), (255, 150, 255)]
    finger_labels = ["TH", "IN", "MI", "RI", "PI"]

    for i, fm in enumerate(fm_list):
        cx, cy = int(fm.tip_x * w), int(fm.tip_y * h)
        col = colors[i]
        cv2.circle(frame, (cx, cy), 7, col, -1)
        cv2.putText(frame, finger_labels[i], (cx - 8, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 2)

        # 显示每根手指指标
        ext_str = "E" if fm.extended else "C"
        txt = f"{finger_labels[i]}:{ext_str} a={fm.angle_deg:.0f} c={fm.curl_deg:.0f}"
        cv2.putText(frame, txt, (w - 180, 60 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    for i, lm in enumerate(lms):
        if i not in [4, 8, 12, 16, 20]:
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame, (cx, cy), 3, (160, 160, 160), -1)

    # 手掌角度可视化
    wrist_x, wrist_y = int(lms[0].x*w), int(lms[0].y*h)
    mid_x, mid_y = int(lms[9].x*w), int(lms[9].y*h)
    cv2.arrowedLine(frame, (wrist_x, wrist_y), (mid_x, mid_y), (0, 200, 255), 2)

    # 信息面板
    base_y = 60 + 50 * idx
    cv2.putText(frame, f"{label}: {gesture_text}", (10, base_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"spread={analysis['spread']:.2f} palm_a={analysis['palm_angle']:.0f}",
                (10, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    # 趋势箭头
    trends_text = []
    for i, t in analysis["trends"].items():
        d, v = t["ext"]
        if d != "stable":
            trends_text.append(f"{finger_labels[i]}_{d}({v:.2f})")
    if trends_text:
        cv2.putText(frame, " ".join(trends_text), (10, base_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)


# ─── 动作函数 ────────────────────────────────────────────────
def take_screenshot():
    global SCREENSHOT_DIR
    now = time.time()
    if now - getattr(take_screenshot, "_last", 0) < 1.5:
        return None
    take_screenshot._last = now
    if SCREENSHOT_DIR is None:
        SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
    ImageGrab.grab().save(p)
    return p


def show_desktop():
    now = time.time()
    if now - getattr(show_desktop, "_last", 0) < 2.0:
        return False
    show_desktop._last = now
    subprocess.run(
        ["powershell", "-command",
         "(New-Object -ComObject Shell.Application).MinimizeAll()"],
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return True


# ─── 引擎类 ──────────────────────────────────────────────────
@dataclass
class HandData:
    landmarks: list
    handedness: list
    label: str
    nn_gesture: str = ""
    nn_score: float = 0.0


class HandTracker:
    def __init__(self, max_hands=2):
        model = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model):
            raise FileNotFoundError(f"模型未找到: {model}")
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO, num_hands=max_hands,
            min_hand_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        self._e = HandLandmarker.create_from_options(opts)
        self._ts = 0

    def process(self, frame):
        self._ts += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = self._e.detect_for_video(MpImage(ImageFormat.SRGB, rgb), self._ts)
        hands = []
        if r.hand_landmarks:
            for i, lm in enumerate(r.hand_landmarks):
                hd = r.handedness[i] if i < len(r.handedness) else []
                label = hd[0].category_name if hd else "?"
                hands.append(HandData(landmarks=lm, handedness=hd, label=label))
        return hands

    def close(self): self._e.close()


class GestureNet:
    def __init__(self, max_hands=2):
        model = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
        if not os.path.exists(model):
            raise FileNotFoundError(f"模型未找到: {model}")
        opts = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO, num_hands=max_hands,
        )
        self._e = GestureRecognizer.create_from_options(opts)
        self._ts = 0

    def process(self, frame, hands):
        self._ts += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = self._e.recognize_for_video(MpImage(ImageFormat.SRGB, rgb), self._ts)
        if r.gestures:
            for i, gs in enumerate(r.gestures):
                if i < len(hands):
                    hands[i].nn_gesture = gs[0].category_name
                    hands[i].nn_score = gs[0].score
        return hands

    def close(self): self._e.close()


# ─── 主循环 ──────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    tracker = HandTracker()
    gesture_net = GestureNet()
    custom_clf = CustomGestureClassifier()
    motion_states: dict[str, HandMotionState] = {}
    screenshot_timer = 0
    desktop_timer = 0

    print("=" * 60)
    print("  Smart Finger v2 — 每指轨迹 + NN + 自定义模型")
    print("=" * 60)
    print("  ▌ MediaPipe NN: Closed_Fist, Open_Palm, Pointing_Up,")
    print("  ▌              Thumb_Up/Down, Victory, ILoveYou, OK")
    if custom_clf.is_ready():
        print(f"  ▌ 自定义模型: {custom_clf.model_type.upper()} 已加载")
    else:
        print("  ▌ 自定义模型: 未训练, 运行 gesture_trainer.py")
    print("  ▌ 五指合并(轨迹) → 截图")
    print("  ▌ 三指向下     → 桌面")
    print("  ▌ 按 1 手部架构图 | ESC 退出")
    print("=" * 60)

    show_diagram = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hands = tracker.process(frame)
        hands = gesture_net.process(frame, hands)

        for hand in hands:
            key = hand.label
            if key not in motion_states:
                motion_states[key] = HandMotionState()
            ms = motion_states[key]

            lms = hand.landmarks
            analysis = analyze_hand_motion(ms, lms)

            # 五指合并(轨迹)
            if detect_pinch_by_fingers(ms, lms):
                p = take_screenshot()
                if p:
                    screenshot_timer = 60
                    print(f"[截图] {key} 五指合并 → {os.path.basename(p)}")

            # 三指向下
            if detect_three_down_by_fingers(ms, lms):
                if not ms.three_down_active:
                    ms.three_down_active = True
                    if desktop_timer <= 0 and show_desktop():
                        desktop_timer = 90
                        print(f"[桌面] {key} 三指向下")
            else:
                ms.three_down_active = False

            # 自定义模型推理
            custom_label, custom_conf, _ = custom_clf.predict(lms) if custom_clf.is_ready() else ("-", 0, -1)

            # 显示
            gesture_text = f"MP:{hand.nn_gesture}({hand.nn_score:.2f}) Custom:{custom_label}({custom_conf:.2f})"
            draw_finger_analysis(frame, lms, ms.finger_metrics, hands.index(hand),
                                 hand.label, analysis, gesture_text)

        # 清理消失的手
        for k in list(motion_states.keys()):
            if not any(h.label == k for h in hands):
                del motion_states[k]

        # 动作反馈
        if screenshot_timer > 0:
            screenshot_timer -= 1
            cv2.putText(frame, "SCREENSHOT!", (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if desktop_timer > 0:
            desktop_timer -= 1
            cv2.putText(frame, "SHOW DESKTOP!", (250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        if show_diagram:
            dia = cv2.imread("hand_structure_diagram.png")
            if dia is not None:
                s = min(640 / dia.shape[1], 480 / dia.shape[0])
                cv2.imshow("Diagram", cv2.resize(dia, (int(dia.shape[1]*s), int(dia.shape[0]*s))))

        cv2.imshow("Smart Finger v2", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('1'):
            show_diagram = not show_diagram
            if not show_diagram:
                cv2.destroyWindow("Diagram")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    gesture_net.close()


if __name__ == "__main__":
    main()
