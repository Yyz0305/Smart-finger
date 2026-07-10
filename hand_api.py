"""
Hand API — 手部追踪数据接口层

在后台线程运行 MediaPipe HandTracker,
通过线程安全的方式暴露手部数据供其他模块 (如 3D 场景) 读取。

用法:
    api = HandAPI()
    api.start()
    data = api.get_hand_data()  # 最新手部数据 dict
    api.stop()
"""

import threading
import time
import math
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field

from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections, RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
HAND_LANDMARK_TIPS = FINGER_TIPS


@dataclass
class FingerState:
    """单根手指状态"""
    index: int
    name: str = ""
    extended: bool = False
    tip_x: float = 0.0
    tip_y: float = 0.0
    tip_z: float = 0.0
    angle_deg: float = 0.0
    curl_deg: float = 0.0


@dataclass
class HandState:
    """完整手部状态"""
    detected: bool = False
    label: str = ""
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    wrist_z: float = 0.0
    palm_x: float = 0.0        # 掌心 = 中指MCP
    palm_y: float = 0.0
    palm_z: float = 0.0
    palm_size: float = 0.0     # 归一化手掌尺寸 (wrist→mid_mcp 距离)
    fingers: list = field(default_factory=lambda: [FingerState(i, FINGER_NAMES[i]) for i in range(5)])
    grab_strength: float = 0.0   # 0=张开, 1=紧握
    pointing_down: bool = False
    landmarks_raw: list = field(default_factory=list)


class DepthCalibrator:
    """深度校准器 — 基于掌心尺寸比例推算真实深度

    原理: 手掌在画面中的大小与到摄像头的距离成反比 (小孔成像模型)
    calibration: 用户将手放在已知距离(约一臂长), 记录参考掌尺寸
    """

    def __init__(self, ref_distance=0.5):
        self.ref_distance = ref_distance       # 参考距离 (米)
        self.ref_palm_size = None              # 参考掌尺寸 (归一化坐标)
        self.calibrated = False

    def calibrate(self, palm_size):
        self.ref_palm_size = palm_size
        self.calibrated = True
        return self.ref_palm_size

    def estimate_distance(self, palm_size):
        """根据当前手掌大小估算到摄像头的距离 (米)"""
        if not self.calibrated or palm_size < 0.001:
            return self.ref_distance
        return self.ref_distance * (self.ref_palm_size / palm_size)

    def map_to_3d(self, hx, hy, palm_size, fov_h=60):
        """将归一化坐标 (0~1) 映射到 3D 世界坐标 (米)

        使用小孔成像模型:
          x_world = (x_norm - 0.5) * distance * 2 * tan(fov_h/2) * (w/h)
          y_world = -(y_norm - 0.5) * distance * 2 * tan(fov_h/2)
          z_world = distance
        """
        d = self.estimate_distance(palm_size)
        scale = d * 2 * math.tan(math.radians(fov_h / 2))
        aspect = 640 / 480  # 假设摄像头宽高比
        x = (hx - 0.5) * scale * aspect
        y = -(hy - 0.5) * scale
        z = d
        return x, y, z


class HandAPI:
    """手部追踪 API — 后台线程运行, 线程安全获取数据"""

    def __init__(self, camera_id=0, max_hands=1):
        self.camera_id = camera_id
        self.max_hands = max_hands
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # 当前帧的手部数据 (线程安全)
        self._hand_data: list[HandState] = [HandState()]
        self._latest_frame = None

        # 初始化追踪器
        model_path = self._find_model()
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._tracker = HandLandmarker.create_from_options(opts)
        self._ts = 0

    def _find_model(self):
        import os
        p = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"模型未找到: {p}")

    def start(self):
        """启动后台追踪线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[HandAPI] 已启动")
        return self

    def stop(self):
        """停止追踪"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._tracker.close()
        if hasattr(self, '_cap'):
            self._cap.release()
        print("[HandAPI] 已停止")

    def get_hand_data(self) -> list[HandState]:
        """获取最新手部数据 (线程安全)"""
        with self._lock:
            return self._hand_data.copy()

    def get_primary_hand(self) -> HandState:
        """获取第一只手 (最优先)"""
        data = self.get_hand_data()
        return data[0] if data else HandState()

    def get_latest_frame(self):
        """获取最新摄像头帧 (用于调试/显示)"""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _compute_finger_state(self, lms, i) -> FingerState:
        """计算单根手指状态"""
        tip = lms[FINGER_TIPS[i]]
        pip = lms[FINGER_PIPS[i]]
        mcp = lms[[1, 5, 9, 13, 17][i]]
        wrist = lms[0]

        fs = FingerState(index=i, name=FINGER_NAMES[i])
        fs.tip_x, fs.tip_y, fs.tip_z = tip.x, tip.y, tip.z

        d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        fs.extended = d_tip > d_pip * 1.15 if d_pip > 0.001 else False

        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        fs.angle_deg = math.degrees(math.atan2(dy, dx))

        v1x, v1y = pip.x - mcp.x, pip.y - mcp.y
        v2x, v2y = tip.x - pip.x, tip.y - pip.y
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 * n2 > 0.001:
            dot = v1x * v2x + v1y * v2y
            fs.curl_deg = math.degrees(math.acos(max(-1, min(1, dot / (n1 * n2)))))

        return fs

    def _compute_grab_strength(self, lms) -> float:
        """计算抓握强度 0~1"""
        tips = [(lms[i].x, lms[i].y) for i in FINGER_TIPS]
        cx = sum(x for x, _ in tips) / 5
        cy = sum(y for _, y in tips) / 5
        avg_dist = sum(math.hypot(x - cx, y - cy) for x, y in tips) / 5
        palm = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
        if palm < 0.01:
            return 0.0
        ratio = avg_dist / palm
        # ratio 大 → 张开, ratio 小 → 握紧
        return max(0.0, min(1.0, 1.0 - (ratio - 0.08) / 0.5))

    def _run_loop(self):
        """后台主循环"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("[HandAPI] 无法打开摄像头")
            self._running = False
            return
        self._cap = cap

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            self._ts += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(ImageFormat.SRGB, rgb)
            results = self._tracker.detect_for_video(mp_img, self._ts)

            hands = []
            if results.hand_landmarks:
                for i, lms in enumerate(results.hand_landmarks):
                    h = HandState(detected=True)
                    hd = results.handedness[i] if i < len(results.handedness) else []
                    h.label = hd[0].category_name if hd else "?"
                    h.wrist_x, h.wrist_y, h.wrist_z = lms[0].x, lms[0].y, lms[0].z
                    h.palm_x, h.palm_y, h.palm_z = lms[9].x, lms[9].y, lms[9].z
                    h.palm_size = math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y)
                    h.fingers = [self._compute_finger_state(lms, fi) for fi in range(5)]
                    h.grab_strength = self._compute_grab_strength(lms)
                    h.pointing_down = lms[9].y > lms[0].y and all(lms[i].y > lms[9].y for i in [8, 12, 16])
                    h.landmarks_raw = lms
                    hands.append(h)

            with self._lock:
                self._hand_data = hands if hands else [HandState()]
                self._latest_frame = frame

            time.sleep(0.005)  # 防止CPU满载
