"""
手势数据采集器
录制手部关键点数据用于训练自定义手势模型

用法:
  1. 按数字键 0-9 切换到对应手势类别
  2. 按 SPACE 录制当前帧 (或按住连续录制)
  3. 按 S 保存数据集
  4. 按 ESC 退出
"""

import cv2
import os
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

from smart_finger import HandTracker, HandLandmarksConnections, HAND_LANDMARK_TIPS

DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 预定义手势类别
GESTURE_CLASSES = {
    0: "five_pinch",       # 五指合并
    1: "three_down",       # 三指向下
    2: "open_palm",        # 五指张开
    3: "fist",             # 握拳
    4: "point_up",         # 食指指上
    5: "ok",               # OK手势
    6: "victory",          # 剪刀手
    7: "thumb_up",         # 竖拇指
    8: "thumb_down",       # 倒拇指
    9: "rest",             # 休息/无手势
}


def extract_features(landmarks) -> np.ndarray:
    """从21个landmarks提取特征向量 (63维: 21×3归一化坐标)"""
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    return arr.flatten()


def draw_collector_ui(frame, current_class_id, class_name, count, total):
    h, w = frame.shape[:2]

    # 顶部状态栏
    cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(frame, f"Class [{current_class_id}]: {class_name}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frames: {count}  Total: {total}",
                (w - 250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

    # 手势类别提示
    for i in range(10):
        name = GESTURE_CLASSES[i][:12]
        color = (0, 255, 0) if i == current_class_id else (150, 150, 150)
        y0 = 80 + i * 25
        cv2.putText(frame, f"[{i}] {name}", (w - 160, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 操作提示
    cv2.putText(frame, "SPACE: record  H: hold-rec  S: save  ESC: exit",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    tracker = HandTracker(max_hands=2, detection_confidence=0.8)
    dataset = defaultdict(list)
    current_class = 0
    frame_count = 0
    hold_mode = False

    print("=" * 50)
    print("  手势数据采集器")
    print("=" * 50)
    print("  数字键 0-9: 切换手势类别")
    print("  SPACE: 录制当前帧")
    print("  H: 切换连续录制模式")
    print("  S: 保存数据集到文件")
    print("  ESC: 退出")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        hands = tracker.process(frame)

        # 显示手部关键点
        for hand in hands:
            lms = hand.landmarks
            h, w = frame.shape[:2]
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                s, e = lms[conn.start], lms[conn.end]
                cv2.line(frame, (int(s.x*w), int(s.y*h)),
                         (int(e.x*w), int(e.y*h)), (0, 200, 100), 2)
            for i, lm in enumerate(lms):
                cx, cy = int(lm.x*w), int(lm.y*h)
                color = (50, 200, 50) if i in HAND_LANDMARK_TIPS else (200, 180, 80)
                cv2.circle(frame, (cx, cy), 4, color, -1)

        class_name = GESTURE_CLASSES[current_class]
        total = sum(len(v) for v in dataset.values())
        count = len(dataset[class_name])
        draw_collector_ui(frame, current_class, class_name, count, total)

        cv2.imshow("Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif ord('0') <= key <= ord('9'):
            current_class = key - ord('0')
            print(f"  切换到类别 {current_class}: {GESTURE_CLASSES[current_class]}")
        elif key == ord('h'):
            hold_mode = not hold_mode
            print(f"  连续录制: {'ON' if hold_mode else 'OFF'}")
        elif key == ord(' '):  # SPACE
            if hands:
                for hand in hands:
                    feat = extract_features(hand.landmarks)
                    dataset[class_name].append({
                        "features": feat.tolist(),
                        "label": class_name,
                        "handedness": hand.label,
                        "timestamp": datetime.now().isoformat(),
                    })
                count = len(dataset[class_name])
                print(f"  [{class_name}] 录制 {len(hands)} 只手 (共{count}帧)")
            else:
                print("  未检测到手")
        elif key == ord(' '):
            pass
        elif key == ord('s'):
            save_dataset(dataset)
            print(f"  数据集已保存! 共{total}帧, {len(dataset)}个类别")

        # 连续录制模式
        if hold_mode and hands:
            for hand in hands:
                feat = extract_features(hand.landmarks)
                dataset[class_name].append({
                    "features": feat.tolist(),
                    "label": class_name,
                    "handedness": hand.label,
                    "timestamp": datetime.now().isoformat(),
                })

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # 退出时自动保存
    total = sum(len(v) for v in dataset.values())
    if total > 0:
        save_dataset(dataset)
        print(f"  退出时自动保存: 共{total}帧")


def save_dataset(dataset):
    path = os.path.join(DATA_DIR, f"gesture_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # 转换为可序列化格式
    serializable = {}
    for cls_name, samples in dataset.items():
        serializable[cls_name] = samples
    with open(path, "w") as f:
        json.dump(serializable, f, indent=1)
    print(f"  已保存: {path} ({sum(len(v) for v in serializable.values())} 帧)")


if __name__ == "__main__":
    main()
