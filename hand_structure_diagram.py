import cv2
import numpy as np

W = 900
H = 700

cx, cy = W // 2, H // 2 + 30
base_y = cy + 100

LANDMARK_POS = {
    # 手腕
    0:  (cx, base_y),
    # 拇指
    1:  (cx - 110, base_y + 30),
    2:  (cx - 160, base_y - 20),
    3:  (cx - 200, base_y - 70),
    4:  (cx - 220, base_y - 130),
    # 食指
    5:  (cx - 40, base_y - 50),
    6:  (cx - 35, base_y - 120),
    7:  (cx - 30, base_y - 180),
    8:  (cx - 25, base_y - 240),
    # 中指
    9:  (cx + 10, base_y - 60),
    10: (cx + 10, base_y - 140),
    11: (cx + 10, base_y - 200),
    12: (cx + 10, base_y - 270),
    # 无名指
    13: (cx + 60, base_y - 55),
    14: (cx + 55, base_y - 130),
    15: (cx + 50, base_y - 190),
    16: (cx + 45, base_y - 245),
    # 小指
    17: (cx + 100, base_y - 40),
    18: (cx + 95, base_y - 100),
    19: (cx + 90, base_y - 150),
    20: (cx + 85, base_y - 200),
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 9), (9, 13), (13, 17),
]

FINGER_LABELS = [
    (0, "WRIST"),
    (4, "THUMB\nTIP"),
    (8, "INDEX\nTIP"),
    (12, "MIDDLE\nTIP"),
    (16, "RING\nTIP"),
    (20, "PINKY\nTIP"),
]

FINGER_NAMES = {
    0:  "Wrist",
    1:  "CMC",
    2:  "MCP",
    3:  "IP",
    4:  "TIP",
    5:  "MCP",
    6:  "PIP",
    7:  "DIP",
    8:  "TIP",
    9:  "MCP",
    10: "PIP",
    11: "DIP",
    12: "TIP",
    13: "MCP",
    14: "PIP",
    15: "DIP",
    16: "TIP",
    17: "MCP",
    18: "PIP",
    19: "DIP",
    20: "TIP",
}

JOINT_LABELS = {
    0:  "Wrist",
    4:  "Thumb TIP",
    8:  "Index TIP",
    12: "Middle TIP",
    16: "Ring TIP",
    20: "Pinky TIP",
}

# section colors
FINGER_COLORS = {
    "palm":   (180, 180, 180),
    "thumb":  (255, 100, 100),
    "index":  (100, 200, 255),
    "middle": (100, 255, 150),
    "ring":   (200, 200, 100),
    "pinky":  (255, 150, 255),
}

FINGER_MAP = {
    (0, 1): "palm", (1, 2): "thumb", (2, 3): "thumb", (3, 4): "thumb",
    (0, 5): "palm", (5, 6): "index", (6, 7): "index", (7, 8): "index",
    (9, 10): "middle", (10, 11): "middle", (11, 12): "middle",
    (13, 14): "ring", (14, 15): "ring", (15, 16): "ring",
    (17, 18): "pinky", (18, 19): "pinky", (19, 20): "pinky",
    (0, 17): "palm", (5, 9): "palm", (9, 13): "palm", (13, 17): "palm",
}


def draw_diagram():
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # title
    cv2.putText(img, "Hand Landmarks Structure  (MediaPipe 21 Key Points)",
                (W // 2 - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)

    # connections (lines)
    for a, b in CONNECTIONS:
        if a not in LANDMARK_POS or b not in LANDMARK_POS:
            continue
        pt1 = LANDMARK_POS[a]
        pt2 = LANDMARK_POS[b]
        finger = FINGER_MAP.get((a, b)) or FINGER_MAP.get((b, a), "palm")
        color = FINGER_COLORS[finger]
        cv2.line(img, pt1, pt2, color, 3)

    # landmarks (circles + numbers)
    for idx, pos in LANDMARK_POS.items():
        x, y = pos
        if idx == 0:
            cv2.circle(img, pos, 10, (80, 80, 200), -1)
        elif idx % 4 == 0:
            cv2.circle(img, pos, 10, (50, 150, 50), -1)
        else:
            cv2.circle(img, pos, 7, (200, 180, 80), -1)
            cv2.circle(img, pos, 7, (100, 100, 100), 2)

        # label
        label = f"{idx} {FINGER_NAMES[idx]}"
        off_x = -30 if idx in (4,) else (-40 if idx in (8, 12, 16, 20) else 12)
        off_y = 0
        cv2.putText(img, label, (x + off_x, y + off_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    # legend
    lx, ly = 30, 620
    legend_items = [
        ("Palm / Wrist", (180, 180, 180)),
        ("Thumb",        (255, 100, 100)),
        ("Index",        (100, 200, 255)),
        ("Middle",       (100, 255, 150)),
        ("Ring",         (200, 200, 100)),
        ("Pinky",        (255, 150, 255)),
        ("Fingertip (TIP)", (50, 150, 50)),
        ("Joint (MCP/PIP/DIP/IP/CMC)", (200, 180, 80)),
    ]
    for i, (text, color) in enumerate(legend_items):
        yy = ly + i * 18
        cv2.rectangle(img, (lx, yy - 6), (lx + 16, yy + 6), color, -1)
        if color == (50, 150, 50):
            cv2.circle(img, (lx + 8, yy), 5, color, -1)
        elif color == (200, 180, 80):
            cv2.circle(img, (lx + 8, yy), 5, (200, 180, 80), -1)
            cv2.circle(img, (lx + 8, yy), 5, (100, 100, 100), 1)
        else:
            cv2.line(img, (lx, yy), (lx + 16, yy), color, 3)
        cv2.putText(img, text, (lx + 22, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    cv2.imwrite("hand_structure_diagram.png", img)

    cv2.imshow("Hand Structure Diagram", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("已生成手部架构图: hand_structure_diagram.png")


if __name__ == "__main__":
    draw_diagram()
