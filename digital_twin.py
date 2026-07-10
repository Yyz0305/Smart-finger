"""
Digital Twin v4 — 高精度手部建模 + 完善3D场景

手部: 21关节 + 20骨骼 (椭球体, 肤色渐变, 指甲)
场景: 墙壁 + 桌面 + 地毯 + 水杯 + 盆栽
交互: 比例法深度校准 + 全空间移动 + 抓取水杯
"""

from hand_api import HandAPI, DepthCalibrator, FINGER_NAMES
from ursina import *
from ursina.shaders import lit_with_shadows_shader, unlit_shader
import math

# ─── 场景 ────────────────────────────────────────────────────
app = Ursina(borderless=False, title='Smart Finger - Digital Twin')
camera.position = (0, 0.8, -1.8)
camera.rotation = (8, 0, 0)

# 灯光
DirectionalLight(shadow_map_resolution=(2048, 2048), direction=(-1, -2, -1))
Entity(type='ambient_light', color=color.hex('#444444'))

# 墙壁
wall = Entity(model='quad', scale=(8, 4), position=(0, 2, -2.5),
              color=color.hex('#F0EDE5'), texture='white_cube', texture_scale=(8, 4),
              shader=unlit_shader, double_sided=True)

# 地板
floor = Entity(model='plane', scale=8, color=color.hex('#C4A882'),
               texture_scale=(4, 4), shader=lit_with_shadows_shader)

# 地毯
rug = Entity(model='plane', scale=(2.5, 1.8), position=(0, 0.01, 0.3),
             color=color.hex('#7B5B3A'), texture_scale=(1, 1),
             shader=lit_with_shadows_shader)

# 桌面 (圆角效果用两层)
table_top = Entity(model='cube', scale=(2.2, 0.06, 1.6), position=(0, 0.9, 0),
                   color=color.hex('#A0724A'), shader=lit_with_shadows_shader)
table_edge = Entity(model='cube', scale=(2.0, 0.08, 1.4), position=(0, 0.86, 0),
                    color=color.hex('#8B5E3C'), shader=lit_with_shadows_shader)
# 桌腿
for x, z in [(-1.0, -0.7), (-1.0, 0.7), (1.0, -0.7), (1.0, 0.7)]:
    Entity(model='cube', scale=(0.06, 0.86, 0.06), position=(x, 0.43, z),
           color=color.hex('#6B4226'), shader=lit_with_shadows_shader)

# 小盆栽
pot_base = Entity(model='cube', scale=(0.10, 0.08, 0.10), position=(0.7, 0.95, 0.5),
                  color=color.hex('#D2691E'), shader=lit_with_shadows_shader)
pot_base2 = Entity(model='cube', scale=(0.12, 0.04, 0.12), position=(0.7, 0.90, 0.5),
                   color=color.hex('#B85C1A'), shader=lit_with_shadows_shader)
pot_plant = Entity(model='sphere', scale=(0.10, 0.08, 0.10), position=(0.7, 1.05, 0.5),
                   color=color.hex('#228B22'), shader=lit_with_shadows_shader)

# ─── 关节连接对 ──────────────────────────────────────────────
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 手指各段粗细 (起点→终点 半径)
SKIN = '#F0C8A0'


# ─── 虚拟手 (Metaball风格重叠球体) ──────────────────────────
class HandModel(Entity):
    """使用重叠球体模拟真实手部形状"""

    def __init__(self):
        super().__init__()
        # 手掌主体
        self.palm = Entity(parent=self, model='sphere',
                           scale=(0.065, 0.022, 0.050),
                           color=color.hex('#F0C8A0'),
                           shader=lit_with_shadows_shader)
        # 拇指肌肉球
        self.thumb_pad = Entity(parent=self, model='sphere',
                                scale=(0.025, 0.020, 0.020),
                                color=color.hex('#F0C8A0'),
                                shader=lit_with_shadows_shader)

        # 21个关节球体 (不同大小模拟手指粗细)
        joint_radius = {
            0: 0.020,   # 手腕
            1: 0.016, 2: 0.014, 3: 0.012, 4: 0.010,  # 拇指
            5: 0.018, 6: 0.014, 7: 0.012, 8: 0.010,  # 食指
            9: 0.020, 10: 0.015, 11: 0.013, 12: 0.010,  # 中指
            13: 0.018, 14: 0.014, 15: 0.012, 16: 0.009,  # 无名
            17: 0.015, 18: 0.012, 19: 0.010, 20: 0.008,  # 小指
        }
        self.joints = {}
        for i in range(21):
            r = joint_radius.get(i, 0.010)
            col = '#E8C0A0' if i in (4, 8, 12, 16, 20) else '#F0C8A0'
            j = Entity(parent=self, model='sphere', scale=r * 2,
                       color=color.hex(col), shader=lit_with_shadows_shader)
            self.joints[i] = j

        # 手指中间加球体使手指更饱满
        self.extra_joints = {}
        extra_segments = [
            (1.5, 0.013), (2.5, 0.012),   # 拇指
            (5.5, 0.014), (6.5, 0.012),   # 食指
            (9.5, 0.015), (10.5, 0.013),  # 中指
            (13.5, 0.014), (14.5, 0.012), # 无名
            (17.5, 0.011), (18.5, 0.010), # 小指
        ]
        for idx_float, r in extra_segments:
            e = Entity(parent=self, model='sphere', scale=r * 2,
                       color=color.hex('#F0C8A0'), shader=lit_with_shadows_shader)
            self.extra_joints[idx_float] = e

        # 指甲
        self.nails = {}
        for ti in [8, 12, 16]:
            n = Entity(parent=self, model='quad', scale=(0.006, 0.005),
                       color=color.hex('#FFE8D0'), shader=unlit_shader,
                       billboard=True, double_sided=True)
            self.nails[ti] = n

        self.prev_pts = None

    def update_from_landmarks(self, pts):
        if not pts or len(pts) < 21:
            return
        if self.prev_pts is not None:
            sm = 0.3
            pts = [p * sm + q * (1 - sm) for p, q in zip(self.prev_pts, pts)]
        self.prev_pts = pts[:]

        wrist = pts[0]
        self.position = wrist
        rel = [p - wrist for p in pts]

        # 手掌位于指根中心
        roots = [rel[i] for i in [5, 9, 13, 17]]
        c = Vec3(0, 0, 0)
        for r in roots:
            c += r
        self.palm.position = c / 4
        # 拇指肌肉在拇指根部
        self.thumb_pad.position = rel[1] * 0.6 + rel[2] * 0.4

        # 关节球体
        for i, j in self.joints.items():
            if i < len(rel):
                j.position = rel[i]

        # 中间填充球体 (线性插值)
        for key, e in self.extra_joints.items():
            a = int(key)
            b = a + 1
            t = key - a
            if a < len(rel) and b < len(rel):
                e.position = rel[a] * (1 - t) + rel[b] * t

        # 指甲
        for ti, n in self.nails.items():
            if ti < len(rel):
                n.position = rel[ti] + Vec3(0, 0.003, 0.001)


# ─── 3D 坐标映射 ────────────────────────────────────────────
def landmarks_to_3d(hand, calib):
    lms = hand.landmarks_raw
    if not lms or len(lms) < 21:
        return None
    wrist = lms[0]
    d = calib.estimate_distance(hand.palm_size)
    s = d * 2 * math.tan(math.radians(30))
    wx, wy, wz = calib.map_to_3d(wrist.x, wrist.y, hand.palm_size)
    base = Vec3(wx, wy + 0.3, -wz + 1.0)
    scale = d * 0.5
    pts = []
    for lm in lms:
        dx = (lm.x - wrist.x) * scale
        dy = -(lm.y - wrist.y) * scale
        dz = (lm.z - wrist.z) * 0.22 * scale
        pts.append(base + Vec3(dx, dy, dz))
    return pts


# ─── 可交互物体基类 ──────────────────────────────────────────
class Grabbable:
    """可抓取物体的基类 mixin"""

    def init_grab(self, root_pos):
        self._grabbed = False
        self._offset = Vec3(0, 0, 0)
        self._root_pos = root_pos
        self._parts = []

    def try_grab(self, hand_pos, hand_label="Right"):
        if (hand_pos - self._root_pos).length() < 0.6:
            self._grabbed = True
            self._offset = self._root_pos - hand_pos
            self._grabber = hand_label
            return True
        return False

    def try_release(self):
        if not self._grabbed:
            return False
        self._grabbed = False
        target_y = 0.92
        for e in self._parts:
            e.animate_position(
                Vec3(e.x, e.y - self._root_pos.y + target_y, e.z),
                duration=0.25,
            )
        return True

    def update_grab(self):
        if not self._grabbed:
            return
        hand_pos = hand_positions.get(self._grabber, Vec3(0, 0, 0))
        target = self._offset + hand_pos
        dt = time.dt * 18
        for e in self._parts:
            e.position = lerp(e.position, target + (e.position - self._root_pos), dt)

    @property
    def grabbed(self):
        return self._grabbed

    @property
    def root_pos(self):
        return self._root_pos


# ─── 水杯 ────────────────────────────────────────────────────
class Cup(Entity, Grabbable):
    def __init__(self, pos=(0, 1.15, 0)):
        super().__init__()
        col = color.hex('#4FC3F7')
        self.body = Entity(model='sphere', scale=(0.14, 0.18, 0.14),
                           position=pos, color=col, shader=lit_with_shadows_shader)
        self.body2 = Entity(model='sphere', scale=(0.14, 0.18, 0.14),
                            position=Vec3(pos) + Vec3(0, 0.30, 0),
                            color=col, shader=lit_with_shadows_shader)
        self.bottom = Entity(model='sphere', scale=(0.12, 0.02, 0.12),
                             position=Vec3(pos) - Vec3(0, 0.19, 0),
                             color=color.hex('#2E7D32'), shader=lit_with_shadows_shader)
        self.water = Entity(model='sphere', scale=(0.11, 0.02, 0.11),
                            position=Vec3(pos), color=color.hex('#81D4FA'),
                            shader=lit_with_shadows_shader)
        self.handle = Entity(model='cube', scale=(0.02, 0.12, 0.02),
                             position=Vec3(pos) + Vec3(0.16, 0.15, 0),
                             color=col, shader=lit_with_shadows_shader)
        self._parts = [self.body, self.body2, self.bottom, self.water, self.handle]
        self.init_grab(Vec3(pos))

    def update(self):
        if self._grabbed:
            self.update_grab()
            self.water.rotation_z = math.sin(time.time() * 5) * 3


# ─── 球 ──────────────────────────────────────────────────────
class Ball(Entity, Grabbable):
    def __init__(self, pos=(0.5, 1.0, 0.3)):
        super().__init__()
        self.body = Entity(model='sphere', scale=0.10, position=pos,
                           color=color.hex('#FF5252'), shader=lit_with_shadows_shader)
        self.stripe = Entity(model='cube', scale=(0.08, 0.01, 0.01),
                             position=Vec3(pos) + Vec3(0, 0, 0.06),
                             color=color.hex('#FFFFFF'), shader=lit_with_shadows_shader)
        self._parts = [self.body, self.stripe]
        self.init_grab(Vec3(pos))

    def update(self):
        if self._grabbed:
            self.update_grab()


# ─── 方块 ──────────────────────────────────────────────────────
class Block(Entity, Grabbable):
    def __init__(self, pos=(-0.5, 1.0, -0.2)):
        super().__init__()
        self.body = Entity(model='cube', scale=(0.12, 0.12, 0.12), position=pos,
                           color=color.hex('#FFC107'), shader=lit_with_shadows_shader)
        self._parts = [self.body]
        self.init_grab(Vec3(pos))

    def update(self):
        if self._grabbed:
            self.update_grab()


# ─── 手机 ────────────────────────────────────────────────────
class Phone(Entity, Grabbable):
    def __init__(self, pos=(0, 1.0, 0.6)):
        super().__init__()
        self.body = Entity(model='cube', scale=(0.06, 0.12, 0.005), position=pos,
                           color=color.hex('#1A1A1A'), shader=lit_with_shadows_shader)
        self.screen = Entity(model='quad', scale=(0.05, 0.10), position=Vec3(pos) + Vec3(0, 0, 0.004),
                             color=color.hex('#2196F3'), shader=unlit_shader, double_sided=True)
        self._parts = [self.body, self.screen]
        self.init_grab(Vec3(pos))

    def update(self):
        if self._grabbed:
            self.update_grab()


# ─── 创建所有物体 ────────────────────────────────────────────
objects = [
    Cup(pos=(0, 1.15, 0)),
    Ball(pos=(0.5, 1.0, 0.3)),
    Block(pos=(-0.5, 1.0, -0.2)),
    Phone(pos=(0, 1.0, 0.6)),
]


# ─── 初始化 ──────────────────────────────────────────────────
hand_api = HandAPI(max_hands=2)
hand_api.start()
calib = DepthCalibrator(ref_distance=0.45)

hand_left = HandModel()
hand_right = HandModel()
hand_left.color_shift = -0.02
hand_right.color_shift = 0.0

hand_positions = {"Left": Vec3(0, 0, 0), "Right": Vec3(0, 0, 0)}
grabbed_prev = {"Left": False, "Right": False}

Text(text='Smart Finger - Dual Hands', position=(-0.85, 0.48), scale=1.2, color=color.white)
status = Text(text='Waiting...', position=(-0.85, 0.44), scale=0.8, color=color.yellow)
hint = Text(text='[C] Calibrate  [Q] Quit', position=(-0.85, 0.40), scale=0.6, color=color.gray)


def update():
    hands = hand_api.get_hand_data()
    active = [h for h in hands if h.detected and h.landmarks_raw]

    if not active:
        status.text = 'No hands'
        hand_left.visible = False
        hand_right.visible = False
        return

    hand_left.visible = False
    hand_right.visible = False

    for h in active:
        hand_model = hand_left if h.label == "Left" else hand_right
        hand_model.visible = True
        pts = landmarks_to_3d(h, calib)
        if pts is None:
            continue
        hand_model.update_from_landmarks(pts)
        hand_positions[h.label] = pts[0]

        d = calib.estimate_distance(h.palm_size)
        status.text = f'{h.label}  Grip:{h.grab_strength:.2f}  Depth:{d:.2f}m'

        is_grab = h.grab_strength > 0.55
        hand_pos = pts[0]
        if is_grab and not grabbed_prev[h.label]:
            for obj in objects:
                if not obj.grabbed and obj.try_grab(hand_pos, h.label):
                    print(f"[Grab] {obj.__class__.__name__} by {h.label}")
                    break
        elif not is_grab and grabbed_prev[h.label]:
            for obj in objects:
                if obj.grabbed:
                    obj.try_release()
                    print(f"[Release] {obj.__class__.__name__}")
        grabbed_prev[h.label] = is_grab


def input(key):
    if key == 'c':
        for h in hand_api.get_hand_data():
            if h.detected and h.palm_size > 0.01:
                calib.calibrate(h.palm_size)
                break
    elif key == 'q':
        app.quit()


if __name__ == '__main__':
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        hand_api.stop()
