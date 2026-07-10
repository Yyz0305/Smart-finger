"""
Digital Twin — 3D 数字孪生场景

通过 HandAPI 实时获取手部数据, 在 3D 场景中驱动虚拟手,
实现抓取/移动/释放物件的数字孪生交互。

场景:
  - 桌面 + 灯光 + 环境
  - 一个可抓取的水杯 (Cup)
  - 虚拟手 (5根手指跟随真实手势)
  - 抓取: 五指合并 / 握拳 → 水杯吸附到手上
  - 释放: 张开手掌 → 水杯掉落 (物理掉落)
"""

from hand_api import HandAPI, HandState, FINGER_NAMES

from ursina import *
from ursina.shaders import lit_with_shadows_shader
import math


# ─── 场景配置 ────────────────────────────────────────────────
app = Ursina(borderless=False, title='Smart Finger - Digital Twin')
camera.position = (0, 2, -4)
camera.rotation = (15, 0, 0)

# 灯光
DirectionalLight(shadow_map_resolution=(2048, 2048))
scene.fog_color = color.hex('#87CEEB')
scene.fog_density = 0.02

# 地面
ground = Entity(
    model='plane', scale=8, texture='white_cube',
    color=color.hex('#D2B48C'), texture_scale=(4, 4),
    shader=lit_with_shadows_shader,
)

# 桌面
table = Entity(
    model='cube', scale=(3, 0.1, 2), position=(0, 0.95, 0),
    color=color.hex('#8B4513'), shader=lit_with_shadows_shader,
)


# ─── 水杯 ────────────────────────────────────────────────────
class Cup(Entity):
    def __init__(self):
        # 杯身
        super().__init__(
            model='cube', scale=(0.3, 0.4, 0.3),
            position=(0, 1.3, 0),
            color=color.hex('#4FC3F7'),
            shader=lit_with_shadows_shader,
        )
        # 杯底装饰
        self.bottom = Entity(
            parent=self, model='cube', scale=(0.32, 0.03, 0.32),
            position=(0, -0.22, 0),
            color=color.hex('#2E7D32'), shader=lit_with_shadows_shader,
        )
        # 水面
        self.water = Entity(
            parent=self, model='cube', scale=(0.26, 0.05, 0.26),
            position=(0, 0.1, 0),
            color=color.hex('#81D4FA'), shader=lit_with_shadows_shader,
        )
        # 手柄
        self.handle = Entity(
            parent=self, model='cube', scale=(0.06, 0.2, 0.06),
            position=(0.2, 0.05, 0),
            color=color.hex('#4FC3F7'), shader=lit_with_shadows_shader,
        )
        self.handle2 = Entity(
            parent=self, model='cube', scale=(0.03, 0.06, 0.06),
            position=(0.23, 0.17, 0),
            color=color.hex('#4FC3F7'), shader=lit_with_shadows_shader,
        )

        self.grabbed = False
        self.grab_offset = Vec3(0, 0, 0)

    def grab(self, hand_pos):
        self.grabbed = True
        self.grab_offset = self.position - hand_pos

    def release(self):
        self.grabbed = False
        # 简单物理: 掉落
        self.animate_position(
            Vec3(self.x, 0.9, self.z),
            duration=0.3,
            curve=curve.linear,
        )

    def update(self):
        global hand_position
        if self.grabbed:
            target = self.grab_offset + hand_position
            self.position = lerp(self.position, target, time.dt * 15)
            self.water.rotation_z = math.sin(time.time() * 5) * 2


cup = Cup()


# ─── 虚拟手 ──────────────────────────────────────────────────
class VirtualFinger(Entity):
    def __init__(self, name, color_hex, parent_entity):
        super().__init__(
            parent=parent_entity,
            model='cube', scale=(0.04, 0.12, 0.04),
            color=color.hex(color_hex),
            shader=lit_with_shadows_shader,
        )
        self.tip = Entity(
            parent=self, model='sphere',
            scale=(0.05, 0.05, 0.05),
            position=(0, 0.08, 0),
            color=color.hex(color_hex),
        )

    def set_curl(self, curl_factor, extended):
        rot = curl_factor * 90
        self.rotation = (rot, 0, 0)
        color_val = 0.4 + 0.6 * (1 - curl_factor)
        self.tip.color = Color(0.2, color_val, 0.2, 1)


class VirtualHand(Entity):
    """虚拟手 — 由手掌 + 5根手指组成"""
    def __init__(self):
        super().__init__(position=(0, 2, 0.5))
        # 手掌
        self.palm = Entity(
            parent=self, model='cube',
            scale=(0.12, 0.04, 0.10),
            color=color.hex('#FFDAB9'),
            shader=lit_with_shadows_shader,
        )
        # 5根手指 (排列在手掌上)
        self.fingers = []
        finger_configs = [
            ("Thumb", '#E8B88A', Vec3(-0.07, 0.03, 0.0)),
            ("Index", '#FFDAB9', Vec3(-0.03, 0.03, -0.04)),
            ("Middle", '#FFDAB9', Vec3(0.0, 0.03, -0.05)),
            ("Ring", '#FFDAB9', Vec3(0.03, 0.03, -0.04)),
            ("Pinky", '#FFDAB9', Vec3(0.06, 0.03, -0.02)),
        ]
        for name, col, pos in finger_configs:
            f = VirtualFinger(name, col, self)
            f.position = pos
            # 手指默认朝上
            f.rotation = (0, 0, 0)
            self.fingers.append(f)

    def update_from_hand(self, hand: HandState, dt):
        """根据API数据驱动虚拟手"""
        if not hand or not hand.detected:
            self.visible = False
            return

        self.visible = True

        # 手部位置映射 (MediaPipe归一化坐标 → 3D场景坐标)
        # x: MediaPipe左右 → 3D左右
        # y: MediaPipe上下 → 3D上下
        # z: MediaPipe深度 → 3D前后
        px = (hand.palm_x - 0.5) * 2.0
        py = (1.0 - hand.palm_y) * 3.0 - 0.5
        pz = hand.palm_z * 2.0 - 0.5
        self.position = lerp(self.position, Vec3(px, py, pz), dt * 10)

        # 手指
        for i, finger_state in enumerate(hand.fingers):
            if i < len(self.fingers):
                vf = self.fingers[i]
                curl = max(0, min(1, finger_state.curl_deg / 120))
                vf.set_curl(curl, finger_state.extended)


# ─── 初始化 ──────────────────────────────────────────────────
virtual_hand = VirtualHand()
hand_api = HandAPI(max_hands=1)
hand_api.start()
hand_position = Vec3(0, 0, 0)
grabbed_prev = False

# UI文字
info_text = Text(
    text='Smart Finger - Digital Twin',
    position=(-0.85, 0.48), scale=1.2, color=color.white,
)
status_text = Text(
    text='Waiting for hand...',
    position=(-0.85, 0.44), scale=0.8, color=color.yellow,
)


# ─── 主更新循环 ──────────────────────────────────────────────
def update():
    global hand_position, grabbed_prev

    hand = hand_api.get_primary_hand()
    if not hand or not hand.detected:
        status_text.text = 'Hand not detected'
        virtual_hand.visible = False
        return

    virtual_hand.update_from_hand(hand, time.dt)
    hand_position = virtual_hand.position

    # 更新手位置供Cup使用
    is_grabbed = hand.grab_strength > 0.6

    if is_grabbed and not grabbed_prev:
        if (virtual_hand.position - cup.position).length() < 0.6:
            cup.grab(virtual_hand.position)
            status_text.text = 'Grabbing Cup!'
            status_text.color = color.green
    elif not is_grabbed and grabbed_prev:
        if cup.grabbed:
            cup.release()
            status_text.text = 'Released Cup'
            status_text.color = color.orange

    grabbed_prev = is_grabbed

    # 状态显示
    finger_states = " ".join([
        f"{FINGER_NAMES[i][0]}={'E' if f.extended else 'C'}"
        for i, f in enumerate(hand.fingers)
    ])
    status_text.text = (
        f'{hand.label}  |  '
        f'Grab:{hand.grab_strength:.2f}  |  '
        f'Pos:({hand_position.x:.2f}, {hand_position.y:.2f})  |  '
        f'{finger_states}'
    )


# ─── 启动 ────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        hand_api.stop()
