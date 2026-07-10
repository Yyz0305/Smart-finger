"""
Digital Twin — 3D 数字孪生场景 (深度校准版)

使用 DepthCalibrator 做比例法深度推算:
  校准: 用户一臂长位置按 C, 记录参考掌尺寸
  实时: palm_size_ratio = ref / current → estimate_distance
  映射: x,y 根据距离做透视缩放, z = 估算深度

场景:
  - 桌面 + 方向光 + 阴影
  - 水杯 (可抓取)
  - 虚拟手 (5指跟随, 深度校准后的精确位置)
  - 按 C 校准 | 按 ESC 退出
"""

from hand_api import HandAPI, DepthCalibrator, FINGER_NAMES
from ursina import *
from ursina.shaders import lit_with_shadows_shader
import math


# ─── 场景 ────────────────────────────────────────────────────
app = Ursina(borderless=False, title='Smart Finger - Digital Twin (Depth Calibrated)')
camera.position = (0, 0.6, -1.5)
camera.rotation = (10, 0, 0)

DirectionalLight(shadow_map_resolution=(2048, 2048))

ground = Entity(model='plane', scale=6,
                color=color.hex('#D2B48C'), texture_scale=(3, 3),
                shader=lit_with_shadows_shader)
table = Entity(model='cube', scale=(2.0, 0.08, 1.5), position=(0, 0.9, 0),
               color=color.hex('#8B4513'), shader=lit_with_shadows_shader)


# ─── 水杯 ────────────────────────────────────────────────────
class Cup(Entity):
    def __init__(self):
        super().__init__(model='cube', scale=(0.25, 0.35, 0.25),
                         position=(0, 1.2, 0), color=color.hex('#4FC3F7'),
                         shader=lit_with_shadows_shader)
        Entity(parent=self, model='cube', scale=(0.27, 0.03, 0.27),
               position=(0, -0.19, 0), color=color.hex('#2E7D32'),
               shader=lit_with_shadows_shader)
        self.water = Entity(parent=self, model='cube', scale=(0.22, 0.04, 0.22),
                            position=(0, 0.08, 0), color=color.hex('#81D4FA'),
                            shader=lit_with_shadows_shader)
        self.grabbed = False
        self.grab_offset = Vec3(0, 0, 0)

    def grab(self, hand_pos):
        self.grabbed = True
        self.grab_offset = self.position - hand_pos

    def release(self):
        self.grabbed = False
        self.animate_position(Vec3(self.x, 0.92, self.z), duration=0.3)

    def update(self):
        global hand_position
        if self.grabbed:
            target = self.grab_offset + hand_position
            self.position = lerp(self.position, target, time.dt * 18)
            self.water.rotation_z = math.sin(time.time() * 5) * 2


cup = Cup()


# ─── 虚拟手 ──────────────────────────────────────────────────
class VirtualFinger(Entity):
    def __init__(self, name, col, parent):
        super().__init__(parent=parent, model='cube',
                         scale=(0.035, 0.10, 0.035),
                         color=color.hex(col), shader=lit_with_shadows_shader)
        self.tip = Entity(parent=self, model='sphere',
                          scale=0.045, position=(0, 0.07, 0),
                          color=color.hex(col))

    def set_curl(self, curl_factor, extended):
        self.rotation = (curl_factor * 90, 0, 0)
        v = 0.4 + 0.6 * (1 - curl_factor)
        self.tip.color = Color(0.2, v, 0.2, 1)


class VirtualHand(Entity):
    def __init__(self):
        super().__init__()
        Entity(parent=self, model='cube', scale=(0.10, 0.035, 0.08),
               color=color.hex('#FFDAB9'), shader=lit_with_shadows_shader)
        configs = [
            ("Thumb", '#E8B88A', Vec3(-0.06, 0.025, 0)),
            ("Index", '#FFDAB9', Vec3(-0.025, 0.025, -0.035)),
            ("Middle", '#FFDAB9', Vec3(0, 0.025, -0.04)),
            ("Ring", '#FFDAB9', Vec3(0.025, 0.025, -0.035)),
            ("Pinky", '#FFDAB9', Vec3(0.05, 0.025, -0.015)),
        ]
        self.fingers = []
        for name, col, pos in configs:
            f = VirtualFinger(name, col, self)
            f.position = pos
            self.fingers.append(f)

    def update_from_hand(self, hand, dt, calib: DepthCalibrator):
        if not hand or not hand.detected:
            self.visible = False
            return
        self.visible = True

        # 比例法深度映射 (校准后更精确)
        x, y, z = calib.map_to_3d(hand.palm_x, hand.palm_y, hand.palm_size)
        pos = Vec3(x, y + 0.3, -z + 1.0)
        self.position = lerp(self.position, pos, dt * 12)
        self.y = max(self.y, 0.2)

        for i, fs in enumerate(hand.fingers):
            if i < len(self.fingers):
                curl = max(0, min(1, fs.curl_deg / 120))
                self.fingers[i].set_curl(curl, fs.extended)


# ─── 初始化 ──────────────────────────────────────────────────
hand_api = HandAPI(max_hands=1)
hand_api.start()
calib = DepthCalibrator(ref_distance=0.45)

virtual_hand = VirtualHand()
hand_position = Vec3(0, 0, 0)
grabbed_prev = False

Text(text='Smart Finger - Digital Twin (Depth Calibrated)',
     position=(-0.85, 0.48), scale=1.2, color=color.white)
status = Text(text='Waiting for hand...', position=(-0.85, 0.44),
              scale=0.8, color=color.yellow)
calib_text = Text(text='Press C to calibrate depth',
                  position=(-0.85, 0.40), scale=0.6, color=color.gray)


# ─── 更新循环 ────────────────────────────────────────────────
def update():
    global hand_position, grabbed_prev

    hand = hand_api.get_primary_hand()
    if not hand or not hand.detected:
        status.text = 'No hand'
        virtual_hand.visible = False
        calib_text.text = 'Press C to calibrate depth'
        return

    virtual_hand.update_from_hand(hand, time.dt, calib)
    hand_position = virtual_hand.position

    dist = calib.estimate_distance(hand.palm_size)
    status.text = (
        f'{hand.label}  |  '
        f'Grip:{hand.grab_strength:.2f}  |  '
        f'Depth:{dist:.2f}m  |  '
        f'Palm:{hand.palm_size:.3f}'
    )

    ref_txt = f" (ref={calib.ref_palm_size:.3f})" if calib.calibrated else ""
    calib_text.text = f'Calibrated:{calib.calibrated}{ref_txt}  [C]校准'

    is_grabbed = hand.grab_strength > 0.55
    if is_grabbed and not grabbed_prev:
        if (virtual_hand.position - cup.position).length() < 0.5:
            cup.grab(virtual_hand.position)
    elif not is_grabbed and grabbed_prev and cup.grabbed:
        cup.release()
    grabbed_prev = is_grabbed


def input(key):
    if key == 'c':
        hand = hand_api.get_primary_hand()
        if hand and hand.detected and hand.palm_size > 0.01:
            calib.calibrate(hand.palm_size)
            print(f"[Calib] ref_palm_size={calib.ref_palm_size:.4f} at {calib.ref_distance}m")


# ─── 启动 ────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        hand_api.stop()
