"""
Hand Broadcaster — 通过 UDP 将手部数据发送到 Unity/Unreal

在 HandAPI 的基础上, 以 JSON 格式广播手部数据到指定端口。
Unity 侧通过 UDP 接收并驱动 3D 手部模型。

用法:
    broadcaster = HandBroadcaster(port=8888)
    broadcaster.start()  # 自动启动 HandAPI 和广播线程
    # ...
    broadcaster.stop()
"""

import json
import socket
import threading
import time
from hand_api import HandAPI


class HandBroadcaster:
    def __init__(self, port=8888, camera_id=0, max_hands=2, host="127.0.0.1"):
        self.host = host
        self.port = port
        self.api = HandAPI(camera_id=camera_id, max_hands=max_hands)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = False
        self._thread = None
        self._fps_counter = 0
        self._fps_timer = 0

    def start(self):
        self.api.start()
        self._running = True
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()
        print(f"[HandBroadcaster] UDP -> {self.host}:{self.port}")
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.api.stop()
        self._sock.close()
        print("[HandBroadcaster] 已停止")

    def _broadcast_loop(self):
        while self._running:
            hands = self.api.get_hand_data()
            data_list = []
            for h in hands:
                if not h.detected or not h.landmarks_raw:
                    continue
                lm_list = [[lm.x, lm.y, lm.z] for lm in h.landmarks_raw]
                fingers = []
                for f in h.fingers:
                    fingers.append({
                        "extended": f.extended,
                        "angle_deg": f.angle_deg,
                        "curl_deg": f.curl_deg,
                        "tip": [f.tip_x, f.tip_y, f.tip_z],
                    })
                data_list.append({
                    "label": h.label,
                    "wrist": [h.wrist_x, h.wrist_y, h.wrist_z],
                    "palm": [h.palm_x, h.palm_y, h.palm_z],
                    "palm_size": h.palm_size,
                    "grab_strength": h.grab_strength,
                    "pointing_down": h.pointing_down,
                    "landmarks": lm_list,
                    "fingers": fingers,
                })

            if data_list:
                payload = json.dumps({"hands": data_list, "ts": time.time()})
                try:
                    self._sock.sendto(payload.encode(), (self.host, self.port))
                except Exception:
                    pass

            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer > 1.0:
                self._fps_timer = now
                self._fps_counter = 0

            time.sleep(0.01)

    @property
    def fps(self):
        return self._fps_counter


if __name__ == "__main__":
    bc = HandBroadcaster(port=8888)
    bc.start()
    print("广播中... Ctrl+C 停止")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bc.stop()
