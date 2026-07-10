# Smart Finger — 环境部署指南

## 已安装

| 软件 | 状态 |
|------|------|
| Python 依赖 | ✅ 已完成 |
| MediaPipe 模型 | ✅ 已下载 |
| Unity Hub 3.19.4 | ✅ 已安装 |
| Ursina/Panda3D | ✅ 已安装 |

## 安装 Unity Editor

Unity Hub 已安装, 需手动安装 Editor 版本:

1. 启动 **Unity Hub**
2. 点击左侧 "Installs" → "Install Editor"
3. 选择 **2022.3 LTS** 或 **6000.0 LTS**
4. 勾选:
   - `Windows Build Support (IL2CPP)`
   - `Universal Windows Platform Build Support`
   - `Documentation`
5. 点击 Install (下载 ~5GB)

## 安装 Unreal Engine

1. 启动 **Epic Games Launcher** (安装完成后)
2. 登录 Epic 账号
3. 左侧 "Unreal Engine" → "Library"
4. 点击 "+" 选择版本 → 安装 UE5.4+

## 运行 Python → Unity 数据链路

```bash
# 终端 1: 启动手部追踪 + UDP 广播
cd Smart\ Finger
python hand_broadcaster.py
# 输出: [HandAPI] 已启动
#        [HandBroadcaster] UDP -> 127.0.0.1:8888

# 终端 2: 启动 Ursina 3D (快速预览)
python digital_twin.py

# 终端 3: Unity (安装完成后)
# 用 Unity Hub 打开 unity_project/ 目录
# 按 Play 运行
```

## 架构总览

```
                    UDP:8888 ──────────┐
                    JSON 格式           │
┌────────────────┐                     ▼  ┌──────────────────┐
│  HandAPI       │    ┌────────────┐      │  Unity 3D        │
│  +MediaPipe    │───→│broadcaster │─────→│  HandReceiver    │
│  21 landmarks  │    └────────────┘      │  → HandController│
└────────────────┘                        │  → SceneSetup    │
         │                                │  → Grabbable     │
         ▼                                └──────────────────┘
  ┌──────────────┐
  │ Ursina 3D    │ ← 快速原型 (Python)
  │ digital_twin │
  └──────────────┘
```

## 数据格式 (UDP JSON)

```json
{
  "hands": [{
    "label": "Left",
    "palm_size": 0.15,
    "grab_strength": 0.7,
    "landmarks": [[x,y,z], ...],  // 21×3
    "fingers": [{"extended": true, "angle_deg": 45, "curl_deg": 30}, ...]
  }],
  "ts": 1712345678.123
}
```
