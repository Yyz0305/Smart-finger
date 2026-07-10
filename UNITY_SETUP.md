# Unity 项目设置指南

## 架构

```
Python (手部追踪)                    Unity (3D 渲染)
┌─────────────────┐     UDP/JSON     ┌──────────────────┐
│  HandAPI         │ ──────────────→ │  HandDataReceiver │
│  HandBroadcaster │    port 8888    │  (C#)            │
└─────────────────┘                 └────────┬─────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   HandController    GrabbableObject     SceneSetup
                   (21关节+骨骼)      (抓取/释放)        (场景搭建)
```

## 快速开始

### 1. 启动 Python 广播端

```bash
cd Smart\ Finger
python hand_broadcaster.py
```

默认 UDP 广播到 `127.0.0.1:8888`, 每秒 ~60 帧。

### 2. 打开 Unity 项目

1. 启动 Unity Hub → 添加项目 → 选择 `unity_project/` 目录
2. 用 Unity 2022.3 LTS 或更新版本打开
3. 等待导入完成

### 3. 运行场景

1. 在 `Project` 窗口找到 `Scenes` (或新建场景)
2. 创建空 GameObject 命名为 `GameManager`
3. 挂载以下脚本:
   - `HandDataReceiver` (UDP 监听)
   - `GestureInput` (手势交互)
   - `SceneSetup` (场景生成)
4. 按 Play 按钮运行

### 4. 交互操作

- 伸出手在摄像头前 → 3D 手部实时跟随
- 握拳/五指合并靠近物体 → 抓起
- 张开手掌 → 释放物体
- 按 C 校准深度

## 文件说明

| 文件 | 说明 |
|------|------|
| `hand_broadcaster.py` | Python UDP 广播 |
| `HandDataReceiver.cs` | Unity UDP 接收 |
| `HandController.cs` | 21关节手部驱动 |
| `GrabbableObject.cs` | 可抓取物体基类 |
| `GestureInput.cs` | 手势→交互映射 |
| `SceneSetup.cs` | 场景自动搭建 |
