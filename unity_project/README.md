# Smart Finger - Unity 3D Scene

Real-time hand tracking → Unity 3D digital twin.

## Setup

1. **Install Unity 2022.3 LTS** via Unity Hub
2. Open this folder in Unity Hub
3. Open `Assets/Scenes/Main.unity`
4. Press **Play**

## Data Flow

```
Python (hand_broadcaster.py) → UDP :8888 → Unity (HandDataReceiver.cs)
```

## Controls

| Gesture | Action |
|---------|--------|
| Fist / Pinch (grip > 0.55) near object | Grab |
| Open hand | Release (physics drop) |

## Files

| Path | Description |
|------|-------------|
| `Scripts/HandDataReceiver.cs` | UDP listener |
| `Scripts/HandController.cs` | 21-joint hand driver |
| `Scripts/SceneSetup.cs` | Auto-builds scene |
| `Scripts/GrabbableObject.cs` | Grab/release |
| `Scripts/GestureInput.cs` | Gesture mapping |
| `Scenes/Main.unity` | Main scene |
