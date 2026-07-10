using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 手势输入 — 根据手部数据触发抓取/释放
/// 挂载到场景中, 监听 HandDataReceiver 事件
/// </summary>
public class GestureInput : MonoBehaviour
{
    [Header("场景物体")]
    public List<GrabbableObject> grabbableObjects;

    private HandDataReceiver _receiver;
    private Dictionary<string, bool> _prevGrab = new();

    void Start()
    {
        _receiver = FindObjectOfType<HandDataReceiver>();
        if (_receiver != null)
        {
            _receiver.OnRightHandUpdate += OnHandUpdate;
            _receiver.OnLeftHandUpdate += OnHandUpdate;
        }
    }

    void OnHandUpdate(HandData data)
    {
        if (data == null) return;

        string key = data.label;
        bool nowGrab = data.grab_strength > 0.55f;
        bool prevGrab = _prevGrab.ContainsKey(key) && _prevGrab[key];

        // 找到对应手的根 Transform
        Transform handRoot = FindHandRoot(data.label);
        if (handRoot == null) return;

        if (nowGrab && !prevGrab)
        {
            // 尝试抓取最近的物体
            foreach (var obj in grabbableObjects)
            {
                if (!obj.IsGrabbed && obj.TryGrab(handRoot))
                {
                    Debug.Log($"[Grab] {obj.name} by {data.label}");
                    break;
                }
            }
        }
        else if (!nowGrab && prevGrab)
        {
            // 释放所有被这只手抓住的物体
            foreach (var obj in grabbableObjects)
            {
                if (obj.IsGrabbed) obj.Release();
            }
        }

        _prevGrab[key] = nowGrab;
    }

    Transform FindHandRoot(string label)
    {
        var hands = FindObjectsOfType<HandController>();
        foreach (var h in hands)
        {
            // 通过手部位置区分 (简单实现)
            return h.transform;
        }
        return null;
    }
}
