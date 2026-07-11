using UnityEngine;
using System.Collections.Generic;

public class GestureInput : MonoBehaviour
{
    private HandDataReceiver _receiver;
    private readonly Dictionary<string, bool> _prevGrab = new();
    private readonly List<GrabbableObject> _objects = new();

    void Start()
    {
        _receiver = FindObjectOfType<HandDataReceiver>();
    }

    void Update()
    {
        // Find grabbable objects in scene
        _objects.Clear();
        _objects.AddRange(FindObjectsOfType<GrabbableObject>());

        // Find hand controllers
        var hands = FindObjectsOfType<HandController>();
        foreach (var hand in hands)
        {
            string key = hand.name;
            bool nowGrab = false;
            // Get grab state from most recent hand data via receiver
            // For simplicity, we check distance-based grab each frame

            bool prevGrab = _prevGrab.ContainsKey(key) && _prevGrab[key];

            // Check if any finger tips are close to objects (grab attempt)
            float closestDist = float.MaxValue;
            GrabbableObject closestObj = null;

            foreach (var obj in _objects)
            {
                if (obj.IsGrabbed) continue;
                float dist = Vector3.Distance(hand.transform.position, obj.transform.position);
                if (dist < obj.grabDistance && dist < closestDist)
                {
                    closestDist = dist;
                    closestObj = obj;
                }
            }

            // Simulate grab: if hand is close to object, grab it
            nowGrab = closestObj != null;

            if (nowGrab && !prevGrab && closestObj != null)
            {
                closestObj.TryGrab(hand.transform);
                Debug.Log($"[Grab] {closestObj.name} by {key}");
            }
            else if (!nowGrab && prevGrab)
            {
                // Release all grabbed by this hand
                foreach (var obj in _objects)
                {
                    if (obj.IsGrabbed) obj.Release();
                }
            }

            _prevGrab[key] = nowGrab;
        }
    }
}
