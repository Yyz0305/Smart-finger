using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 手部控制器 — 接收 21 个 landmarks, 重建 3D 手部
/// </summary>
public class HandController : MonoBehaviour
{
    class Joint
    {
        public GameObject go;
        public Transform t;
        public float radius;
    }

    private Joint[] _joints = new Joint[21];
    private LineRenderer _line;

    void Awake()
    {
        float[] radii = {
            0.020f, 0.016f, 0.014f, 0.012f, 0.008f,
            0.018f, 0.014f, 0.012f, 0.008f,
            0.020f, 0.015f, 0.013f, 0.008f,
            0.018f, 0.014f, 0.012f, 0.008f,
            0.015f, 0.012f, 0.010f, 0.007f,
        };

        // Create 21 joint spheres
        for (int i = 0; i < 21; i++)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.transform.SetParent(transform);
            var j = new Joint { go = go, t = go.transform, radius = radii[i] };
            Destroy(go.GetComponent<Collider>());
            _joints[i] = j;
        }

        // Line renderer for connections
        _line = gameObject.AddComponent<LineRenderer>();
        _line.startWidth = 0.008f;
        _line.endWidth = 0.004f;
        _line.material = new Material(Shader.Find("Sprites/Default"));
        _line.startColor = new Color(0.94f, 0.78f, 0.63f);
        _line.endColor = new Color(0.88f, 0.66f, 0.51f);
        _line.positionCount = 0;
    }

    public void UpdateFromData(HandData data)
    {
        if (data.landmarks == null || data.landmarks.Count < 21) return;

        // Map normalized coords to world space
        float scale = 1.2f;
        float depth = 1.5f;
        Vector3[] pts = new Vector3[21];

        for (int i = 0; i < 21; i++)
        {
            var lm = data.landmarks[i];
            float x = (lm[0] - 0.5f) * scale;
            float y = -(lm[1] - 0.5f) * scale;
            float z = lm[2] * 0.3f * scale - depth;
            pts[i] = new Vector3(x, y + 0.5f, z);
        }

        // Smooth follow wrist
        Vector3 wrist = pts[0];
        transform.position = Vector3.Lerp(transform.position, wrist, Time.deltaTime * 15f);

        // Update joints
        for (int i = 0; i < 21; i++)
        {
            Vector3 localPos = pts[i] - wrist;
            _joints[i].t.localPosition = localPos;
            float r = _joints[i].radius;
            _joints[i].t.localScale = Vector3.one * r * 2f;
        }

        // Draw bones as line strip
        DrawBones(pts);
    }

    void DrawBones(Vector3[] pts)
    {
        int[][] bones = {
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 },
            new[] { 0, 5 }, new[] { 5, 6 }, new[] { 6, 7 }, new[] { 7, 8 },
            new[] { 0, 9 }, new[] { 9, 10 }, new[] { 10, 11 }, new[] { 11, 12 },
            new[] { 0, 13 }, new[] { 13, 14 }, new[] { 14, 15 }, new[] { 15, 16 },
            new[] { 0, 17 }, new[] { 17, 18 }, new[] { 18, 19 }, new[] { 19, 20 },
            new[] { 5, 9 }, new[] { 9, 13 }, new[] { 13, 17 },
        };

        _line.positionCount = bones.Length * 2;
        int idx = 0;
        foreach (var b in bones)
        {
            _line.SetPosition(idx++, pts[b[0]]);
            _line.SetPosition(idx++, pts[b[1]]);
        }
    }
}
