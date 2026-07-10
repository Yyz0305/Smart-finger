using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 手部控制器 — 利用 21 个关节位置驱动 3D 手部模型
/// 
/// 层次结构:
///   Hand (root)
///   ├─ Palm (球体)
///   ├─ Joint_0 (手腕)
///   ├─ Joint_1 ~ Joint_20 (关节球体)
///   ├─ Bone_0_1 ~ Bone_19_20 (骨骼圆柱)
///   └─ Finger_0_Tip ~ Finger_4_Tip (指尖)
/// </summary>
public class HandController : MonoBehaviour
{
    [Header("关节设置")]
    public float jointRadius = 0.015f;
    public float tipRadius = 0.010f;
    public Material skinMaterial;
    public Material tipMaterial;

    [Header("骨骼设置")]
    public float boneRadius = 0.012f;
    public Material boneMaterial;

    // MediaPipe 21 关键点索引
    private static readonly int[][] BONES = {
        new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 },
        new[] { 0, 5 }, new[] { 5, 6 }, new[] { 6, 7 }, new[] { 7, 8 },
        new[] { 0, 9 }, new[] { 9, 10 }, new[] { 10, 11 }, new[] { 11, 12 },
        new[] { 0, 13 }, new[] { 13, 14 }, new[] { 14, 15 }, new[] { 15, 16 },
        new[] { 0, 17 }, new[] { 17, 18 }, new[] { 18, 19 }, new[] { 19, 20 },
        new[] { 5, 9 }, new[] { 9, 13 }, new[] { 13, 17 },
    };

    // 手指粗细 (根部→尖端)
    private static readonly float[] JOINT_RADII = {
        0.020f, 0.016f, 0.014f, 0.012f, 0.010f,
        0.018f, 0.014f, 0.012f, 0.010f,
        0.020f, 0.015f, 0.013f, 0.010f,
        0.018f, 0.014f, 0.012f, 0.009f,
        0.015f, 0.012f, 0.010f, 0.008f,
    };

    private Transform[] _joints = new Transform[21];
    private Transform[] _bones;
    private Transform _palm;
    private Transform _thumbPad;

    void Awake()
    {
        // 手掌
        var palmGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        palmGO.transform.SetParent(transform);
        palmGO.transform.localScale = Vector3.one * 0.06f;
        palmGO.GetComponent<Renderer>().material = skinMaterial;
        _palm = palmGO.transform;

        // 21 个关节
        for (int i = 0; i < 21; i++)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.transform.SetParent(transform);
            float r = JOINT_RADII[i];
            go.transform.localScale = Vector3.one * r * 2;
            go.GetComponent<Renderer>().material = i switch
            {
                4 or 8 or 12 or 16 or 20 => tipMaterial,
                _ => skinMaterial,
            };
            _joints[i] = go.transform;
        }

        // 骨骼 (圆柱)
        _bones = new Transform[BONES.Length];
        for (int i = 0; i < BONES.Length; i++)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.transform.SetParent(transform);
            go.GetComponent<Renderer>().material = boneMaterial;
            _bones[i] = go.transform;
        }
    }

    public void UpdateFromData(HandData data)
    {
        if (data.landmarks == null || data.landmarks.Length < 21) return;

        // 将 MediaPipe 归一化坐标映射到 Unity 世界坐标
        // 使用 DepthCalibrator 逻辑: 基于手掌大小估算深度
        float depth = 1.5f;
        float scale = depth * 1.2f;

        // 计算 21 个 3D 位置
        Vector3[] positions = new Vector3[21];
        for (int i = 0; i < 21; i++)
        {
            float x = (data.landmarks[i][0] - 0.5f) * scale;
            float y = -(data.landmarks[i][1] - 0.5f) * scale;
            float z = data.landmarks[i][2] * 0.3f * scale - depth;
            positions[i] = new Vector3(x, y + 0.5f, z);
        }

        // 手腕位置 = 整体偏移
        Vector3 wristPos = positions[0];
        transform.position = Vector3.Lerp(transform.position, wristPos, Time.deltaTime * 15f);

        // 更新关节本地位置
        for (int i = 0; i < 21; i++)
        {
            _joints[i].localPosition = positions[i] - wristPos;
        }

        // 手掌在指根中心
        Vector3 palmCenter = Vector3.zero;
        foreach (int i in new[] { 5, 9, 13, 17 })
            palmCenter += _joints[i].localPosition;
        _palm.localPosition = palmCenter / 4f;

        // 更新骨骼 (圆柱对齐两点)
        for (int i = 0; i < BONES.Length && i < _bones.Length; i++)
        {
            int a = BONES[i][0], b = BONES[i][1];
            AlignBone(_bones[i], _joints[a].localPosition, _joints[b].localPosition);
        }
    }

    void AlignBone(Transform bone, Vector3 p1, Vector3 p2)
    {
        Vector3 dir = p2 - p1;
        float len = dir.magnitude;
        if (len < 0.001f) { bone.gameObject.SetActive(false); return; }
        bone.gameObject.SetActive(true);
        bone.position = transform.TransformPoint((p1 + p2) / 2f);
        bone.localScale = new Vector3(boneRadius * 2, len / 2f, boneRadius * 2);
        bone.up = dir.normalized;
    }
}
