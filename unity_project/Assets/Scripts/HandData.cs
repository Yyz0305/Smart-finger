using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class HandData
{
    public string label;
    public float[] wrist;
    public float[] palm;
    public float palm_size;
    public float grab_strength;
    public bool pointing_down;
    public float[][] landmarks;   // 21 x [x, y, z]
    public FingerData[] fingers;
}

[Serializable]
public class FingerData
{
    public bool extended;
    public float angle_deg;
    public float curl_deg;
    public float[] tip;
}

[Serializable]
public class HandsMessage
{
    public List<HandData> hands;
    public double ts;
}
