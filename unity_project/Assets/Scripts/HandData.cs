using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

[System.Serializable]
public class HandData
{
    public string label;
    public float palm_size;
    public float grab_strength;
    public bool pointing_down;
    public List<List<float>> landmarks;
    public List<FingerData> fingers;
}

[System.Serializable]
public class FingerData
{
    public bool extended;
    public float angle_deg;
    public float curl_deg;
}

[System.Serializable]
public class HandsMessage
{
    public List<HandData> hands;
    public double ts;
}
