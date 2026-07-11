using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using UnityEngine;

public class HandDataReceiver : MonoBehaviour
{
    public int listenPort = 8888;
    public GameObject handPrefab;

    private UdpClient _udp;
    private Thread _recvThread;
    private HandsMessage _latestMsg;
    private readonly object _lock = new();

    private HandController _handLeft;
    private HandController _handRight;

    void Start()
    {
        _udp = new UdpClient(listenPort);
        _recvThread = new Thread(ReceiveLoop) { IsBackground = true };
        _recvThread.Start();
        Debug.Log($"[HandDataReceiver] Listening on :{listenPort}");
    }

    void Update()
    {
        HandsMessage msg;
        lock (_lock)
        {
            msg = _latestMsg;
            _latestMsg = null;
        }
        if (msg?.hands == null) return;

        foreach (var h in msg.hands)
        {
            if (h.landmarks == null || h.landmarks.Count < 21) continue;

            if (h.label == "Left")
            {
                if (_handLeft == null) _handLeft = GetOrCreateHand("LeftHand");
                if (_handLeft) _handLeft.UpdateFromData(h);
            }
            else if (h.label == "Right")
            {
                if (_handRight == null) _handRight = GetOrCreateHand("RightHand");
                if (_handRight) _handRight.UpdateFromData(h);
            }
        }
    }

    HandController GetOrCreateHand(string name)
    {
        var existing = GameObject.Find(name)?.GetComponent<HandController>();
        if (existing) return existing;

        var go = new GameObject(name);
        var hc = go.AddComponent<HandController>();
        if (handPrefab != null)
        {
            var inst = Instantiate(handPrefab, go.transform);
            inst.name = "Model";
        }
        return hc;
    }

    void ReceiveLoop()
    {
        var endpoint = new IPEndPoint(IPAddress.Any, listenPort);
        try
        {
            while (true)
            {
                var data = _udp.Receive(ref endpoint);
                var json = Encoding.UTF8.GetString(data);
                var msg = JsonConvert.DeserializeObject<HandsMessage>(json);
                lock (_lock) { _latestMsg = msg; }
            }
        }
        catch { }
    }

    void OnDestroy()
    {
        _recvThread?.Abort();
        _udp?.Close();
    }
}
