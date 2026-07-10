using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// UDP 接收手部数据, 通过事件分发到各手部控制器
/// </summary>
public class HandDataReceiver : MonoBehaviour
{
    [Header("网络设置")]
    public int listenPort = 8888;

    [Header("手部预制体")]
    public GameObject leftHandPrefab;
    public GameObject rightHandPrefab;

    private UdpClient _udp;
    private Thread _recvThread;
    private HandsMessage _latestMsg;
    private readonly object _lock = new();

    public Action<HandData> OnLeftHandUpdate;
    public Action<HandData> OnRightHandUpdate;

    private HandController _leftCtrl;
    private HandController _rightCtrl;

    void Start()
    {
        _udp = new UdpClient(listenPort);
        _recvThread = new Thread(ReceiveLoop) { IsBackground = true };
        _recvThread.Start();
        Debug.Log($"[HandDataReceiver] UDP 监听 :{listenPort}");
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
            if (h.label == "Left")
            {
                if (_leftCtrl == null && leftHandPrefab != null)
                {
                    var go = Instantiate(leftHandPrefab);
                    _leftCtrl = go.GetComponent<HandController>();
                }
                OnLeftHandUpdate?.Invoke(h);
                if (_leftCtrl) _leftCtrl.UpdateFromData(h);
            }
            else if (h.label == "Right")
            {
                if (_rightCtrl == null && rightHandPrefab != null)
                {
                    var go = Instantiate(rightHandPrefab);
                    _rightCtrl = go.GetComponent<HandController>();
                }
                OnRightHandUpdate?.Invoke(h);
                if (_rightCtrl) _rightCtrl.UpdateFromData(h);
            }
        }
    }

    void ReceiveLoop()
    {
        var endpoint = new IPEndPoint(IPAddress.Any, listenPort);
        while (true)
        {
            try
            {
                var data = _udp.Receive(ref endpoint);
                var json = Encoding.UTF8.GetString(data);
                var msg = JsonUtility.FromJson<HandsMessage>(json);
                lock (_lock) { _latestMsg = msg; }
            }
            catch { break; }
        }
    }

    void OnDestroy()
    {
        _recvThread?.Abort();
        _udp?.Close();
    }
}
