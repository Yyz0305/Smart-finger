using UnityEngine;

/// <summary>
/// 可抓取物体 — 收到 grab/release 信号后吸附/离开手部
/// </summary>
public class GrabbableObject : MonoBehaviour
{
    [Header("抓取设置")]
    public float grabDistance = 0.4f;
    public float lerpSpeed = 15f;
    public float releaseHeight = 0.5f;

    [HideInInspector] public bool IsGrabbed { get; private set; }
    private Transform _grabber;
    private Vector3 _offset;
    private Vector3 _startPos;

    void Start()
    {
        _startPos = transform.position;
    }

    void Update()
    {
        if (IsGrabbed && _grabber != null)
        {
            Vector3 target = _grabber.position + _offset;
            transform.position = Vector3.Lerp(transform.position, target, Time.deltaTime * lerpSpeed);
        }
    }

    public bool TryGrab(Transform handRoot)
    {
        if (IsGrabbed) return false;
        float dist = Vector3.Distance(transform.position, handRoot.position);
        if (dist > grabDistance) return false;

        IsGrabbed = true;
        _grabber = handRoot;
        _offset = transform.position - handRoot.position;
        return true;
    }

    public void Release()
    {
        if (!IsGrabbed) return;
        IsGrabbed = false;
        _grabber = null;
        // 物理掉落
        var rb = GetComponent<Rigidbody>();
        if (rb != null) rb.isKinematic = false;
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(transform.position, grabDistance);
    }
}
