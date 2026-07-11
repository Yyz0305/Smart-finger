using UnityEngine;

public static class AutoSetup
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void Init()
    {
        if (GameObject.Find("GameManager") != null) return;

        var gm = new GameObject("GameManager");
        gm.AddComponent<SceneSetup>();
        gm.AddComponent<HandDataReceiver>();
        gm.AddComponent<GestureInput>();

        if (Camera.main != null)
        {
            Camera.main.transform.position = new Vector3(0, 0.8f, -2.5f);
            Camera.main.clearFlags = CameraClearFlags.SolidColor;
            Camera.main.backgroundColor = new Color(0.15f, 0.15f, 0.18f);
        }
    }
}
