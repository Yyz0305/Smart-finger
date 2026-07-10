using UnityEngine;

/// <summary>
/// 场景初始化 — 灯光、地面、桌子、物体
/// </summary>
public class SceneSetup : MonoBehaviour
{
    void Start()
    {
        SetupLighting();
        CreateEnvironment();
        CreateObjects();
    }

    void SetupLighting()
    {
        // 方向光
        var sun = new GameObject("DirectionalLight").AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.transform.rotation = Quaternion.Euler(40, -30, 0);
        sun.shadowStrength = 0.8f;
        sun.shadows = LightShadows.Soft;

        // 环境光
        RenderSettings.ambientIntensity = 0.4f;
        RenderSettings.ambientLight = new Color(0.3f, 0.3f, 0.35f);
    }

    void CreateEnvironment()
    {
        // 地板
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.localScale = Vector3.one * 3f;
        floor.GetComponent<Renderer>().material.color = new Color(0.77f, 0.66f, 0.51f);

        // 墙壁
        var wall = GameObject.CreatePrimitive(PrimitiveType.Quad);
        wall.name = "Wall";
        wall.transform.SetPositionAndRotation(new Vector3(0, 2, -3), Quaternion.identity);
        wall.transform.localScale = new Vector3(8, 4, 1);
        wall.GetComponent<Renderer>().material.color = new Color(0.94f, 0.93f, 0.90f);

        // 桌子
        var tableTop = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tableTop.name = "TableTop";
        tableTop.transform.SetPositionAndRotation(new Vector3(0, 0.9f, 0), Quaternion.identity);
        tableTop.transform.localScale = new Vector3(2.2f, 0.06f, 1.6f);
        tableTop.GetComponent<Renderer>().material.color = new Color(0.55f, 0.35f, 0.20f);

        // 桌腿
        foreach (var pos in new Vector3[] {
            new(-1.0f, 0.45f, -0.7f), new(-1.0f, 0.45f, 0.7f),
            new(1.0f, 0.45f, -0.7f), new(1.0f, 0.45f, 0.7f) })
        {
            var leg = GameObject.CreatePrimitive(PrimitiveType.Cube);
            leg.transform.position = pos;
            leg.transform.localScale = new Vector3(0.05f, 0.85f, 0.05f);
            leg.GetComponent<Renderer>().material.color = new Color(0.35f, 0.22f, 0.12f);
        }
    }

    void CreateObjects()
    {
        // 水杯
        CreateCup(new Vector3(0, 1.15f, 0));
        // 球
        CreateBall(new Vector3(0.5f, 1.0f, 0.3f));
        // 方块
        CreateBlock(new Vector3(-0.5f, 1.0f, -0.2f));
    }

    void CreateCup(Vector3 pos)
    {
        var cup = new GameObject("Cup");
        cup.transform.position = pos;
        cup.AddComponent<GrabbableObject>();

        var body = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        body.transform.SetParent(cup.transform);
        body.transform.localScale = new Vector3(0.25f, 0.3f, 0.25f);
        body.GetComponent<Renderer>().material.color = new Color(0.31f, 0.76f, 0.97f);

        var water = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        water.transform.SetParent(cup.transform);
        water.transform.localPosition = new Vector3(0, 0.1f, 0);
        water.transform.localScale = new Vector3(0.22f, 0.05f, 0.22f);
        water.GetComponent<Renderer>().material.color = new Color(0.5f, 0.83f, 0.98f);

        var handle = GameObject.CreatePrimitive(PrimitiveType.Cube);
        handle.transform.SetParent(cup.transform);
        handle.transform.localPosition = new Vector3(0.18f, 0.1f, 0);
        handle.transform.localScale = new Vector3(0.04f, 0.15f, 0.04f);
        handle.GetComponent<Renderer>().material.color = new Color(0.31f, 0.76f, 0.97f);
    }

    void CreateBall(Vector3 pos)
    {
        var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ball.name = "Ball";
        ball.transform.position = pos;
        ball.transform.localScale = Vector3.one * 0.18f;
        ball.GetComponent<Renderer>().material.color = Color.red;
        ball.AddComponent<GrabbableObject>();
    }

    void CreateBlock(Vector3 pos)
    {
        var block = GameObject.CreatePrimitive(PrimitiveType.Cube);
        block.name = "Block";
        block.transform.position = pos;
        block.transform.localScale = new Vector3(0.15f, 0.15f, 0.15f);
        block.GetComponent<Renderer>().material.color = Color.yellow;
        block.AddComponent<GrabbableObject>();
    }
}
