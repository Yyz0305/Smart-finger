# Smart Finger - 环境部署脚本
# 以管理员身份运行

Write-Host "=== Smart Finger 环境部署 ===" -ForegroundColor Cyan

# 1. Python 依赖
Write-Host "[1/4] 安装 Python 包..." -ForegroundColor Yellow
pip install -r requirements.txt 2>&1 | Out-Null
Write-Host "  Python 包就绪" -ForegroundColor Green

# 2. 下载 MediaPipe 模型
Write-Host "[2/4] 下载 MediaPipe 模型..." -ForegroundColor Yellow
$models = @{
    "hand_landmarker.task" = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    "gesture_recognizer.task" = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
}
foreach ($name in $models.Keys) {
    $path = Join-Path $PSScriptRoot $name
    if (-not (Test-Path $path)) {
        Write-Host "  下载 $name ..."
        Invoke-WebRequest -Uri $models[$name] -OutFile $path -UseBasicParsing
    }
}
Write-Host "  模型就绪" -ForegroundColor Green

# 3. 安装 Unity Hub (如未安装)
Write-Host "[3/4] 安装 Unity Hub..." -ForegroundColor Yellow
$hubPath = "C:\Program Files\Unity Hub\Unity Hub.exe"
if (-not (Test-Path $hubPath)) {
    winget install --id Unity.UnityHub -e --silent 2>&1 | Out-Null
    Write-Host "  Unity Hub 已安装" -ForegroundColor Green
} else {
    Write-Host "  Unity Hub 已存在" -ForegroundColor Green
}

# 4. Unity Editor 安装提示
Write-Host "[4/4] Unity Editor 就绪检查..." -ForegroundColor Yellow
$editorPath = "C:\Program Files\Unity\Hub\Editor\2022.3.62f1\Editor\Unity.exe"
if (-not (Test-Path $editorPath)) {
    Write-Host "  Unity Editor 未安装. 请手动操作:" -ForegroundColor Yellow
    Write-Host "  1. 启动 Unity Hub" -ForegroundColor White
    Write-Host "  2. 点击 Installs → Install Editor → 选择 2022.3 LTS" -ForegroundColor White
    Write-Host "  3. 安装完成后打开 unity_project/ 目录" -ForegroundColor White
    
    # 自动启动 Unity Hub 并尝试安装
    if (Test-Path $hubPath) {
        Start-Process $hubPath
        Start-Process "unityhub://2022.3.62f1/96770f904ca7"
    }
} else {
    Write-Host "  Unity Editor 就绪" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== 部署完成 ===" -ForegroundColor Cyan
Write-Host "启动命令:" -ForegroundColor Cyan
Write-Host "  python hand_broadcaster.py    # UDP 广播 (供 Unity 使用)" -ForegroundColor White
Write-Host "  python digital_twin.py        # Ursina 3D 快速预览" -ForegroundColor White
Write-Host "  python smart_finger.py        # 2D 手势识别 + 轨迹分析" -ForegroundColor White
