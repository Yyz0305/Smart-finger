# Smart Finger — 一键启动项目
Write-Host "=== Smart Finger 一键启动 ===" -ForegroundColor Cyan

# 1. Kill old processes
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force 2>$null
Start-Sleep 1

# 2. Start Hand Broadcaster (UDP -> Unity)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = "hand_broadcaster.py"
$psi.WorkingDirectory = $PSScriptRoot
$psi.UseShellExecute = $true
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Minimized
$p = [System.Diagnostics.Process]::Start($psi)
Write-Host "[1/3] HandBroadcaster started (PID $($p.Id))" -ForegroundColor Green

Start-Sleep 2

# 3. Launch Unity Editor with project
$unityExe = "C:\Program Files\Unity\Hub\Editor\2022.3.62f1\Editor\Unity.exe"
$projPath = Join-Path $PSScriptRoot "unity_project"
if (Test-Path $unityExe) {
    Start-Process $unityExe -ArgumentList "-projectPath `"$projPath`""
    Write-Host "[2/3] Unity Editor launched" -ForegroundColor Green
} else {
    Write-Host "[2/3] Unity not found at $unityExe" -ForegroundColor Yellow
    Write-Host "       Open Unity Hub and add the project manually" -ForegroundColor Yellow
}

# 4. Open project folder
Start-Process $projPath
Write-Host "[3/3] Project folder opened" -ForegroundColor Green

Write-Host ""
Write-Host "=== Ready ===" -ForegroundColor Cyan
Write-Host "Unity Editor is opening your project." -ForegroundColor White
Write-Host "Once loaded, just press Play ▶️" -ForegroundColor Green
Write-Host "The GameManager will auto-create with all components." -ForegroundColor White
Write-Host ""
Write-Host "Hand broadcaster sending data to Unity via UDP :8888" -ForegroundColor DarkGray
