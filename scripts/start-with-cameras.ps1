# Start the Windows host camera agent, then the CoastWatch Docker stack.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start-with-cameras.ps1
#
# The agent is stopped when this script exits (Ctrl+C). If an agent is already
# healthy on port 8765, it is reused and left running.
# See docs/USB_CAMERAS.md.

param(
    [int]$Port = 8765,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "backend\.venv\Scripts\python.exe"
$agent = Join-Path $repoRoot "tools\host_camera_agent.py"
$healthUrl = "http://127.0.0.1:$Port/health"

function Test-Agent {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 1
        return [bool]$response.ok
    } catch {
        return $false
    }
}

$agentProcess = $null
if (Test-Agent) {
    Write-Host "Host camera agent already running on port $Port."
} else {
    if (-not (Test-Path $python)) {
        Write-Error "Python venv not found at $python. Create backend\.venv first (see README)."
    }
    Write-Host "Starting host camera agent on port $Port..."
    $agentProcess = Start-Process -FilePath $python `
        -ArgumentList @($agent, "--host", "0.0.0.0", "--port", "$Port") `
        -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Agent) { break }
    }
    if (-not (Test-Agent)) {
        if ($agentProcess -and -not $agentProcess.HasExited) {
            Stop-Process -Id $agentProcess.Id -Force -ErrorAction SilentlyContinue
        }
        Write-Error "Host camera agent did not become healthy at $healthUrl."
    }
}

$composeArgs = @("compose", "up")
if (-not $NoBuild) { $composeArgs += "--build" }

try {
    Push-Location $repoRoot
    & docker @composeArgs
} finally {
    Pop-Location
    if ($agentProcess -and -not $agentProcess.HasExited) {
        Write-Host "Stopping host camera agent..."
        Stop-Process -Id $agentProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
