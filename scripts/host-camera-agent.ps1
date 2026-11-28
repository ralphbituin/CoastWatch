# Start the CoastWatch Windows host camera agent.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\host-camera-agent.ps1
#
# The agent enumerates DirectShow cameras (including OBS Virtual Camera) and
# serves them as MJPEG so the Dockerized backend can use them.
# See docs/USB_CAMERAS.md.

param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8765,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "backend\.venv\Scripts\python.exe"
$agent = Join-Path $repoRoot "tools\host_camera_agent.py"

if (-not (Test-Path $python)) {
    Write-Error "Python venv not found at $python. Create backend\.venv first (see README)."
}

if (-not (Test-Path $agent)) {
    Write-Error "Agent script not found at $agent."
}

$arguments = @($agent, "--host", $BindHost, "--port", "$Port")
if ($Verbose) {
    $arguments += "--verbose"
}

Write-Host "Starting CoastWatch host camera agent on http://${BindHost}:${Port}"
Write-Host "Press Ctrl+C to stop."
& $python @arguments
