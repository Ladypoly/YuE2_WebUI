# Start the YuE2 Console on http://127.0.0.1:7865
#
#   .\webui\start.ps1          just this machine
#   .\webui\start.ps1 -Lan     also reachable from your phone on the same Wi-Fi
#
# -Lan binds every interface and turns on the PIN. Windows Firewall asks once
# on the first LAN start; allow it for Private networks only. Do not forward
# this port on your router: it is plain HTTP with a six-digit PIN.
param(
  [switch]$Lan,
  [int]$Port = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Error "No virtual environment at $python. Create it with: py -3.12 -m venv .venv"
}

$env:PYTHONPATH = Join-Path $root "src"
# Without -Lan the saved setting decides, so a double-clicked launcher can put
# the console on the network without anyone remembering a switch.
if ($Lan) { $env:YUE2_HOST = "0.0.0.0" }
if ($Port -gt 0) { $env:YUE2_PORT = "$Port" }
& $python (Join-Path $PSScriptRoot "server.py")
