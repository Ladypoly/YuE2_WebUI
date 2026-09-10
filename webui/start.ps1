# Start the YuE2 Console on http://127.0.0.1:7865
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Error "No virtual environment at $python. Create it with: py -3.12 -m venv .venv"
}

$env:PYTHONPATH = Join-Path $root "src"
& $python (Join-Path $PSScriptRoot "server.py")
