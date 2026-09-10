# Build a shareable YuE2 Console zip.
#
#   .\package-release.ps1                  code only, ~1 MB, weights download on
#                                          first run
#   .\package-release.ps1 -IncludeWeights  code plus the 7.3 GB of model weights,
#                                          for a machine with slow or no internet
#
# The environment is never bundled: a .venv hard-codes its own machine's paths.
# install.bat rebuilds it in a few minutes.

param(
  [string] $OutDir = "$PSScriptRoot\..\dist",
  [switch] $IncludeWeights
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root = (Resolve-Path "$PSScriptRoot\..").Path
$stamp = Get-Date -Format "yyyyMMdd"
$name = if ($IncludeWeights) { "YuE2-Console-$stamp-with-weights" } else { "YuE2-Console-$stamp" }
$stage = Join-Path $env:TEMP $name

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Everything `pip install .` and the console need, and nothing machine-specific.
$items = @(
  "src", "webui", "examples", "docs", "licenses", "assets",
  "pyproject.toml", "MANIFEST.in", "LICENSE", "MODEL_LICENSE",
  "THIRD_PARTY_NOTICES.md", "README.md",
  "install.bat", "start-console.bat", "QUICKSTART.txt"
)

Write-Host "Staging the code ..."
foreach ($item in $items) {
  $source = Join-Path $root $item
  if (-not (Test-Path $source)) { Write-Warning "skipping missing $item"; continue }
  Copy-Item $source -Destination (Join-Path $stage $item) -Recurse -Force
}

# Strip anything local: caches, this machine's settings, uploaded audio.
Get-ChildItem $stage -Recurse -Directory -Include "__pycache__", ".pytest_cache", "_uploads", "*.egg-info" -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $stage -Recurse -File -Include "*.pyc", "settings.json" -ErrorAction SilentlyContinue |
  Remove-Item -Force -ErrorAction SilentlyContinue

if ($IncludeWeights) {
  # snapshot_download with local_dir writes real files. Copying the Hugging Face
  # cache directly would archive its symlinks as empty stubs instead.
  $python = Join-Path $root ".venv\Scripts\python.exe"
  if (-not (Test-Path $python)) { throw "No .venv found; the weights are fetched through it." }
  $models = Join-Path $stage "models"
  New-Item -ItemType Directory -Path $models -Force | Out-Null

  foreach ($pair in @(@("m-a-p/YuE2-3B", "YuE2-3B"), @("m-a-p/YuE2-Vae", "YuE2-Vae"))) {
    Write-Host "Materialising $($pair[0]) (this reads from the local cache if it is there) ..."
    & $python -c @"
from huggingface_hub import snapshot_download
snapshot_download('$($pair[0])', local_dir=r'$(Join-Path $models $pair[1])')
"@
    if ($LASTEXITCODE -ne 0) { throw "Could not materialise $($pair[0])" }
  }

  @"
The model weights ship inside this package, in models\.

The console uses them automatically, so install.bat has nothing to download and
works on a machine with no internet.

The weights are licensed separately from the code, under CC BY-NC 4.0, which
means non-commercial use only. See MODEL_LICENSE.
"@ | Set-Content (Join-Path $stage "models\README.txt") -Encoding utf8
}

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

# Store, do not deflate: safetensors are dense, so compression buys almost
# nothing and costs many minutes. CreateFromDirectory writes Zip64, which the
# 6.8 GB weight file needs and Compress-Archive does not reliably produce.
$level = if ($IncludeWeights) {
  [System.IO.Compression.CompressionLevel]::NoCompression
} else {
  [System.IO.Compression.CompressionLevel]::Optimal
}
Write-Host "Writing the archive ..."
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip, $level, $false)
Remove-Item $stage -Recurse -Force

$size = (Get-Item $zip).Length / 1GB
Write-Host ""
Write-Host ("Built {0}  ({1:N2} GB)" -f $zip, $size)
Write-Host "The other machine unzips it and double-clicks install.bat."
