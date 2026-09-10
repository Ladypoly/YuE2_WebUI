@echo off
rem ===================================================================
rem  YuE2 Console - one-click installer for Windows + NVIDIA GPU
rem  Creates a local Python environment, installs the CUDA build of
rem  PyTorch, then starts the console in your browser.
rem ===================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title YuE2 Console - installer

echo.
echo   YuE2 Console installer
echo   ======================
echo.

rem ---------------------------------------------------------------- GPU check
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo   [!] No NVIDIA driver found. YuE2 needs an NVIDIA GPU with BF16 support
  echo       and about 24 GB of VRAM. Install the current driver, then run this
  echo       installer again.
  echo.
  pause
  exit /b 1
)
for /f "delims=" %%G in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader 2^>nul') do set "GPU=%%G"
echo   GPU: !GPU!

rem ------------------------------------------------------------- python check
set "PYEXE="
py -3.12 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.12"
if not defined PYEXE py -3.11 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.11"
if not defined PYEXE py -3.10 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.10"

if not defined PYEXE (
  echo.
  echo   Python 3.12 was not found. It can be installed automatically.
  choice /c YN /t 30 /d Y /m "   Install Python 3.12 now with winget (Y in 30s)"
  if errorlevel 2 (
    echo   Install Python 3.12 from https://www.python.org/downloads/ then rerun.
    pause
    exit /b 1
  )
  winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
  echo.
  echo   Python installed. Close this window and run install.bat again so the
  echo   new PATH is picked up.
  pause
  exit /b 0
)
for /f "delims=" %%G in ('%PYEXE% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%G"
echo   Python: !PYVER!
echo.

rem ------------------------------------------------------------- environment
set "VENV=%CD%\.venv"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VPY%" (
  echo   [1/5] Reusing the existing environment in .venv
) else (
  echo   [1/5] Creating the Python environment ...
  %PYEXE% -m venv "%VENV%"
  if errorlevel 1 goto failed
)

echo   [2/5] Updating pip ...
"%VPY%" -m pip install --upgrade pip --quiet
if errorlevel 1 goto failed

rem PyPI ships a CPU-only torch on Windows, so take the CUDA 12.8 build first.
rem That wheel covers Ada and Blackwell alike (sm_89 through sm_120), so a 4090
rem and a 5090 install identically. Doing it before the package avoids
rem downloading the CPU build only to replace it.
echo   [3/5] Installing PyTorch with CUDA support (about 2.5 GB, please wait) ...
"%VPY%" -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 goto failed

echo   [4/5] Installing YuE2 and the console ...
"%VPY%" -m pip install . --quiet
if errorlevel 1 goto failed
"%VPY%" -m pip install -r webui\requirements.txt --quiet
if errorlevel 1 goto failed
rem Faster Hugging Face downloads for the 7 GB of model weights.
"%VPY%" -m pip install hf_xet --quiet

echo   [5/5] Checking the installation ...
"%VPY%" -c "import torch,yue2,sys; ok=torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print('        torch', torch.__version__); print('        device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA DEVICE'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
  echo.
  echo   [!] CUDA or BF16 is not available. The console will start, but
  echo       generation needs an NVIDIA GPU with BF16 support.
  echo.
)

echo.
echo   Installed.
echo.
echo   The model weights ^(about 7 GB^) download from Hugging Face the first
echo   time you generate a song. You can fetch them now instead of waiting.
echo.
choice /c YN /t 30 /d Y /m "   Download the model weights now (Y in 30s)"
if not errorlevel 2 (
  echo.
  echo   Downloading YuE2-3B and the decoder ...
  "%VPY%" -c "from huggingface_hub import snapshot_download; [snapshot_download(r, max_workers=4) for r in ('m-a-p/YuE2-3B','m-a-p/YuE2-Vae')]; print('        weights ready')"
)

echo.
echo   Starting the console. From now on, use start-console.bat.
echo.
call "%CD%\start-console.bat"
exit /b 0

:failed
echo.
echo   [!] Installation failed at the step above. Scroll up for the error.
echo       Re-running install.bat resumes from where it stopped.
echo.
pause
exit /b 1
