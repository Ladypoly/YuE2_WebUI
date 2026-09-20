@echo off
rem Launch the YuE2 Console and open it in the default browser.
rem
rem   start-console.bat         follows the "Answer on the network" setting
rem   start-console.bat lan     forces network access on, whatever the setting says
rem
rem "lan" binds every interface and turns on the PIN, which the console prints
rem in this window. Windows Firewall asks once: allow Private networks only.
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if "%YUE2_PORT%"=="" set "YUE2_PORT=7865"
if /i "%~1"=="lan" set "YUE2_HOST=0.0.0.0"
if /i "%~1"=="-lan" set "YUE2_HOST=0.0.0.0"

if not exist "%PYTHON%" (
  echo No virtual environment found at %PYTHON%
  echo Create it first:  py -3.12 -m venv .venv
  pause
  exit /b 1
)

set "PYTHONPATH=%ROOT%src"
title YuE2 Console

if defined YUE2_HOST (
  echo Starting the YuE2 Console on every interface, port %YUE2_PORT%.
  echo The address and PIN for your phone appear just below.
) else (
  echo Starting the YuE2 Console on http://127.0.0.1:%YUE2_PORT%
  echo For phone access: tick "Answer on the network" under Engine, then
  echo start this again -- or run:  start-console.bat lan
)
echo Close this window, or press Ctrl+C, to stop it.
echo.

rem Open the browser only once the port answers, so the tab never lands on a
rem connection error. This waits in its own process while the server starts.
start "YuE2 Console browser" /min "%PYTHON%" "%ROOT%webui\open_when_ready.py" %YUE2_PORT%

"%PYTHON%" "%ROOT%webui\server.py"

endlocal
