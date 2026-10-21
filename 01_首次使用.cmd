@echo off
setlocal
title DingTalk Report Panel - First Run
pushd "%~dp0"

if /I "%DINGTALK_REPORT_LAUNCHER_DRY_RUN%"=="1" (
    if not exist "%~dp0install.ps1" exit /b 21
    if not exist "%~dp0login_dingtalk.ps1" exit /b 22
    if not exist "%~dp0start.ps1" exit /b 23
    echo LAUNCHER_OK
    popd
    exit /b 0
)

echo ============================================================
echo             DingTalk Report Panel - First Run
echo ============================================================
echo.
echo This window will install, sign in, and start the panel.
echo Keep this window open.
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 goto powershell_failed

echo [1/3] Installing required components. Please wait...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 goto install_failed

echo.
echo [2/3] Complete DingTalk authorization in the browser...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0login_dingtalk.ps1"
if errorlevel 1 goto login_failed

echo.
echo [3/3] Starting the report panel. The browser will open...
echo Keep this window open while using the panel.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
goto stopped

:powershell_failed
echo.
echo [FAILED] Windows PowerShell is unavailable. Contact your computer administrator.
goto failed_end

:install_failed
echo.
echo [FAILED] Installation did not finish. Keep this window and capture the error above.
goto failed_end

:login_failed
echo.
echo [FAILED] DingTalk sign-in did not finish. Check the browser authorization page.
goto failed_end

:stopped
echo.
echo The report panel has stopped.
echo If you did not stop it, capture the error shown above.
goto normal_end

:failed_end
echo.
pause
popd
exit /b 1

:normal_end
echo.
pause
popd
exit /b 0
