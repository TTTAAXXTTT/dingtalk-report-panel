@echo off
setlocal
title DingTalk Report Panel - Start
pushd "%~dp0"

if /I "%DINGTALK_REPORT_LAUNCHER_DRY_RUN%"=="1" (
    if not exist "%~dp0start.ps1" exit /b 21
    echo LAUNCHER_OK
    popd
    exit /b 0
)

echo ============================================================
echo               DingTalk Report Panel - Start
echo ============================================================
echo.
echo Starting the panel. The browser will open automatically.
echo Keep this window open while using the panel.
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 goto powershell_failed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 goto start_failed

echo.
echo The report panel has stopped.
goto normal_end

:powershell_failed
echo.
echo [FAILED] Windows PowerShell is unavailable. Contact your computer administrator.
goto failed_end

:start_failed
echo.
echo [FAILED] The panel could not start. Keep this window and capture the error above.
goto failed_end

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
