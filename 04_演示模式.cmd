@echo off
setlocal
title DingTalk Report Panel - Demo Mode
pushd "%~dp0"

if /I "%DINGTALK_REPORT_LAUNCHER_DRY_RUN%"=="1" (
    if not exist "%~dp0start_demo.ps1" exit /b 21
    echo LAUNCHER_OK
    popd
    exit /b 0
)

echo ============================================================
echo              DingTalk Report Panel - Demo Mode
echo ============================================================
echo.
echo Demo mode uses sample chats and does not read DingTalk messages.
echo Starting the panel. The browser will open automatically.
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 goto powershell_failed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1"
if errorlevel 1 goto start_failed

echo.
echo Demo mode has stopped.
goto normal_end

:powershell_failed
echo.
echo [FAILED] Windows PowerShell is unavailable. Contact your computer administrator.
goto failed_end

:start_failed
echo.
echo [FAILED] Demo mode could not start. Keep this window and capture the error above.
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
