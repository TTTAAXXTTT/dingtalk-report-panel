@echo off
setlocal
title DingTalk Report Panel - Sign In Again
pushd "%~dp0"

if /I "%DINGTALK_REPORT_LAUNCHER_DRY_RUN%"=="1" (
    if not exist "%~dp0login_dingtalk.ps1" exit /b 21
    echo LAUNCHER_OK
    popd
    exit /b 0
)

echo ============================================================
echo                 DingTalk - Sign In Again
echo ============================================================
echo.
echo Complete sign-in and authorization in the browser.
echo If asked, select a normal organization joined by this account.
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 goto powershell_failed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0login_dingtalk.ps1"
if errorlevel 1 goto login_failed

echo.
echo DingTalk sign-in completed. You can now run 02_Start.
goto normal_end

:powershell_failed
echo.
echo [FAILED] Windows PowerShell is unavailable. Contact your computer administrator.
goto failed_end

:login_failed
echo.
echo [FAILED] DingTalk sign-in did not finish. Review the error above and retry.
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
