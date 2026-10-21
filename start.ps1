param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [switch]$FixtureMode,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

try {
    python -c "import fastapi, uvicorn, httpx, pydantic; from zoneinfo import ZoneInfo; ZoneInfo('Asia/Shanghai')" 2>$null
    $runtimeReady = $LASTEXITCODE -eq 0
} catch {
    $runtimeReady = $false
}
if (-not $runtimeReady) {
    Write-Host "Installing missing runtime dependencies. Please keep this window open."
    & (Join-Path $projectRoot 'install.ps1') -SkipDws
}

if ($FixtureMode) {
    $env:DINGTALK_REPORT_MODE = "fixture"
} else {
    $env:DINGTALK_REPORT_MODE = "real"
}

if (-not $env:DWS_PATH) {
    $bundledDws = Join-Path $projectRoot ".tools\dws\v1.0.59\dws.exe"
    if (Test-Path -LiteralPath $bundledDws) {
        $env:DWS_PATH = $bundledDws
    }
}

$url = "http://${HostAddress}:${Port}"
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($targetUrl)
        Start-Sleep -Milliseconds 900
        Start-Process $targetUrl
    } -ArgumentList $url | Out-Null
}

Write-Host "钉钉聊天报告面板：$url"
Write-Host "运行模式：$env:DINGTALK_REPORT_MODE"
python -m uvicorn app.main:app --host $HostAddress --port $Port
