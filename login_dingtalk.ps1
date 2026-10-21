[CmdletBinding()]
param(
    [switch]$Device
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dwsPath = if ($env:DWS_PATH) {
    $env:DWS_PATH
} else {
    Join-Path $projectRoot '.tools\dws\v1.0.59\dws.exe'
}

if (-not (Test-Path -LiteralPath $dwsPath -PathType Leaf)) {
    & (Join-Path $projectRoot 'setup_dws.ps1')
}

$arguments = @('auth', 'login')
if ($Device) {
    $arguments += '--device'
}

& $dwsPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw '钉钉 OAuth 登录未完成。'
}

& $dwsPath auth status --format json
if ($LASTEXITCODE -ne 0) {
    throw '无法验证钉钉 OAuth 状态。'
}
