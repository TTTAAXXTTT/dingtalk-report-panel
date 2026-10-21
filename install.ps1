[CmdletBinding()]
param(
    [switch]$SkipDws
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

python -c "import sys; assert sys.version_info >= (3, 11), '需要 Python 3.11 或更高版本'; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw '未找到可用的 Python 3.11+。'
}

python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Python 依赖安装失败。'
}

if (-not $SkipDws) {
    & (Join-Path $projectRoot 'setup_dws.ps1')
}

Write-Output '安装完成。先运行 .\login_dingtalk.ps1 完成一次钉钉登录，再运行 .\start.ps1。'
