[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$version = 'v1.0.59'
$expectedSha256 = '5393a0d5e00c70b58833c60610ad3a772926ca5e4eb38c360928e3d2552451bc'
$expectedExeSha256 = '6e4972b59c8858fabba5e24158833b4069114392cb7ac1bccc96fdd880411dd1'
$downloadUrl = 'https://github.com/DingTalk-Real-AI/dingtalk-workspace-cli/releases/download/v1.0.59/dws-windows-amd64.zip'
$toolRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".tools\dws\$version"))
$zipPath = Join-Path $toolRoot 'dws.zip'
$exePath = Join-Path $toolRoot 'dws.exe'

New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null

if (Test-Path -LiteralPath $exePath -PathType Leaf) {
    $actualExeSha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualExeSha256 -ne $expectedExeSha256) {
        throw "已有 dws.exe 校验失败；期望 $expectedExeSha256，实际 $actualExeSha256"
    }
    $versionOutput = (& $exePath version 2>&1 | Out-String)
    if ($versionOutput -match [regex]::Escape($version)) {
        Write-Output "DWS $version 已安装并通过校验：$exePath"
        exit 0
    }
    throw "已有 dws.exe 版本不符合固定版本 $version：$exePath"
}

Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
$actualSha256 = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "DWS 下载校验失败；期望 $expectedSha256，实际 $actualSha256"
}

Expand-Archive -LiteralPath $zipPath -DestinationPath $toolRoot -Force
if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    throw "解压后未找到 dws.exe：$exePath"
}

$actualExeSha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualExeSha256 -ne $expectedExeSha256) {
    throw "安装后的 dws.exe 校验失败；期望 $expectedExeSha256，实际 $actualExeSha256"
}

$installedVersion = (& $exePath version 2>&1 | Out-String)
if ($installedVersion -notmatch [regex]::Escape($version)) {
    throw "安装后的 DWS 版本验证失败：$installedVersion"
}
Write-Output "DWS $version 安装并通过 ZIP/EXE SHA-256 校验：$exePath"
