$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

python -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
