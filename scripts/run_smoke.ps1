$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src summarize --results results/smoke
exit $LASTEXITCODE
