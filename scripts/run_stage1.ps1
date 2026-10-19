$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m src prepare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src run --config configs/stage1.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src summarize --results results/stage1
exit $LASTEXITCODE
