$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m src prepare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src run --config configs/course.yaml
$runCode = $LASTEXITCODE
if ($runCode -notin @(0, 2)) { exit $runCode }
& $python -m src summarize --results results/course
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit $runCode
