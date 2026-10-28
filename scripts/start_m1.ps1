param([int]$Port = 8002)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = 'disabled'
$env:CPR_DATA_DIR = Join-Path $taskProjectRoot 'data\m1-stability-v1\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\m1-stability-v1\pipeline-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
Write-Host "CPR M1 workspace: http://127.0.0.1:$Port | separate database | DeepSeek disabled"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'M1 workspace exited with an error.' }
