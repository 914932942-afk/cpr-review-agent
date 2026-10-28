param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = 'disabled'
$env:CPR_DATA_DIR = Join-Path $taskProjectRoot 'data\branch-integration-v1\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\branch-integration-v1\pipeline-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
Write-Host "CPR branch workspace: http://127.0.0.1:$Port | separate local demo database | DeepSeek disabled"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Branch workspace exited with an error.' }
