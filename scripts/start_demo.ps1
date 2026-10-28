param([int]$Port = 8005)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = 'disabled'
$env:CPR_CONTEXT_MODE = 'bound-v1'
$env:CPR_DATA_DIR = Join-Path $taskProjectRoot 'data\demo-acceptance-v1\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\module-slots-v1\pipeline-manifest.json'
$env:CPR_VISION_MANIFEST = Join-Path $taskProjectRoot 'data\public_cpr\web-integration-v1\bundle-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
& '.\.venv\Scripts\python.exe' scripts/demo_preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Demo dependencies or pinned files did not pass preflight.' }
& '.\.venv\Scripts\python.exe' scripts/context_start_check.py --data-dir $env:CPR_DATA_DIR --graph-version 'cpr-review-12-v0.6-grounded'
if ($LASTEXITCODE -ne 0) { throw 'An incompatible unfinished demo request exists.' }
Write-Host "CPR demo: http://127.0.0.1:$Port | Separate demo database | Real vision | DeepSeek disabled"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Demo service exited with an error.' }
