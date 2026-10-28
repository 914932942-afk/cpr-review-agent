param([int]$Port = 8006, [switch]$EnableDeepSeek)
$ErrorActionPreference = 'Stop'
if (-not $EnableDeepSeek) { throw 'This is the live integration workspace. Include -EnableDeepSeek to explicitly allow API requests.' }
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$taskLiveData = Join-Path $taskProjectRoot 'data\deepseek-live-v1\runtime'
if (-not (Test-Path -LiteralPath (Join-Path $taskLiveData 'business.sqlite3'))) { throw 'The isolated live workspace has not been prepared.' }
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = 'deepseek'
$env:CPR_CONTEXT_MODE = 'bound-v1'
$env:CPR_DATA_DIR = $taskLiveData
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\module-slots-v1\pipeline-manifest.json'
$env:CPR_VISION_MANIFEST = Join-Path $taskProjectRoot 'data\public_cpr\web-integration-v1\bundle-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
& '.\.venv\Scripts\python.exe' scripts/context_start_check.py --data-dir $env:CPR_DATA_DIR --graph-version 'cpr-review-12-v0.6-grounded'
if ($LASTEXITCODE -ne 0) { throw 'An incompatible unfinished request exists.' }
Write-Host "CPR live integration: http://127.0.0.1:$Port | DeepSeek enabled | Isolated data | Key loaded privately from .env"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
