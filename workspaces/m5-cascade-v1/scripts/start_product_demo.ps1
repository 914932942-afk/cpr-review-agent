param([int]$Port = 8007, [switch]$EnableDeepSeek)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = if ($EnableDeepSeek) { 'deepseek' } else { 'disabled' }
$env:CPR_CONTEXT_MODE = 'bound-v1'
$env:CPR_UPLOAD_PREFLIGHT = 'true'
$env:CPR_REFERENCE_CATALOG = Join-Path $taskProjectRoot 'data\reference-v1\catalog.json'
$env:CPR_DATA_DIR = Join-Path $taskProjectRoot 'data\product-demo-v2\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\module-slots-v1\pipeline-manifest.json'
$env:CPR_VISION_MANIFEST = Join-Path $taskProjectRoot 'data\public_cpr\web-integration-v1\bundle-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
& '.\.venv\Scripts\python.exe' scripts/demo_preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Product demo dependencies did not pass preflight.' }
& '.\.venv\Scripts\python.exe' scripts/context_start_check.py --data-dir $env:CPR_DATA_DIR --graph-version 'cpr-review-12-v0.6-grounded'
if ($LASTEXITCODE -ne 0) { throw 'An incompatible unfinished product demo request exists.' }
Write-Host "CPR product demo: http://127.0.0.1:$Port | LLM $env:CPR_LLM_MODE | Separate product demo data"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Product demo service exited with an error.' }
