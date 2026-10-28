param([int]$Port = 8004, [switch]$LegacyContext, [switch]$EnableDeepSeek)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskProjectRoot
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = if ($EnableDeepSeek) { 'deepseek' } else { 'disabled' }
$env:CPR_CONTEXT_MODE = if ($LegacyContext) { 'legacy' } else { 'bound-v1' }
$env:CPR_DATA_DIR = Join-Path $taskProjectRoot 'data\module-slots-v1\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskProjectRoot 'data\module-slots-v1\pipeline-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
$taskGraphVersion = if ($LegacyContext) { 'cpr-review-12-v0.4-slots' } else { 'cpr-review-12-v0.6-grounded' }
& '.\.venv\Scripts\python.exe' scripts/context_start_check.py --data-dir $env:CPR_DATA_DIR --graph-version $taskGraphVersion
if ($LASTEXITCODE -ne 0) { throw 'Finish or cancel existing requests in their original context mode before switching.' }
Write-Host "CPR review workspace: http://127.0.0.1:$Port | Context $env:CPR_CONTEXT_MODE | M2-M5 baseline adapters | LLM $env:CPR_LLM_MODE"
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Module workspace exited with an error.' }
