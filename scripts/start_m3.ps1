param([int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskWorkspace = Join-Path $taskProjectRoot 'workspaces\m3-calibrated-v1'
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $taskWorkspace 'workspace.json'))) { throw 'Prepare and validate the M3 workspace first.' }
Set-Location -LiteralPath $taskProjectRoot
$env:PYTHONPATH = Join-Path $taskProjectRoot 'backend'
$taskMlPython = (& $taskPython -c 'from cpr_agent.config import Settings; print(Settings.from_env().ml_python)').Trim()
if (-not (Test-Path -LiteralPath $taskMlPython)) { throw 'ML Python unavailable' }
$env:PYTHONPATH = Join-Path $taskWorkspace 'backend'
$env:CPR_ML_PYTHON = $taskMlPython
$env:CPR_VISION_MODE = 'real'
$env:CPR_VISION_PIPELINE = 'branches'
$env:CPR_LLM_MODE = 'disabled'
$env:CPR_CONTEXT_MODE = 'bound-v1'
$env:CPR_UPLOAD_PREFLIGHT = 'true'
$env:CPR_REFERENCE_CATALOG = Join-Path $taskWorkspace 'data\reference-v1\catalog.json'
$env:CPR_DATA_DIR = Join-Path $taskWorkspace 'data\m3-calibration-v1\runtime'
$env:CPR_BRANCH_MANIFEST = Join-Path $taskWorkspace 'data\m3-calibration-v1\pipeline-manifest.json'
$env:CPR_VISION_MANIFEST = Join-Path $taskWorkspace 'data\public_cpr\web-integration-v1\bundle-manifest.json'
$env:CPR_SERVICE_BUDGET_SECONDS = '900'
Set-Location -LiteralPath $taskWorkspace
& $taskPython scripts/check_module_slots.py --profile $env:CPR_BRANCH_MANIFEST
if ($LASTEXITCODE -ne 0) { throw 'M3 profile validation failed' }
& $taskPython scripts/context_start_check.py --data-dir $env:CPR_DATA_DIR --graph-version 'cpr-review-12-v0.6-grounded'
if ($LASTEXITCODE -ne 0) { throw 'Unfinished incompatible requests exist' }
Write-Host "CPR M3 calibrated review: http://127.0.0.1:$Port | source-quality-selective-v1 | LLM disabled"
& $taskPython -m uvicorn cpr_agent.api:app --app-dir (Join-Path $taskWorkspace 'backend') --host 127.0.0.1 --port $Port --workers 1
