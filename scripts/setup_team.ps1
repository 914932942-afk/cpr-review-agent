param(
    [string]$MLPython = '',
    [string]$AssetsZip = '',
    [string]$Python = '',
    [switch]$InstallGPU
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$taskWeb = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskWeb)) {
    if ($Python) { & $Python -m venv .venv }
    else { & py -3.11 -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
}
& $taskWeb -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Web dependency installation failed.' }
if ($AssetsZip) { & $taskWeb scripts/team_assets.py --archive $AssetsZip }
else { & $taskWeb scripts/team_assets.py }
if ($LASTEXITCODE -ne 0) { throw 'Resource installation failed. See README for manual ZIP download.' }
if ($InstallGPU) {
    $MLPython = Join-Path $taskRoot '.venv-ml\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $MLPython)) {
        if ($Python) { & $Python -m venv .venv-ml }
        else { & py -3.11 -m venv .venv-ml }
        if ($LASTEXITCODE -ne 0) { throw 'ML environment creation failed.' }
    }
    & $MLPython -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $MLPython -m pip install -r requirements-ml.txt
    if ($LASTEXITCODE -ne 0) { throw 'ML dependency installation failed.' }
    # The frozen Ultralytics release declares numpy<2; this project uses the
    # verified numpy 2 compatibility path. Avoid changing the validated versions.
    & $MLPython -m pip install --no-deps 'ultralytics==8.3.0' 'mmaction2==1.2.0' 'mmcv-lite==2.1.0'
    if ($LASTEXITCODE -ne 0) { throw 'Frozen vision package installation failed.' }
}
if (-not $MLPython) { throw 'Use -InstallGPU or pass -MLPython with an existing compatible environment.' }
$taskML = (Resolve-Path -LiteralPath $MLPython).Path
& $taskML -c "import torch,cv2,numpy,ultralytics,transformers; assert torch.cuda.is_available(), 'CUDA GPU is required for new video analysis'; print('CUDA:',torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'ML runtime is not ready.' }
$taskEnv = Join-Path $taskRoot '.env'
if (-not (Test-Path -LiteralPath $taskEnv)) {
    $taskLines = @('CPR_LLM_MODE=disabled', ('CPR_ML_PYTHON=' + $taskML.Replace('\','/')), 'DEEPSEEK_API_KEY=', 'DEEPSEEK_BASE_URL=https://api.deepseek.com', 'DEEPSEEK_MODEL=deepseek-flash')
    [IO.File]::WriteAllLines($taskEnv, $taskLines, (New-Object Text.UTF8Encoding($false)))
} else {
    Write-Host '.env already exists and was preserved. Confirm CPR_ML_PYTHON points to your ML Python.'
}
& $taskWeb scripts/run_main.py --check --mode disabled
if ($LASTEXITCODE -ne 0) { throw 'Mainline integrity check failed.' }
Write-Host 'Ready: .\scripts\start.ps1'
Write-Host 'Workspace: http://127.0.0.1:8013/'
Write-Host 'Demo guide: http://127.0.0.1:8013/demo'
