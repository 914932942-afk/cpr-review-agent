param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Virtual environment missing. Run scripts\bootstrap.ps1 first.'
}
if (-not (Test-Path -LiteralPath 'frontend\dist\index.html')) {
    throw 'Frontend build missing. Run npm.cmd --prefix frontend run build first.'
}
if (Test-Path -LiteralPath '.env') {
    foreach ($line in Get-Content -LiteralPath '.env') {
        if ($line -match '^\s*((?:CPR|DEEPSEEK)_[A-Z_]+)=(.*)$') {
            $configName = $Matches[1]
            $configValue = $Matches[2].Trim()
            if ($configValue.Length -ge 2 -and (($configValue.StartsWith('"') -and $configValue.EndsWith('"')) -or ($configValue.StartsWith("'") -and $configValue.EndsWith("'")))) {
                $configValue = $configValue.Substring(1, $configValue.Length - 2)
            }
            Set-Item -LiteralPath ('Env:' + $configName) -Value $configValue
        }
    }
}
Write-Host "CPR Review: http://127.0.0.1:$Port  |  API: http://127.0.0.1:$Port/docs"
Write-Host 'Local development only. LLM mode follows CPR_LLM_MODE (disabled by default). Press Ctrl+C to stop.'
& '.\.venv\Scripts\python.exe' -m uvicorn cpr_agent.api:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Backend exited with an error.' }
