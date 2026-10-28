param([int]$Port = 8013, [switch]$EnableDeepSeek, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Project web Python is missing' }
Set-Location -LiteralPath $taskRoot
$taskMode = if ($EnableDeepSeek) { 'deepseek' } else { 'disabled' }
$taskArguments = @((Join-Path $PSScriptRoot 'run_main.py'), '--port', $Port, '--mode', $taskMode)
if ($CheckOnly) { $taskArguments += '--check' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'Current mainline did not start or pass preflight' }
