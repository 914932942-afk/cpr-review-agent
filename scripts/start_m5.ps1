param([int]$Port = 8013, [switch]$EnableDeepSeek, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'start_main.ps1') -Port $Port -EnableDeepSeek:$EnableDeepSeek -CheckOnly:$CheckOnly
