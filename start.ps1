Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '.runtime\python\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '..\SugarSense\.tools\python\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython = 'python' }
& $taskPython run.py app.py
