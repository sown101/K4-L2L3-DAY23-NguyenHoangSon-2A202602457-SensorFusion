# Run from PowerShell: . .\Activate-Lab.ps1
# Activates the local Python 3.12 environment and loads .env for this terminal.
$taskRepoRoot = $PSScriptRoot
$taskCondaExe = Join-Path (Split-Path $taskRepoRoot -Parent) '.tools\conda\Scripts\conda.exe'
$taskEnvironment = Join-Path $taskRepoRoot '.venv'

if (-not (Test-Path -LiteralPath $taskCondaExe -PathType Leaf)) {
    throw "Conda was not found: $taskCondaExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $taskEnvironment 'python.exe') -PathType Leaf)) {
    throw "The lab Python environment was not found: $taskEnvironment"
}

$taskCondaHook = & $taskCondaExe shell.powershell hook
if ($LASTEXITCODE -ne 0) {
    throw 'Could not initialize Conda for PowerShell.'
}
$taskCondaHook | Out-String | Invoke-Expression
conda activate $taskEnvironment
if ($LASTEXITCODE -ne 0) {
    throw 'Could not activate the lab Python environment.'
}

$taskEnvFile = Join-Path $taskRepoRoot '.env'
if (Test-Path -LiteralPath $taskEnvFile -PathType Leaf) {
    foreach ($taskLine in Get-Content -LiteralPath $taskEnvFile -Encoding utf8) {
        if ($taskLine -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim(), 'Process')
        }
    }
}

if ($env:PYTEST_DEBUG_TEMPROOT) {
    New-Item -ItemType Directory -Path $env:PYTEST_DEBUG_TEMPROOT -Force | Out-Null
}

Write-Host 'Day 23 environment activated. Run: python -m pytest student/tests -q'
