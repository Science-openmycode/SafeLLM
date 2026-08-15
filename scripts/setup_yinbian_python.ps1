param(
    [switch]$Refresh
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspace ".venv\Scripts\python.exe"
$stamp = Join-Path $workspace ".venv\.yinbian-lock-sha256"
$lockFile = Join-Path $workspace "uv.lock"
$lockHash = (Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash.ToLowerInvariant()
$installedHash = if (Test-Path -LiteralPath $stamp) {
    (Get-Content -LiteralPath $stamp -Raw).Trim()
} else {
    ""
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    throw "uv was not found. Install uv once, then run this script again."
}

if (-not $Refresh -and (Test-Path -LiteralPath $venvPython)) {
    if ($installedHash -in @($lockHash, "compatible:$lockHash")) {
        Write-Output "Python environment is already ready; dependency installation was skipped."
        exit 0
    }
    Push-Location $workspace
    try {
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $uv.Source sync --frozen --check *> $null
        $environmentMatches = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedErrorActionPreference
        if ($environmentMatches) {
            Set-Content -LiteralPath $stamp -Value $lockHash -Encoding ascii
            Write-Output "Existing Python environment matches uv.lock; installation was skipped."
            exit 0
        }
        & $venvPython -c "import aloepri, asyncssh, fastapi, torch, webview" *> $null
        if ($LASTEXITCODE -eq 0) {
            Set-Content -LiteralPath $stamp -Value "compatible:$lockHash" -Encoding ascii
            Write-Warning "The existing environment is usable but differs from uv.lock. It was left unchanged so running services are not interrupted. Close Yinbian processes and rerun with -Refresh when convenient."
            exit 0
        }
    }
    finally {
        Pop-Location
    }
}

$env:UV_LINK_MODE = "hardlink"
$env:UV_HTTP_TIMEOUT = "300"
Push-Location $workspace
try {
    & $uv.Source sync --frozen
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
    Set-Content -LiteralPath $stamp -Value $lockHash -Encoding ascii
    & $venvPython -c "import aloepri, torch; print('Yinbian Python environment ready; torch=' + torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python environment self-check failed"
    }
}
finally {
    Pop-Location
}
