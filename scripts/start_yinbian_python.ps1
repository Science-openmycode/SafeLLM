param(
    [ValidateSet("deploy", "chat", "cli")]
    [string]$App = "deploy",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArguments
)

$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Run 'uv sync --frozen' once in $workspace."
}

Push-Location $workspace
try {
    if ($App -eq "deploy") {
        & $python -m aloepri.desktop.deploy @AppArguments
    }
    elseif ($App -eq "chat") {
        & $python -m aloepri.desktop.chat @AppArguments
    }
    else {
        & $python -m aloepri.desktop.launcher @AppArguments
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
