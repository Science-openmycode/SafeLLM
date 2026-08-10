param(
    [string]$InputFile = "docs/AloePri_0.5B_Research_Progress_Report.html",
    [string]$OutputFile = "output/pdf/AloePri_0.5B_Research_Progress_Report.pdf"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$inputPath = Join-Path $workspace $InputFile
$outputPath = Join-Path $workspace $OutputFile
$outputDirectory = Split-Path -Parent $outputPath

if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Markdown input does not exist: $inputPath"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

if ([IO.Path]::GetExtension($inputPath) -ieq ".html") {
    $browserCandidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    )
    $browser = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $browser) {
        throw "Chrome or Edge was not found for HTML-to-PDF rendering"
    }
    $inputUri = ([Uri]$inputPath).AbsoluteUri
    & $browser `
        --headless=new `
        --disable-gpu `
        --no-pdf-header-footer `
        "--print-to-pdf=$outputPath" `
        $inputUri
    if ($LASTEXITCODE -ne 0) {
        throw "HTML-to-PDF rendering failed with exit code $LASTEXITCODE"
    }
}
else {
    & pandoc $inputPath `
        --from "gfm+tex_math_dollars" `
        --pdf-engine xelatex `
        --toc `
        --number-sections `
        -V "CJKmainfont=Microsoft YaHei" `
        -V "mainfont=Times New Roman" `
        -V "sansfont=Microsoft YaHei" `
        -V "monofont=Microsoft YaHei" `
        -V "geometry:landscape,margin=1.5cm" `
        -o $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Pandoc/XeLaTeX compilation failed with exit code $LASTEXITCODE"
    }
}

Write-Output $outputPath
