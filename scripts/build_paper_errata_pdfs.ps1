$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $workspace "docs\paper_errata"
$outputDir = Join-Path $workspace "output\pdf\paper_errata"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Get-ChildItem -LiteralPath $sourceDir -Filter "E*.md" | Sort-Object Name | ForEach-Object {
    $target = Join-Path $outputDir ($_.BaseName + ".pdf")
    pandoc $_.FullName `
        --from markdown+tex_math_dollars `
        --pdf-engine=xelatex `
        --variable "CJKmainfont=Microsoft YaHei" `
        --variable "mainfont=Times New Roman" `
        --variable "monofont=Consolas" `
        --variable "geometry:margin=2.2cm" `
        --variable "fontsize=11pt" `
        --number-sections `
        --output $target
    if ($LASTEXITCODE -ne 0) {
        throw "pandoc failed for $($_.Name) with exit code $LASTEXITCODE"
    }
}
