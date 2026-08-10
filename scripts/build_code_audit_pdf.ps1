$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
$source = Join-Path $workspace "docs\PAPER_CODE_LINE_BY_LINE_AUDIT_0.5B.md"
$outputDirectory = Join-Path $workspace "output\pdf"
$target = Join-Path $outputDirectory "AloePri_0.5B_Paper_Code_Line_Audit.pdf"

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

pandoc $source `
    --from markdown+tex_math_dollars `
    --pdf-engine=xelatex `
    --variable "CJKmainfont=Microsoft YaHei" `
    --variable "mainfont=Times New Roman" `
    --variable "monofont=Consolas" `
    --variable "geometry:margin=2cm" `
    --variable "fontsize=10pt" `
    --variable "colorlinks=true" `
    --variable "linkcolor=NavyBlue" `
    --variable "urlcolor=NavyBlue" `
    --toc `
    --output $target
if ($LASTEXITCODE -ne 0) {
    throw "pandoc failed with exit code $LASTEXITCODE"
}

Write-Host $target
