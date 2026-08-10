param(
    [switch]$SkipModel,
    [int]$MaxNewTokens = 16
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$sourceModel = Join-Path $workspace "data\models\qwen2.5-0.5b"
$privateModel = Join-Path $workspace "data\checkpoints\qwen2.5-0.5b-paper-v18-audited-bf16"
$keyDirectory = Join-Path $workspace "data\keys\dev-qwen05b-paper-v18-audited-bf16"
$auditDirectory = Join-Path $workspace "artifacts\audit"
$verificationDirectory = Join-Path $workspace "artifacts\verification"

Set-Location -LiteralPath $workspace
New-Item -ItemType Directory -Force -Path $auditDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $verificationDirectory | Out-Null

Write-Host "[1/7] Build source-line classification ledger"
uv run python scripts/build_paper_line_audit.py

Write-Host "[2/7] Static checks"
uv run ruff check .
$auditPythonFiles = @(
    "src/aloepri/transforms/vocab.py",
    "src/aloepri/transforms/paper_noise.py",
    "src/aloepri/transforms/paper_key_matrix.py",
    "src/aloepri/transforms/qwen_structural.py",
    "src/aloepri/conversion/paper_qwen2.py",
    "src/aloepri/models/configuration_aloepri_qwen2.py",
    "src/aloepri/models/modeling_aloepri_qwen2.py",
    "src/aloepri/client/sdk.py",
    "src/aloepri/serving/protocol.py",
    "src/aloepri/serving/hf_runtime.py",
    "src/aloepri/serving/app.py",
    "scripts/build_paper_line_audit.py",
    "scripts/verify_paper_formula_checkpoint.py",
    "scripts/verify_paper_qwen2_checkpoint.py",
    "scripts/verify_tokenizer_roundtrip.py"
)
uv run mypy $auditPythonFiles

Write-Host "[3/7] Unit and integration test suite"
uv run pytest -q

Write-Host "[4/7] Verify the paper's tokenizer text round-trip assumption"
uv run python scripts/verify_tokenizer_roundtrip.py `
    --tokenizer $sourceModel `
    --key-dir $keyDirectory `
    --out (Join-Path $auditDirectory "tokenizer-roundtrip-qwen05b-v18-key.json")

if ($SkipModel) {
    Write-Host "Model checks skipped by -SkipModel. Source/static/test evidence is complete."
    exit 0
}

Write-Host "[5/7] Reconstruct every transformed tensor from paper formulas"
uv run python scripts/verify_paper_formula_checkpoint.py `
    --source $sourceModel `
    --private $privateModel `
    --key-dir $keyDirectory `
    --out (Join-Path $auditDirectory "paper-formula-checkpoint-v18.json")

Write-Host "[6/7] Load source/private HF models and compare prefill, decode and generation"
uv run python scripts/verify_paper_qwen2_checkpoint.py `
    --source $sourceModel `
    --private $privateModel `
    --key-dir $keyDirectory `
    --dtype bfloat16 `
    --max-new-tokens $MaxNewTokens `
    --out (Join-Path $verificationDirectory "paper-qwen05b-v18-audited-bf16.json")

Write-Host "[7/7] Start the real 0.5B API in-process and verify private token streaming"
$env:ALOEPRI_RUN_MODEL_TESTS = "1"
$env:ALOEPRI_SOURCE_MODEL = $sourceModel
$env:ALOEPRI_PRIVATE_MODEL = $privateModel
$env:ALOEPRI_KEY_DIR = $keyDirectory
uv run pytest -q tests/integration/test_real_private_api.py

Write-Host "All requested verification stages completed successfully."
