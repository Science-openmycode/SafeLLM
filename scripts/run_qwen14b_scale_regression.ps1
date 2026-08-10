param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$GpuMemory = "5GiB",
    [string]$CpuMemory = "8GiB",
    [switch]$SkipGeneration
)

$ErrorActionPreference = "Stop"
$Source = "data/models/qwen2.5-14b"
$Private = "data/checkpoints/qwen2.5-14b-paper-functional-stream"
$KeyDir = "data/keys/dev-qwen14b-paper-functional-stream-001"
$Key = "$KeyDir/paper_key.safetensors"
$Structural = "artifacts/verification/streaming-14b-structural.json"
$Smoke = "artifacts/verification/qwen2.5-14b-offload-smoke.json"

if (-not (Test-Path "$Source/download_receipt.json")) {
    throw "14B download receipt is missing: $Source/download_receipt.json"
}

& $Python scripts/convert_paper_qwen2_streaming.py `
    --source $Source `
    --output $Private `
    --key-dir $KeyDir `
    --h 128 `
    --seed 20260805 `
    --alpha-e 0 `
    --alpha-h 0 `
    --algorithm2 `
    --block-beta 8 `
    --sampling-gamma 1000 `
    --qk-scale-min 0.5 `
    --qk-scale-max 2 `
    --ffn-scale-min 0.5 `
    --ffn-scale-max 2
if ($LASTEXITCODE -ne 0) { throw "14B streaming conversion failed" }

& $Python scripts/verify_streaming_qwen2_checkpoint.py `
    --source $Source `
    --private $Private `
    --key $Key `
    --out $Structural
if ($LASTEXITCODE -ne 0) { throw "14B structural verification failed" }

if (-not $SkipGeneration) {
    & $Python scripts/run_scale_offload_smoke.py `
        --source $Source `
        --private $Private `
        --key $Key `
        --offload-root artifacts/offload/qwen2.5-14b `
        --gpu-memory $GpuMemory `
        --cpu-memory $CpuMemory `
        --max-new-tokens 8 `
        --out $Smoke
    if ($LASTEXITCODE -ne 0) { throw "14B generation round-trip failed" }
}
