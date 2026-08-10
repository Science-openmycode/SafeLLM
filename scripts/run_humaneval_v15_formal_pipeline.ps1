param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$baselineGeneration = "artifacts\eval\humaneval-hf-plaintext-bf16-generations-v15.json"
$candidateGeneration = "artifacts\eval\humaneval-hf-paper-v15-complete-bf16-generations-v15.json"
$baselineEvaluation = "artifacts\accuracy\humaneval-hf-plaintext-bf16-evaluated-v15.json"
$candidateEvaluation = "artifacts\accuracy\humaneval-hf-paper-v15-complete-bf16-evaluated-v15.json"
$comparison = "artifacts\accuracy\humaneval-paper-v15-complete-bf16-comparison.json"

if (!(Test-Path $baselineGeneration)) {
  & $PythonExe scripts\run_lm_eval.py `
    --model data\models\qwen2.5-0.5b `
    --tokenizer data\models\qwen2.5-0.5b `
    --tasks humaneval_generate_local `
    --include-path configs\eval\humaneval_local `
    --out $baselineGeneration `
    --predict-only --max-gen-toks 512 --dtype bfloat16 --batch-size 4
}

if (!(Test-Path $candidateGeneration)) {
  & $PythonExe scripts\run_lm_eval.py `
    --model data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
    --tokenizer data\models\qwen2.5-0.5b `
    --key data\keys\dev-qwen05b-paper-v15-complete-bf16\paper_key.safetensors `
    --tasks humaneval_generate_local `
    --include-path configs\eval\humaneval_local `
    --out $candidateGeneration `
    --predict-only --max-gen-toks 512 --dtype bfloat16 --batch-size 4
}

& $PythonExe scripts\evaluate_humaneval_wsl.py `
  --input $baselineGeneration --out $baselineEvaluation `
  --wsl-distro Ubuntu-24.04 --timeout 10

& $PythonExe scripts\evaluate_humaneval_wsl.py `
  --input $candidateGeneration --out $candidateEvaluation `
  --wsl-distro Ubuntu-24.04 --timeout 10

& $PythonExe scripts\compare_humaneval.py `
  --baseline $baselineEvaluation --candidate $candidateEvaluation --out $comparison

& $PythonExe scripts\build_qwen05b_final_acceptance.py `
  --config configs\transform\paper_qwen05b_complete.yaml `
  --out artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json

Write-Output "HUMANEVAL_V15_FORMAL_PIPELINE_COMPLETE"
