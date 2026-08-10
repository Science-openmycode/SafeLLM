param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$privateGeneration = "artifacts\eval\ifeval-hf-paper-v15-complete-bf16-generations-v5.json"
$baselineGeneration = "artifacts\eval\ifeval-hf-plaintext-bf16-generations-v5.json"
$privateScore = "artifacts\accuracy\ifeval-paper-v15-complete-bf16-scored-v5.json"
$baselineScore = "artifacts\accuracy\ifeval-plaintext-bf16-scored-v5.json"
$comparison = "artifacts\accuracy\ifeval-paper-v15-complete-bf16-comparison.json"

& $PythonExe scripts\run_hf_ifeval.py `
  --model data\checkpoints\qwen2.5-0.5b-paper-v15-complete-bf16 `
  --tokenizer data\models\qwen2.5-0.5b `
  --key data\keys\dev-qwen05b-paper-v15-complete-bf16\paper_key.safetensors `
  --out $privateGeneration `
  --max-new-tokens 1280 --batch-size 4 --minimum-free-gpu-gib 2.0 `
  --attn-implementation eager --dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw "private IFEval generation failed" }

& $PythonExe scripts\run_hf_ifeval.py `
  --model data\models\qwen2.5-0.5b `
  --tokenizer data\models\qwen2.5-0.5b `
  --out $baselineGeneration `
  --max-new-tokens 1280 --batch-size 4 --minimum-free-gpu-gib 2.0 `
  --attn-implementation eager --dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw "baseline IFEval generation failed" }

& $PythonExe scripts\score_ifeval.py --input $baselineGeneration --out $baselineScore
if ($LASTEXITCODE -ne 0) { throw "baseline IFEval scoring failed" }
& $PythonExe scripts\score_ifeval.py --input $privateGeneration --out $privateScore
if ($LASTEXITCODE -ne 0) { throw "private IFEval scoring failed" }
& $PythonExe scripts\compare_ifeval.py `
  --baseline $baselineScore --candidate $privateScore --out $comparison
if ($LASTEXITCODE -ne 0) { throw "IFEval comparison failed" }

& $PythonExe scripts\build_qwen05b_final_acceptance.py `
  --config configs\transform\paper_qwen05b_complete.yaml `
  --out artifacts\acceptance\paper-qwen05b-v15-complete-bf16.json
if ($LASTEXITCODE -ne 0) { throw "acceptance rebuild failed" }

Write-Output "IFEVAL_V15_FORMAL_PIPELINE_COMPLETE"
