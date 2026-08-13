param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$model = "data\packages\qwen05b-candidate-v47-stable-factor"
$source = "data\models\qwen2.5-0.5b"
$key = "data\keys\dev-qwen05b-candidate-v47-best-single\paper_key.safetensors"
$onlineKey = "data\keys\qwen05b-candidate-v47-best-single-online"
$serverPackage = "data\server-packages\qwen05b-candidate-v47-stable-factor"
$ifevalBaseline = "artifacts\eval\v47-final-ifeval-baseline-fp32.json"
$ifevalGeneration = "artifacts\eval\v47-final-ifeval-candidate.json"

function Get-IfevalSampleCount([string]$Path) {
  if (!(Test-Path -LiteralPath $Path)) { return 0 }
  $payload = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
  return @($payload.samples).Count
}

if ((Get-IfevalSampleCount $ifevalBaseline) -lt 541) {
  & $PythonExe scripts\run_hf_ifeval.py `
    --model $source --tokenizer $source `
    --out $ifevalBaseline --max-new-tokens 1280 --batch-size 2 `
    --minimum-free-gpu-gib 0.85 --attn-implementation sdpa `
    --dtype float32 --gpu-memory-fraction 0.65
  if ($LASTEXITCODE -ne 0) { throw "v47 baseline IFEval generation failed" }
}

if ((Get-IfevalSampleCount $ifevalGeneration) -lt 541) {
  & $PythonExe scripts\run_hf_ifeval.py `
    --model $model --tokenizer $source --key $key `
    --out $ifevalGeneration --max-new-tokens 1280 --batch-size 2 `
    --minimum-free-gpu-gib 0.85 --attn-implementation sdpa `
    --dtype float32 --gpu-memory-fraction 0.65
  if ($LASTEXITCODE -ne 0) { throw "v47 IFEval generation failed" }
}

& $PythonExe scripts\score_ifeval.py `
  --input $ifevalBaseline `
  --out artifacts\accuracy\v47-final-ifeval-baseline-scored.json
if ($LASTEXITCODE -ne 0) { throw "baseline IFEval scoring failed" }
& $PythonExe scripts\score_ifeval.py `
  --input $ifevalGeneration `
  --out artifacts\accuracy\v47-final-ifeval-candidate-scored.json
if ($LASTEXITCODE -ne 0) { throw "candidate IFEval scoring failed" }
& $PythonExe scripts\compare_ifeval.py `
  --baseline artifacts\accuracy\v47-final-ifeval-baseline-scored.json `
  --candidate artifacts\accuracy\v47-final-ifeval-candidate-scored.json `
  --out artifacts\accuracy\v47-final-ifeval-comparison.json
if ($LASTEXITCODE -ne 0) { throw "IFEval comparison failed" }

$humanBaseline = "artifacts\eval\v47-final-humaneval-baseline-fp32.json"
$humanCandidate = "artifacts\eval\v47-final-humaneval-candidate-fp32.json"
if (!(Test-Path -LiteralPath $humanBaseline)) {
  & $PythonExe scripts\run_lm_eval.py `
    --model $source --tokenizer $source --tasks humaneval_generate_local `
    --include-path configs\eval\humaneval_local --out $humanBaseline `
    --predict-only --max-gen-toks 512 --dtype float32 --batch-size 1 `
    --gpu-memory-fraction 0.65
  if ($LASTEXITCODE -ne 0) { throw "baseline HumanEval generation failed" }
}
if (!(Test-Path -LiteralPath $humanCandidate)) {
  & $PythonExe scripts\run_lm_eval.py `
    --model $model --tokenizer $source --key $key `
    --tasks humaneval_generate_local --include-path configs\eval\humaneval_local `
    --out $humanCandidate --predict-only --max-gen-toks 512 `
    --dtype auto --batch-size 1 --gpu-memory-fraction 0.65
  if ($LASTEXITCODE -ne 0) { throw "candidate HumanEval generation failed" }
}
& $PythonExe scripts\evaluate_humaneval_wsl.py --input $humanBaseline `
  --out artifacts\accuracy\v47-final-humaneval-baseline-evaluated.json `
  --wsl-distro Ubuntu-24.04 --timeout 10
if ($LASTEXITCODE -ne 0) { throw "baseline HumanEval execution failed" }
& $PythonExe scripts\evaluate_humaneval_wsl.py --input $humanCandidate `
  --out artifacts\accuracy\v47-final-humaneval-candidate-evaluated.json `
  --wsl-distro Ubuntu-24.04 --timeout 10
if ($LASTEXITCODE -ne 0) { throw "candidate HumanEval execution failed" }
& $PythonExe scripts\compare_humaneval.py `
  --baseline artifacts\accuracy\v47-final-humaneval-baseline-evaluated.json `
  --candidate artifacts\accuracy\v47-final-humaneval-candidate-evaluated.json `
  --out artifacts\accuracy\v47-final-humaneval-comparison.json
if ($LASTEXITCODE -ne 0) { throw "HumanEval comparison failed" }

if (!(Test-Path -LiteralPath $serverPackage)) {
  & .\.venv\Scripts\aloepri.exe build-server-package `
    --source-checkpoint $model --output $serverPackage
  if ($LASTEXITCODE -ne 0) { throw "server package build failed" }
}
& .\.venv\Scripts\aloepri.exe inspect-package --server-package $serverPackage
if ($LASTEXITCODE -ne 0) { throw "server package inspection failed" }

New-Item -ItemType Directory -Force -Path artifacts\product\v47 | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = "artifacts\product\v47\server-$stamp.stdout.log"
$stderrLog = "artifacts\product\v47\server-$stamp.stderr.log"
$serverProcess = Start-Process -FilePath .\.venv\Scripts\aloepri.exe `
  -ArgumentList @("serve", "--config", "configs\product\qwen05b_v47_best_single_candidate.yaml") `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
try {
  $ready = $false
  for ($attempt = 0; $attempt -lt 180; $attempt++) {
    if ($serverProcess.HasExited) { throw "private server exited during startup" }
    try {
      $health = Invoke-RestMethod -Uri http://127.0.0.1:8000/healthz -TimeoutSec 2
      if ($health.status -eq "ok") { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
  }
  if (!$ready) { throw "private server did not become healthy within 180 seconds" }
  & $PythonExe scripts\verify_product_privacy_boundary.py `
    --server http://127.0.0.1:8000 --tokenizer $source `
    --online-key-dir $onlineKey --server-package $serverPackage `
    --server-log $stdoutLog --server-log $stderrLog `
    --out artifacts\product\v47\transport-privacy.json
  if ($LASTEXITCODE -ne 0) { throw "transport privacy verification failed" }
  & $PythonExe scripts\run_product_question_gate.py `
    --server http://127.0.0.1:8000 --tokenizer $source `
    --online-key-dir $onlineKey --server-package $serverPackage `
    --prompts configs\eval\gate1_prompts_200.json `
    --out artifacts\product\v47\question-gate.json --max-new-tokens 64
  if ($LASTEXITCODE -ne 0) { throw "product question gate execution failed" }
} finally {
  if (!$serverProcess.HasExited) {
    Stop-Process -Id $serverProcess.Id -Force
    $serverProcess.WaitForExit(10000)
  }
}

& scripts\run_balanced_hf_performance.ps1 `
  -BaselineModel $source -CandidateModel $model -Tokenizer $source `
  -Prompts configs\eval\benchmark_prompts_20.json -CandidateKey $key `
  -OutputDir artifacts\performance\v47\balanced `
  -PythonExe $PythonExe -Dtype float32 -GpuMemoryFraction 0.65
if ($LASTEXITCODE -ne 0) { throw "balanced performance pipeline failed" }
Copy-Item -LiteralPath artifacts\performance\v47\balanced\balanced-comparison.json `
  -Destination artifacts\performance\v47\comparison.json -Force

& $PythonExe scripts\build_qwen05b_v47_acceptance.py `
  --config configs\acceptance\qwen05b_v47.yaml `
  --out artifacts\acceptance\qwen05b-v47-final.json
if ($LASTEXITCODE -notin @(0, 2)) { throw "acceptance report build failed" }
Write-Output "V47_REMAINING_ACCEPTANCE_COMPLETE"
