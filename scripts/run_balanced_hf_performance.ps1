param(
  [Parameter(Mandatory=$true)][string]$BaselineModel,
  [Parameter(Mandatory=$true)][string]$CandidateModel,
  [Parameter(Mandatory=$true)][string]$Tokenizer,
  [Parameter(Mandatory=$true)][string]$Prompts,
  [string]$CandidateKey = "",
  [string]$PairedInputs = "",
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [int]$MaxNewTokens = 100,
  [int]$Warmup = 2,
  [ValidateSet("float32", "bfloat16")][string]$Dtype = "float32",
  [double]$GpuMemoryFraction = 0.65
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$runs = @(
  @{Order=1; Role="baseline";  Pair="abba-1"},
  @{Order=2; Role="candidate"; Pair="abba-1"},
  @{Order=3; Role="candidate"; Pair="abba-2"},
  @{Order=4; Role="baseline";  Pair="abba-2"},
  @{Order=5; Role="candidate"; Pair="baab-1"},
  @{Order=6; Role="baseline";  Pair="baab-1"},
  @{Order=7; Role="baseline";  Pair="baab-2"},
  @{Order=8; Role="candidate"; Pair="baab-2"}
)

$baselineFiles = @()
$candidateFiles = @()
foreach ($run in $runs) {
  $model = if ($run.Role -eq "baseline") { $BaselineModel } else { $CandidateModel }
  $out = Join-Path $OutputDir ("run-{0:D2}-{1}-{2}.json" -f $run.Order, $run.Role, $run.Pair)
  $arguments = @(
    "scripts\benchmark_hf.py",
    "--model", $model,
    "--tokenizer", $Tokenizer,
    "--prompts", $Prompts,
    "--dtype", $Dtype,
    "--gpu-memory-fraction", $GpuMemoryFraction,
    "--max-new-tokens", $MaxNewTokens,
    "--warmup", $Warmup,
    "--run-id", $run.Pair,
    "--run-order", $run.Order,
    "--model-role", $run.Role,
    "--out", $out
  )
  if ($PairedInputs) {
    $arguments += @("--paired-inputs", $PairedInputs)
  } elseif ($run.Role -eq "candidate") {
    if (-not $CandidateKey) { throw "CandidateKey is required without PairedInputs" }
    $arguments += @("--key", $CandidateKey)
  }
  if ($run.Role -eq "candidate") {
    $candidateFiles += $out
  } else {
    $baselineFiles += $out
  }
  & $PythonExe @arguments
  if ($LASTEXITCODE -ne 0) { throw "benchmark run $($run.Order) failed" }
}

$comparison = Join-Path $OutputDir "balanced-comparison.json"
& $PythonExe scripts\compare_performance.py `
  --baseline $baselineFiles `
  --candidate $candidateFiles `
  --out $comparison
if ($LASTEXITCODE -ne 0) { throw "performance comparison failed" }
Write-Output $comparison
