param(
    [string]$Stage2Checkpoint = "weights\best_stage2.pt",
    [string]$IMUBlurRoot = "data\IMUBlur",
    [string]$IMURealRoot = "data\IMURealBlur",
    [string]$Split = "test",
    [string]$OutputRoot = "runs",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1
)

$ErrorActionPreference = "Stop"

$commonArgs = @(
    "--checkpoint", $Stage2Checkpoint,
    "--split", $Split,
    "--output-root", $OutputRoot,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers
)

if ($MaxBatches -ge 0) {
    $commonArgs += @("--max-batches", $MaxBatches)
}
if ($SaveLimit -ge 0) {
    $commonArgs += @("--save-limit", $SaveLimit)
}

Write-Host "== Stage2 | IMUBlur | $Split =="
python validate_stage2.py @commonArgs `
    --dataset-root $IMUBlurRoot

Write-Host "== Stage2 | IMURealBlur | $Split =="
python validate_stage2.py @commonArgs `
    --dataset-root $IMURealRoot `
    --allow-missing-gt `
    --realblur-metrics
