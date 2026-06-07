param(
    [string]$Stage1Checkpoint = "weights\best_stage1.pt",
    [string]$Stage2Checkpoint = "weights\best_stage2.pt",
    [string]$IMUBlurRoot = "data\IMUBlur",
    [string]$IMURealRoot = "data\IMURealBlur",
    [string]$Split = "test",
    [string]$OutputRoot = "runs",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1,
    [switch]$LoadIMURealTargetGyro
)

$ErrorActionPreference = "Stop"

$commonArgs = @(
    "--stage1-checkpoint", $Stage1Checkpoint,
    "--stage2-checkpoint", $Stage2Checkpoint,
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

Write-Host "== Stage1 + Stage2 | IMUBlur | $Split =="
python validate_stage1_stage2.py @commonArgs `
    --dataset-root $IMUBlurRoot `
    --load-target-gyro

$imuRealArgs = @($commonArgs)
if ($LoadIMURealTargetGyro) {
    $imuRealArgs += "--load-target-gyro"
}

Write-Host "== Stage1 + Stage2 | IMURealBlur | $Split =="
python validate_stage1_stage2.py @imuRealArgs `
    --dataset-root $IMURealRoot `
    --allow-missing-gt `
    --realblur-metrics
