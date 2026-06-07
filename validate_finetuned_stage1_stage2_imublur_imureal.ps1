param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
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
    "--checkpoint", $Checkpoint,
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

Write-Host "== Fine-tuned Stage1 + Stage2 | IMUBlur | $Split =="
python validate_stage1_stage2_finetune.py @commonArgs `
    --dataset-root $IMUBlurRoot `
    --load-target-gyro

$imuRealArgs = @($commonArgs)
if ($LoadIMURealTargetGyro) {
    $imuRealArgs += "--load-target-gyro"
}

Write-Host "== Fine-tuned Stage1 + Stage2 | IMURealBlur | $Split =="
python validate_stage1_stage2_finetune.py @imuRealArgs `
    --dataset-root $IMURealRoot `
    --allow-missing-gt `
    --realblur-metrics
