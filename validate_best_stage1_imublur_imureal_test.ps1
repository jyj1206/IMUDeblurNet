param(
    [string]$Stage1Checkpoint = "weights/best_freeze_aux_all.pt",
    [string]$IMUBlurRoot = "data/IMUBlur",
    [string]$IMURealRoot = "data/IMURealBlur",
    [string]$Split = "test",
    [string]$OutputRoot = "runs/best_freeze_aux_all",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1
)

$ErrorActionPreference = "Stop"

$commonArgs = @(
    "--checkpoint", $Stage1Checkpoint,
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

Write-Host "== Stage1 | IMUBlur | $Split =="
python validate_stage1.py @commonArgs `
    --dataset-root $IMUBlurRoot

Write-Host "== Stage1 | IMURealBlur | $Split =="
Write-Host "   note: Stage1 validation needs sensor_windows.npy gyro GT."
python validate_stage1.py @commonArgs `
    --dataset-root $IMURealRoot
