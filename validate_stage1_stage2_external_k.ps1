param(
    [string]$Stage1Checkpoint = "weights\best_stage1_V2.pt",
    [string]$Stage2Checkpoint = "weights\best_stage2_V2.pt",
    [string]$GoProRoot = "data\GoPro",
    [string]$RealBlurJRoot = "data\RealBlur_J",
    [string]$Split = "test",
    [string]$OutputRoot = "runs",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1,
    [switch]$RealBlurMetrics,
    [switch]$LoadTargetGyro
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

if ($LoadTargetGyro) {
    $commonArgs += "--load-target-gyro"
}

Write-Host "== GoPro | K = [[960,0,640],[0,960,360],[0,0,1]] =="
python validate_stage1_stage2.py @commonArgs `
    --dataset-root $GoProRoot `
    --camera-fx 960.0 `
    --camera-fy 960.0 `
    --camera-cx 640.0 `
    --camera-cy 360.0

$realBlurArgs = @($commonArgs)
if ($RealBlurMetrics) {
    $realBlurArgs += "--realblur-metrics"
}

Write-Host "== RealBlur_J | K = [[1000,0,344],[0,1000,392],[0,0,1]] =="
python validate_stage1_stage2.py @realBlurArgs `
    --dataset-root $RealBlurJRoot `
    --camera-fx 1000.0 `
    --camera-fy 1000.0 `
    --camera-cx 344.0 `
    --camera-cy 392.0
