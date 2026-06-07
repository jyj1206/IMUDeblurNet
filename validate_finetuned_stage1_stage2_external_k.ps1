param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$GoProRoot = "data\GoPro",
    [string]$RealBlurJRoot = "data\RealBlur_J",
    [string]$Split = "test",
    [string]$OutputRoot = "runs",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1
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

Write-Host "== Fine-tuned Stage1 + Stage2 | GoPro | K = [[960,0,640],[0,960,360],[0,0,1]] =="
python validate_stage1_stage2_finetune.py @commonArgs `
    --dataset-root $GoProRoot `
    --allow-missing-gt `
    --camera-fx 960.0 `
    --camera-fy 960.0 `
    --camera-cx 640.0 `
    --camera-cy 360.0

Write-Host "== Fine-tuned Stage1 + Stage2 | RealBlur_J | K = [[1000,0,344],[0,1000,392],[0,0,1]] =="
python validate_stage1_stage2_finetune.py @commonArgs `
    --dataset-root $RealBlurJRoot `
    --allow-missing-gt `
    --realblur-metrics `
    --camera-fx 1000.0 `
    --camera-fy 1000.0 `
    --camera-cx 344.0 `
    --camera-cy 392.0
