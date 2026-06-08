param(
    [string]$Stage2Checkpoint = "weights\best_stage2_image_only.pt",
    [string]$Stage2Config = "config\stage2_image_only.yaml",
    [string]$IMUBlurRoot = "data\IMUBlur",
    [string]$IMURealRoot = "data\IMURealBlur",
    [string]$GoProRoot = "data\GoPro",
    [string]$RealBlurJRoot = "data\RealBlur_J",
    [string]$Split = "test",
    [string]$OutputRoot = "runs",
    [int]$BatchSize = 1,
    [int]$NumWorkers = 0,
    [int]$MaxBatches = -1,
    [int]$SaveLimit = -1,
    [switch]$NonStrict,
    [switch]$RealBlurJMetrics,
    [switch]$AllowMissingRealBlurJ
)

$ErrorActionPreference = "Stop"

$commonArgs = @(
    "--config", $Stage2Config,
    "--checkpoint", $Stage2Checkpoint,
    "--split", $Split,
    "--output-root", $OutputRoot,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--image-only"
)

if ($NonStrict) {
    $commonArgs += "--non-strict"
}
if ($MaxBatches -ge 0) {
    $commonArgs += @("--max-batches", $MaxBatches)
}
if ($SaveLimit -ge 0) {
    $commonArgs += @("--save-limit", $SaveLimit)
}

Write-Host "== Stage2 image-only | IMUBlur | $Split =="
python validate_stage2.py @commonArgs `
    --dataset-root $IMUBlurRoot

Write-Host "== Stage2 image-only | IMURealBlur | $Split =="
python validate_stage2.py @commonArgs `
    --dataset-root $IMURealRoot `
    --allow-missing-gt `
    --realblur-metrics

Write-Host "== Stage2 image-only | GoPro | $Split =="
python validate_stage2.py @commonArgs `
    --dataset-root $GoProRoot

$realBlurJArgs = @($commonArgs)
if ($RealBlurJMetrics) {
    $realBlurJArgs += "--realblur-metrics"
}
if ($AllowMissingRealBlurJ) {
    $realBlurJArgs += "--allow-missing-gt"
}

Write-Host "== Stage2 image-only | RealBlur_J | $Split =="
python validate_stage2.py @realBlurJArgs `
    --dataset-root $RealBlurJRoot
