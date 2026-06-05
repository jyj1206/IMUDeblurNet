import torch

from models.stage1_iaai_gyro_model import build_stage1_iaai_model
from models.stage2_deblur_model import build_model as build_stage2_model
from utils.utils_eval import load_model_weights
from utils.utils_stage_pipeline import camera_matrix_from_config, predicted_gyro_to_cmf, resolve_device


def load_stage1_iaai_stage2_models(
    stage1_config,
    stage2_config,
    stage1_checkpoint,
    stage2_checkpoint,
    device,
    strict_stage1=True,
    strict_stage2=True,
):
    stage1_model = build_stage1_iaai_model(stage1_config).to(device).eval()
    stage2_model = build_stage2_model(stage2_config).to(device).eval()
    stage1_report = load_model_weights(
        stage1_model,
        stage1_checkpoint,
        device=device,
        strict=strict_stage1,
    )
    stage2_report = load_model_weights(
        stage2_model,
        stage2_checkpoint,
        device=device,
        strict=strict_stage2,
    )
    return stage1_model, stage2_model, {"stage1": stage1_report, "stage2": stage2_report}


@torch.no_grad()
def run_stage1_iaai_stage2_batch(
    stage1_model,
    stage2_model,
    batch,
    stage2_config,
    device,
    default_dt=1.0 / 240.0,
    camera_matrix=None,
    return_aux=True,
):
    stage1_image = batch["stage1_image"].to(device, non_blocking=True).float()
    blur = batch["lq"].to(device, non_blocking=True).float()
    timestamp_windows = batch["timestamp_window"]
    focal_length = batch.get("focal_length")
    if focal_length is not None:
        focal_length = focal_length.to(device, non_blocking=True).float()

    stage1_out = stage1_model(stage1_image, focal_length=focal_length, return_aux=return_aux)
    pred_gyro = stage1_out["gyro"]
    cmf = predicted_gyro_to_cmf(
        pred_gyro,
        image_hw=blur.shape[-2:],
        timestamp_windows=timestamp_windows,
        downsample=stage2_config.get("dataset", {}).get("motion_downsample", 2),
        default_dt=default_dt,
        camera_matrix=camera_matrix,
        device=device,
    )
    pred_raw = stage2_model(blur, cmf)
    return {
        "stage1": stage1_out,
        "pred_gyro": pred_gyro,
        "cmf": cmf,
        "motion_field": cmf,
        "pred_raw": pred_raw,
        "pred": pred_raw.clamp(0.0, 1.0),
    }
