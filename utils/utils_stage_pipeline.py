import numpy as np
import torch

from generate_camera_motion_field import (
    build_center_vectors,
    camera_matrix_from_values,
    make_camera_motion_field,
)
from models.stage1_model import build_stage1_model
from models.stage2_deblur_model import build_model as build_stage2_model
from utils.utils_eval import load_model_weights


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def camera_matrix_from_config(config=None, fx=None, fy=None, cx=None, cy=None):
    camera_cfg = {}
    if isinstance(config, dict):
        camera_cfg = config.get("camera", {}) or {}
    return camera_matrix_from_values(
        fx=fx if fx is not None else camera_cfg.get("fx"),
        fy=fy if fy is not None else camera_cfg.get("fy"),
        cx=cx if cx is not None else camera_cfg.get("cx"),
        cy=cy if cy is not None else camera_cfg.get("cy"),
    )


def load_stage1_stage2_models(
    stage1_config,
    stage2_config,
    stage1_checkpoint,
    stage2_checkpoint,
    device,
    strict_stage1=True,
    strict_stage2=True,
):
    stage1_model = build_stage1_model(stage1_config).to(device).eval()
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
    return (
        stage1_model,
        stage2_model,
        {"stage1": stage1_report, "stage2": stage2_report},
    )


def predicted_gyro_to_cmf(
    pred_gyro,
    image_hw,
    timestamp_windows,
    downsample=2,
    default_dt=1.0 / 240.0,
    camera_matrix=None,
    device=None,
):
    if pred_gyro.ndim != 3:
        raise ValueError(f"pred_gyro must be BxNx3, got {tuple(pred_gyro.shape)}")

    height, width = int(image_hw[0]), int(image_hw[1])
    center_vectors = build_center_vectors(height, width, int(downsample))
    pred_gyro_np = pred_gyro.detach().float().cpu().numpy()
    timestamp_np = timestamp_windows.detach().float().cpu().numpy()

    motion_fields = []
    for gyro_window, timestamp_window in zip(pred_gyro_np, timestamp_np):
        motion_field = make_camera_motion_field(
            gyro_window=gyro_window,
            timestamp_window=timestamp_window,
            center_vectors=center_vectors,
            default_dt=default_dt,
            camera_matrix=camera_matrix,
        )
        motion_field = motion_field.astype(np.float32).transpose(2, 0, 1)
        motion_fields.append(torch.from_numpy(np.ascontiguousarray(motion_field)))

    motion = torch.stack(motion_fields, dim=0)
    return motion.to(device=device or pred_gyro.device, dtype=pred_gyro.dtype)


@torch.no_grad()
def run_stage1_stage2_batch(
    stage1_model,
    stage2_model,
    batch,
    stage2_config,
    device,
    default_dt=1.0 / 240.0,
    camera_matrix=None,
    return_aux=False,
):
    stage1_image = batch["stage1_image"].to(device, non_blocking=True).float()
    blur = batch["lq"].to(device, non_blocking=True).float()
    timestamp_windows = batch["timestamp_window"]
    focal_length = batch.get("focal_length")
    if focal_length is not None:
        focal_length = focal_length.to(device, non_blocking=True).float()

    stage1_out = stage1_model(
        stage1_image, focal_length=focal_length, return_aux=return_aux
    )
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
