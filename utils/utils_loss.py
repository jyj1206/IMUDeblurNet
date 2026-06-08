import math

import torch
import torch.nn.functional as F
from torch import nn


class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 10.0 / math.log(10.0)

    def forward(self, pred, target):
        return (
            self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )


class NegativePSNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = float(eps)
        self.scale = 10.0 / math.log(10.0)

    def forward(self, pred, target):
        mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
        return self.scale * torch.log(mse + self.eps).mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps * self.eps).mean()


def build_scalar_loss(name):
    name = str(name).lower()
    if name in ("psnr", "negative_psnr"):
        return NegativePSNRLoss()
    if name in ("l1", "mae"):
        return nn.L1Loss()
    if name in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss()
    if name in ("charbonnier", "charb"):
        return CharbonnierLoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")


def build_stage1_loss(name, reduction="mean"):
    name = str(name).lower()
    if name == "mse":
        return nn.MSELoss(reduction=reduction)
    if name in ("l1", "mae"):
        return nn.L1Loss(reduction=reduction)
    if name in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss(reduction=reduction)
    raise ValueError(f"Unknown train.loss: {name}")


def build_criterion(name):
    name = str(name).lower()
    if name == "psnr":
        return PSNRLoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown train.loss: {name}")


def timestamp_deltas(timestamp_window, default_dt=1.0 / 240.0):
    if timestamp_window is None:
        raise ValueError("timestamp_window is required for Stage1 auxiliary pose loss.")
    timestamps = timestamp_window.float()
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    median_dt = torch.nanmedian(dt.detach())
    if torch.isfinite(median_dt) and median_dt > 1e6:
        dt = dt * 1e-9
    valid = torch.isfinite(dt) & (dt > 0)
    fallback = torch.full_like(dt, float(default_dt))
    return torch.where(valid, dt, fallback)


def gyro_window_to_integrated_omega(
    gyro_window, timestamp_window, default_dt=1.0 / 240.0
):
    if gyro_window.ndim != 3 or gyro_window.shape[-1] != 3:
        raise ValueError(f"gyro_window must be BxNx3, got {tuple(gyro_window.shape)}")
    dt = timestamp_deltas(timestamp_window, default_dt=default_dt).to(
        device=gyro_window.device,
        dtype=gyro_window.dtype,
    )
    interval_gyro = 0.5 * (gyro_window[:, :-1] + gyro_window[:, 1:])
    theta = (interval_gyro * dt.unsqueeze(-1)).sum(dim=1)
    total_dt = dt.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return theta / total_dt


class Stage1AuxLoss(nn.Module):
    def __init__(
        self,
        gyro_loss="smooth_l1",
        aux_loss="smooth_l1",
        aux_weight=0.05,
        default_dt=1.0 / 240.0,
        target_norm_weight=0.0,
        target_norm_reference=2.5,
        target_norm_max_weight=3.0,
    ):
        super().__init__()
        self.gyro_loss = str(gyro_loss).lower()
        self.aux_loss = str(aux_loss).lower()
        self.aux_weight = float(aux_weight)
        self.default_dt = float(default_dt)
        self.target_norm_weight = float(target_norm_weight)
        self.target_norm_reference = float(target_norm_reference)
        self.target_norm_max_weight = float(target_norm_max_weight)

    def _elementwise_loss(self, pred, target, name):
        if name in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(pred, target, reduction="none")
        if name in ("l1", "mae"):
            return F.l1_loss(pred, target, reduction="none")
        if name == "mse":
            return F.mse_loss(pred, target, reduction="none")
        raise ValueError(f"Unknown loss: {name}")

    def _target_sample_weight(self, target_gyro):
        if self.target_norm_weight <= 0 or self.target_norm_reference <= 0:
            return None
        norm = target_gyro.detach().norm(dim=-1).mean(dim=1)
        extra = (norm / self.target_norm_reference - 1.0).clamp_min(0.0)
        weight = 1.0 + self.target_norm_weight * extra
        return weight.clamp_max(self.target_norm_max_weight)

    def _loss(self, pred, target, name, sample_weight=None):
        elementwise = self._elementwise_loss(pred, target, name)
        per_sample = elementwise.flatten(1).mean(dim=1)
        if sample_weight is None:
            return per_sample.mean()
        sample_weight = sample_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)

    def forward(self, outputs, target_gyro, timestamp_window):
        pred_gyro = outputs["gyro"]
        sample_weight = self._target_sample_weight(target_gyro)
        gyro_loss = self._loss(
            pred_gyro, target_gyro, self.gyro_loss, sample_weight=sample_weight
        )
        pred_pose = outputs.get("pose")

        if pred_pose is None or self.aux_weight <= 0:
            aux_loss = gyro_loss.new_tensor(0.0)
            omega_gt = None
        else:
            omega_gt = gyro_window_to_integrated_omega(
                target_gyro,
                timestamp_window,
                default_dt=self.default_dt,
            )
            aux_loss = self._loss(
                pred_pose[:, :3], omega_gt, self.aux_loss, sample_weight=sample_weight
            )

        total = gyro_loss + self.aux_weight * aux_loss
        abs_diff = (pred_gyro - target_gyro).abs()
        mae = abs_diff.mean()
        axis_mae = abs_diff.mean(dim=(0, 1))
        rmse = torch.sqrt(((pred_gyro - target_gyro) ** 2).mean())
        metrics = {
            "loss": total.detach(),
            "gyro_loss": gyro_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "mae": mae.detach(),
            "gyro_x_mae": axis_mae[0].detach(),
            "gyro_y_mae": axis_mae[1].detach(),
            "gyro_z_mae": axis_mae[2].detach(),
            "rmse": rmse.detach(),
        }
        return total, metrics, omega_gt


def temporal_smoothness(gyro):
    if gyro.shape[1] < 2:
        return gyro.new_tensor(0.0)
    return (gyro[:, 1:, :] - gyro[:, :-1, :]).abs().mean()


def cmf_epe(pred_cmf, target_cmf):
    batch, channels, height, width = pred_cmf.shape
    if channels % 2 != 0:
        raise ValueError(f"CMF channels must be even, got {channels}")
    diff = (pred_cmf - target_cmf).view(batch, channels // 2, 2, height, width)
    return torch.sqrt((diff * diff).sum(dim=2) + 1e-12).mean()


class Stage1Stage2FinetuneLoss(nn.Module):
    def __init__(
        self,
        image_loss="charbonnier",
        gyro_loss="smooth_l1",
        cmf_loss="smooth_l1",
        image_weight=1.0,
        gyro_weight=0.1,
        cmf_weight=0.05,
        cmf_epe_weight=0.0,
        smooth_weight=0.001,
    ):
        super().__init__()
        self.image_loss = build_scalar_loss(image_loss)
        self.gyro_loss = build_scalar_loss(gyro_loss)
        self.cmf_loss = build_scalar_loss(cmf_loss)
        self.image_weight = float(image_weight)
        self.gyro_weight = float(gyro_weight)
        self.cmf_weight = float(cmf_weight)
        self.cmf_epe_weight = float(cmf_epe_weight)
        self.smooth_weight = float(smooth_weight)

    def forward(self, outputs, batch):
        pred_raw = outputs["pred_raw"]
        pred_gyro = outputs["pred_gyro"]
        pred_cmf = outputs["cmf"]
        sharp = batch.get("gt")
        target_gyro = batch.get("gyro")
        target_cmf = batch.get("target_cmf")

        device = pred_raw.device
        zero = pred_raw.new_tensor(0.0)

        if sharp is None:
            image_loss = zero
        else:
            sharp = sharp.to(device=device, dtype=pred_raw.dtype)
            image_loss = self.image_loss(pred_raw, sharp)

        if target_gyro is None or self.gyro_weight <= 0:
            gyro_loss = zero
        else:
            target_gyro = target_gyro.to(device=device, dtype=pred_gyro.dtype)
            gyro_loss = self.gyro_loss(pred_gyro, target_gyro)

        if target_cmf is None or self.cmf_weight <= 0:
            cmf_loss = zero
        else:
            target_cmf = target_cmf.to(device=device, dtype=pred_cmf.dtype)
            cmf_loss = self.cmf_loss(pred_cmf, target_cmf)

        if target_cmf is None or self.cmf_epe_weight <= 0:
            epe_loss = zero
        else:
            target_cmf = target_cmf.to(device=device, dtype=pred_cmf.dtype)
            epe_loss = cmf_epe(pred_cmf, target_cmf)

        smooth_loss = temporal_smoothness(pred_gyro) if self.smooth_weight > 0 else zero
        total = (
            self.image_weight * image_loss
            + self.gyro_weight * gyro_loss
            + self.cmf_weight * cmf_loss
            + self.cmf_epe_weight * epe_loss
            + self.smooth_weight * smooth_loss
        )
        return {
            "loss": total,
            "image_loss": image_loss.detach(),
            "gyro_loss": gyro_loss.detach(),
            "cmf_loss": cmf_loss.detach(),
            "cmf_epe": epe_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
        }


def build_stage1_stage2_finetune_loss(config):
    loss_cfg = config.get("loss", {})
    return Stage1Stage2FinetuneLoss(
        image_loss=loss_cfg.get(
            "image_loss", config.get("train", {}).get("loss", "charbonnier")
        ),
        gyro_loss=loss_cfg.get("gyro_loss", "smooth_l1"),
        cmf_loss=loss_cfg.get("cmf_loss", "smooth_l1"),
        image_weight=loss_cfg.get("image_weight", 1.0),
        gyro_weight=loss_cfg.get("gyro_weight", 0.1),
        cmf_weight=loss_cfg.get("cmf_weight", 0.05),
        cmf_epe_weight=loss_cfg.get("cmf_epe_weight", 0.0),
        smooth_weight=loss_cfg.get("smooth_weight", 0.001),
    )
