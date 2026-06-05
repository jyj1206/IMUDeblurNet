import torch
import torch.nn as nn
import torch.nn.functional as F


def timestamp_deltas(timestamp_window, default_dt=1.0 / 240.0):
    if timestamp_window is None:
        raise ValueError("timestamp_window is required for IAAI auxiliary pose loss.")
    timestamps = timestamp_window.float()
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    median_dt = torch.nanmedian(dt.detach())
    if torch.isfinite(median_dt) and median_dt > 1e6:
        dt = dt * 1e-9
    valid = torch.isfinite(dt) & (dt > 0)
    fallback = torch.full_like(dt, float(default_dt))
    return torch.where(valid, dt, fallback)


def gyro_window_to_integrated_omega(gyro_window, timestamp_window, default_dt=1.0 / 240.0):
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


class IAAIGyroAuxLoss(nn.Module):
    def __init__(
        self,
        gyro_loss="smooth_l1",
        aux_loss="smooth_l1",
        aux_weight=0.05,
        default_dt=1.0 / 240.0,
    ):
        super().__init__()
        self.gyro_loss = str(gyro_loss).lower()
        self.aux_loss = str(aux_loss).lower()
        self.aux_weight = float(aux_weight)
        self.default_dt = float(default_dt)

    def _loss(self, pred, target, name):
        if name in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(pred, target)
        if name in ("l1", "mae"):
            return F.l1_loss(pred, target)
        if name == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unknown loss: {name}")

    def forward(self, outputs, target_gyro, timestamp_window):
        pred_gyro = outputs["gyro"]
        gyro_loss = self._loss(pred_gyro, target_gyro, self.gyro_loss)
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
            aux_loss = self._loss(pred_pose[:, :3], omega_gt, self.aux_loss)

        total = gyro_loss + self.aux_weight * aux_loss
        mae = (pred_gyro - target_gyro).abs().mean()
        rmse = torch.sqrt(((pred_gyro - target_gyro) ** 2).mean())
        metrics = {
            "loss": total.detach(),
            "gyro_loss": gyro_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "mae": mae.detach(),
            "rmse": rmse.detach(),
        }
        return total, metrics, omega_gt
