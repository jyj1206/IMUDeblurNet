import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batch_focal(focal_length, batch_size, device, dtype):
    if focal_length is None:
        return torch.ones(batch_size, device=device, dtype=dtype)
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.as_tensor(focal_length, device=device, dtype=dtype)
    focal_length = focal_length.to(device=device, dtype=dtype)
    if focal_length.ndim == 0:
        focal_length = focal_length.repeat(batch_size)
    return focal_length.view(batch_size).clamp_min(1e-6)


class DifferentiablePoseSolver(nn.Module):
    """Solve exposure-level camera motion from optical flow and depth.

    The output layout is [omega_x, omega_y, omega_z, trans_x, trans_y, trans_z].
    It is intended as an auxiliary branch; Stage2 still consumes the gyro-window head.
    """

    def __init__(self, ridge=1e-4, max_points=4096):
        super().__init__()
        self.ridge = float(ridge)
        self.max_points = int(max_points)

    def forward(self, flow, depth, focal_length=None):
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"flow must be Bx2xHxW, got {tuple(flow.shape)}")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"depth must be Bx1xHxW, got {tuple(depth.shape)}")

        batch_size, _, height, width = flow.shape
        if depth.shape[-2:] != (height, width):
            depth = F.interpolate(
                depth, size=(height, width), mode="bilinear", align_corners=False
            )

        if self.max_points > 0 and height * width > self.max_points:
            stride = int((height * width / self.max_points) ** 0.5)
            stride = max(1, stride)
            flow = flow[:, :, ::stride, ::stride]
            depth = depth[:, :, ::stride, ::stride]
            height, width = flow.shape[-2:]

        device = flow.device
        dtype = flow.dtype
        focal = _as_batch_focal(focal_length, batch_size, device, dtype)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        x = (xx - cx).reshape(1, -1).expand(batch_size, -1)
        y = (yy - cy).reshape(1, -1).expand(batch_size, -1)
        f = focal[:, None]

        z = depth[:, 0].reshape(batch_size, -1).clamp_min(1e-3)
        u = flow[:, 0].reshape(batch_size, -1)
        v = flow[:, 1].reshape(batch_size, -1)

        zeros = torch.zeros_like(x)
        ones_over_z = 1.0 / z

        a_u = torch.stack(
            [
                x * y / f,
                -(f + (x * x / f)),
                y,
                -f * ones_over_z,
                zeros,
                x * ones_over_z,
            ],
            dim=-1,
        )
        a_v = torch.stack(
            [
                f + (y * y / f),
                -(x * y / f),
                -x,
                zeros,
                -f * ones_over_z,
                y * ones_over_z,
            ],
            dim=-1,
        )
        a = torch.cat([a_u, a_v], dim=1)
        b = torch.cat([u, v], dim=1).unsqueeze(-1)

        eye = torch.eye(6, device=device, dtype=dtype).unsqueeze(0)
        ata = a.transpose(1, 2) @ a
        atb = a.transpose(1, 2) @ b
        solution = torch.linalg.solve(ata + self.ridge * eye, atb).squeeze(-1)
        omega = solution[:, 0:3]
        translation = solution[:, 3:6]
        return torch.cat([omega, translation], dim=1)
