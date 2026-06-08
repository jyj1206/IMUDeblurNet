import torch


DEFAULT_CAMERA_K = (
    (923.7181693, 0.0, 969.4457779),
    (0.0, 924.51235192, 532.9090534),
    (0.0, 0.0, 1.0),
)


def camera_matrix_tensor(camera_matrix=None, device=None, dtype=None):
    if camera_matrix is None:
        camera_matrix = DEFAULT_CAMERA_K
    return torch.as_tensor(camera_matrix, device=device, dtype=dtype or torch.float32)


def build_center_vectors_torch(
    height,
    width,
    downsample=2,
    batch_size=None,
    origin_yx=None,
    device=None,
    dtype=None,
):
    dtype = dtype or torch.float32
    ys = torch.arange(0, int(height), int(downsample), device=device, dtype=dtype)
    xs = torch.arange(0, int(width), int(downsample), device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

    if origin_yx is not None:
        origin_yx = torch.as_tensor(origin_yx, device=device, dtype=dtype)
        if origin_yx.ndim == 1:
            origin_yx = origin_yx.view(1, 2)
        x_grid = x_grid.unsqueeze(0) + origin_yx[:, 1].view(-1, 1, 1)
        y_grid = y_grid.unsqueeze(0) + origin_yx[:, 0].view(-1, 1, 1)
        ones = torch.ones_like(x_grid)
        return torch.stack((x_grid, y_grid, ones), dim=-1)

    center_vectors = torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1)
    if batch_size is not None:
        center_vectors = center_vectors.unsqueeze(0).expand(int(batch_size), -1, -1, -1)
    return center_vectors


def compute_rotation_matrix_torch(theta):
    if theta.ndim != 2 or theta.shape[-1] != 3:
        raise ValueError(f"theta must be Bx3, got {tuple(theta.shape)}")

    batch = theta.shape[0]
    device = theta.device
    dtype = theta.dtype
    x, y, z = theta[:, 0], theta[:, 1], theta[:, 2]
    sx, cx = torch.sin(x), torch.cos(x)
    sy, cy = torch.sin(y), torch.cos(y)
    sz, cz = torch.sin(z), torch.cos(z)
    zeros = torch.zeros(batch, device=device, dtype=dtype)
    ones = torch.ones(batch, device=device, dtype=dtype)

    r_x = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cx, sx), dim=-1),
            torch.stack((zeros, -sx, cx), dim=-1),
        ),
        dim=1,
    )
    r_y = torch.stack(
        (
            torch.stack((cy, zeros, -sy), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((sy, zeros, cy), dim=-1),
        ),
        dim=1,
    )
    r_z = torch.stack(
        (
            torch.stack((cz, sz, zeros), dim=-1),
            torch.stack((-sz, cz, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=1,
    )
    return r_x @ r_y @ r_z


def compute_interval_rotations_torch(
    gyro_window, timestamp_window, default_dt=1.0 / 240.0
):
    if gyro_window.ndim != 3 or gyro_window.shape[-1] != 3:
        raise ValueError(f"gyro_window must be BxNx3, got {tuple(gyro_window.shape)}")
    if gyro_window.shape[1] != 7:
        raise ValueError(
            f"gyro_window must contain 7 vectors, got {gyro_window.shape[1]}"
        )

    timestamp_window = timestamp_window.to(
        device=gyro_window.device, dtype=gyro_window.dtype
    )
    if timestamp_window.ndim == 1:
        timestamp_window = timestamp_window.unsqueeze(0).expand(
            gyro_window.shape[0], -1
        )
    if timestamp_window.shape[1] != gyro_window.shape[1]:
        raise ValueError(
            f"timestamp_window shape must be Bx{gyro_window.shape[1]}, got {tuple(timestamp_window.shape)}"
        )

    dt = timestamp_window[:, 1:] - timestamp_window[:, :-1]
    median_dt = torch.nanmedian(dt.detach(), dim=1).values
    scale = torch.where(median_dt > 1e6, dt.new_tensor(1e-9), dt.new_tensor(1.0))
    dt = dt * scale.view(-1, 1)
    valid = torch.isfinite(dt) & (dt > 0)
    dt = torch.where(valid, dt, dt.new_full(dt.shape, float(default_dt)))

    theta = 0.5 * (gyro_window[:, :-1, :] + gyro_window[:, 1:, :]) * dt.unsqueeze(-1)
    return [
        compute_rotation_matrix_torch(theta[:, idx, :]) for idx in range(theta.shape[1])
    ]


def compute_homography_torch(rotation, camera_matrix):
    batch = rotation.shape[0]
    camera_matrix = camera_matrix.to(device=rotation.device, dtype=rotation.dtype)
    if camera_matrix.ndim == 2:
        camera_matrix = camera_matrix.unsqueeze(0).expand(batch, -1, -1)
    k_inv = torch.linalg.inv(camera_matrix)
    return camera_matrix @ rotation @ k_inv


def _project_vectors(homography, center_vectors, eps=1e-8):
    projected = torch.einsum("bij,bhwj->bhwi", homography, center_vectors)
    denom = projected[..., 2:3]
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    denom = torch.where(denom.abs() < eps, sign * eps, denom)
    projected = projected / denom
    return projected[..., :2] - center_vectors[..., :2]


def make_camera_motion_field_torch(
    gyro_window,
    timestamp_window,
    height,
    width,
    downsample=2,
    default_dt=1.0 / 240.0,
    camera_matrix=None,
    origin_yx=None,
    eps=1e-8,
):
    gyro_window = gyro_window.float()
    batch = gyro_window.shape[0]
    device = gyro_window.device
    dtype = gyro_window.dtype
    camera_matrix = camera_matrix_tensor(camera_matrix, device=device, dtype=dtype)
    center_vectors = build_center_vectors_torch(
        height=height,
        width=width,
        downsample=downsample,
        batch_size=batch,
        origin_yx=origin_yx,
        device=device,
        dtype=dtype,
    )
    if center_vectors.ndim == 3:
        center_vectors = center_vectors.unsqueeze(0).expand(batch, -1, -1, -1)

    r_list = compute_interval_rotations_torch(
        gyro_window=gyro_window,
        timestamp_window=timestamp_window,
        default_dt=default_dt,
    )

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    r = eye
    pro_vectors = []
    for idx in range(len(r_list) // 2, len(r_list)):
        r = r_list[idx] @ r
        h_pro = compute_homography_torch(r, camera_matrix)
        pro_vectors.append(_project_vectors(h_pro, center_vectors, eps=eps))

    r = eye
    pre_vectors = []
    for idx in range((len(r_list) // 2) - 1, -1, -1):
        r = r @ r_list[idx]
        h_pre = torch.linalg.inv(compute_homography_torch(r, camera_matrix))
        pre_vectors.insert(0, _project_vectors(h_pre, center_vectors, eps=eps))

    pairs = pre_vectors + pro_vectors
    adjusted = [pair for pair in pairs]
    adjusted[5] = adjusted[5] - adjusted[4]
    adjusted[4] = adjusted[4] - adjusted[3]
    adjusted[0] = adjusted[0] - adjusted[1]
    adjusted[1] = adjusted[1] - adjusted[2]
    adjusted = [-pair if idx < 3 else pair for idx, pair in enumerate(adjusted)]
    cmf = torch.cat(adjusted, dim=-1)
    return cmf.permute(0, 3, 1, 2).contiguous()
