from pathlib import Path

import cv2
import numpy as np
import torch


def write_image(path, image_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image_bgr)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def tensor_to_rgb_uint8(image, mean=None, std=None):
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu()
        if image.ndim == 4:
            image = image[0]
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
            image = image * std_t + mean_t
        image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    return image


def _resize_to_height(image, height):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * height / h)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _put_text(image, text, org, scale=0.55, color=(245, 245, 245), thickness=1):
    cv2.putText(
        image,
        str(text),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def make_stage2_comparison(blur, pred, sharp, psnr=None, ssim=None, title=None, max_panel_height=360):
    blur = tensor_to_rgb_uint8(blur)
    pred = tensor_to_rgb_uint8(pred)
    sharp = tensor_to_rgb_uint8(sharp)

    h = min(max_panel_height, max(1, blur.shape[0]))
    panels = [_resize_to_height(img, h) for img in (blur, pred, sharp)]
    panel_w = min(panel.shape[1] for panel in panels)
    panels = [cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_AREA) for panel in panels]

    body = np.concatenate(panels, axis=1)
    body_bgr = cv2.cvtColor(body, cv2.COLOR_RGB2BGR)
    header = np.full((44, body_bgr.shape[1], 3), 28, dtype=np.uint8)

    metric = ""
    if psnr is not None:
        metric += f"PSNR {float(psnr):.2f} dB"
    if ssim is not None:
        metric += (" | " if metric else "") + f"SSIM {float(ssim):.4f}"
    if title:
        metric = f"{title}   {metric}" if metric else title
    _put_text(header, metric, (12, 28), scale=0.62, color=(245, 245, 245), thickness=1)

    labels = ["Blur", "Deblur", "Sharp"]
    for idx, label in enumerate(labels):
        x0 = idx * panel_w
        cv2.rectangle(body_bgr, (x0, 0), (x0 + 90, 28), (0, 0, 0), -1)
        _put_text(body_bgr, label, (x0 + 8, 20), scale=0.55)

    return np.concatenate([header, body_bgr], axis=0)


def make_stage1_gyro_visualization(
    image,
    pred_gyro,
    target_gyro=None,
    title=None,
    mean=None,
    std=None,
    max_image_height=520,
):
    image = tensor_to_rgb_uint8(image, mean=mean, std=std)
    if image.shape[0] > max_image_height:
        image = _resize_to_height(image, max_image_height)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred = _to_numpy_sequence(pred_gyro)
    target = _to_numpy_sequence(target_gyro) if target_gyro is not None else None
    panel_h = 260
    panel = np.full((panel_h, image_bgr.shape[1], 3), 34, dtype=np.uint8)

    header = np.full((44, image_bgr.shape[1], 3), 26, dtype=np.uint8)
    metric = ""
    if target is not None:
        metric = f"gyro MAE {np.abs(pred - target).mean():.6f}"
    if title:
        metric = f"{title}   {metric}" if metric else title
    _put_text(header, metric, (12, 28), scale=0.62)

    all_values = pred
    if target is not None:
        all_values = np.concatenate([pred, target], axis=0)
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    pad = max((value_max - value_min) * 0.1, 1e-4)
    value_min -= pad
    value_max += pad

    plot_left = 56
    plot_right = image_bgr.shape[1] - 18
    plot_top = 34
    plot_bottom = panel_h - 46
    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (78, 78, 78), 1)
    zero_y = _value_to_y(0.0, value_min, value_max, plot_top, plot_bottom)
    cv2.line(panel, (plot_left, zero_y), (plot_right, zero_y), (60, 60, 60), 1, cv2.LINE_AA)

    colors = [(80, 210, 255), (90, 230, 120), (245, 170, 85)]
    labels = ["gyro x", "gyro y", "gyro z"]
    for axis_idx, (color, label) in enumerate(zip(colors, labels)):
        _draw_series(panel, pred[:, axis_idx], value_min, value_max, plot_left, plot_right, plot_top, plot_bottom, color, 2)
        if target is not None:
            _draw_series(panel, target[:, axis_idx], value_min, value_max, plot_left, plot_right, plot_top, plot_bottom, color, 1, dashed=True)
        _put_text(panel, label, (plot_left + axis_idx * 92, 22), scale=0.46, color=color)

    _put_text(panel, "pred: solid", (plot_left, panel_h - 20), scale=0.45, color=(225, 225, 225))
    if target is not None:
        _put_text(panel, "target: dashed", (plot_left + 105, panel_h - 20), scale=0.45, color=(180, 180, 180))
    _put_text(panel, f"{value_max:+.3g}", (6, plot_top + 6), scale=0.38, color=(180, 180, 180))
    _put_text(panel, f"{value_min:+.3g}", (6, plot_bottom), scale=0.38, color=(180, 180, 180))

    return np.concatenate([header, image_bgr, panel], axis=0)


def make_stage1_v_visualization(*args, **kwargs):
    if "target_v" in kwargs and "target_gyro" not in kwargs:
        kwargs["target_gyro"] = kwargs.pop("target_v")
    return make_stage1_gyro_visualization(*args, **kwargs)


def make_cmf_visualization(
    image,
    motion_field,
    title=None,
    max_panel_height=540,
    grid_step=48,
    trajectory_scale=1.0,
):
    image = tensor_to_rgb_uint8(image)
    if image.shape[0] > max_panel_height:
        scale = max_panel_height / image.shape[0]
        image = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max_panel_height),
            interpolation=cv2.INTER_AREA,
        )
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cmf = _to_numpy_cmf(motion_field)
    cmf_h, cmf_w, channels = cmf.shape
    vector_count = channels // 2
    scale_y = canvas.shape[0] / max(cmf_h, 1)
    scale_x = canvas.shape[1] / max(cmf_w, 1)
    step_y = max(1, int(round(grid_step / max(scale_y, 1e-6))))
    step_x = max(1, int(round(grid_step / max(scale_x, 1e-6))))

    for y in range(step_y // 2, cmf_h, step_y):
        for x in range(step_x // 2, cmf_w, step_x):
            base_x = x * scale_x
            base_y = y * scale_y
            points = [(int(round(base_x)), int(round(base_y)))]
            cur_x = base_x
            cur_y = base_y
            for idx in range(vector_count):
                dx = float(cmf[y, x, idx * 2]) * scale_x * trajectory_scale
                dy = float(cmf[y, x, idx * 2 + 1]) * scale_y * trajectory_scale
                cur_x += dx
                cur_y += dy
                points.append((int(round(cur_x)), int(round(cur_y))))
            for p0, p1 in zip(points[:-1], points[1:]):
                cv2.line(canvas, p0, p1, (60, 230, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, points[0], 2, (40, 120, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, points[-1], 2, (80, 255, 120), -1, lineType=cv2.LINE_AA)

    header = np.full((44, canvas.shape[1], 3), 26, dtype=np.uint8)
    label = f"CMF / paper V   shape={cmf_h}x{cmf_w}x{channels}"
    if title:
        label = f"{title}   {label}"
    _put_text(header, label, (12, 28), scale=0.58)
    return np.concatenate([header, canvas], axis=0)


def _to_numpy_sequence(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    return value


def _to_numpy_cmf(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"CMF must be CHW or HWC, got {value.shape}")
    if value.shape[0] <= 64 and value.shape[0] < value.shape[-1]:
        value = value.transpose(1, 2, 0)
    return value


def _value_to_y(value, value_min, value_max, top, bottom):
    ratio = (float(value) - value_min) / max(value_max - value_min, 1e-8)
    return int(round(bottom - ratio * (bottom - top)))


def _draw_series(canvas, values, value_min, value_max, left, right, top, bottom, color, thickness, dashed=False):
    values = np.asarray(values, dtype=np.float32)
    xs = np.linspace(left, right, len(values))
    points = [
        (int(round(x)), _value_to_y(value, value_min, value_max, top, bottom))
        for x, value in zip(xs, values)
    ]
    for idx, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        if dashed and idx % 2:
            continue
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, point, max(1, thickness), color, -1, lineType=cv2.LINE_AA)
