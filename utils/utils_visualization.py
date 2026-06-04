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


def make_stage2_comparison(blur, pred, sharp, psnr=None, ssim=None, title=None, max_panel_height=520):
    blur = tensor_to_rgb_uint8(blur)
    pred = tensor_to_rgb_uint8(pred)
    sharp = tensor_to_rgb_uint8(sharp)

    h = min(max_panel_height, max(1, blur.shape[0]))
    panels = [_resize_to_height(img, h) for img in (sharp, blur, pred)]
    panel_w = min(panel.shape[1] for panel in panels)
    panels = [cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_AREA) for panel in panels]

    separator_w = 8
    separator = np.full((h, separator_w, 3), 245, dtype=np.uint8)
    body_rgb = np.concatenate(
        [panels[0], separator, panels[1], separator, panels[2]],
        axis=1,
    )
    body_bgr = cv2.cvtColor(body_rgb, cv2.COLOR_RGB2BGR)

    label_h = 34
    header = np.full((label_h, body_bgr.shape[1], 3), 245, dtype=np.uint8)
    labels = ["Sharp", "Blur", "Inference"]
    x_offsets = [0, panel_w + separator_w, (panel_w + separator_w) * 2]
    for label, x0 in zip(labels, x_offsets):
        _put_text(
            header,
            label,
            (x0 + 18, 23),
            scale=0.62,
            color=(35, 35, 35),
            thickness=1,
        )

    if title:
        _put_text(
            header,
            title,
            (max(0, body_bgr.shape[1] - 420), 23),
            scale=0.48,
            color=(85, 85, 85),
            thickness=1,
        )

    metric_lines = []
    if psnr is not None:
        metric_lines.append(f"PSNR {float(psnr):.2f} dB")
    if ssim is not None:
        metric_lines.append(f"SSIM {float(ssim):.4f}")
    if metric_lines:
        text_lines = metric_lines
        pad_x = 10
        pad_y = 8
        line_h = 21
        box_w = 138
        box_h = pad_y * 2 + line_h * len(text_lines)
        x1 = body_bgr.shape[1] - 12
        y0 = 12
        x0 = x1 - box_w
        y1 = y0 + box_h
        overlay = body_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        body_bgr = cv2.addWeighted(overlay, 0.72, body_bgr, 0.28, 0.0)
        cv2.rectangle(body_bgr, (x0, y0), (x1, y1), (235, 235, 235), 1)
        for line_idx, line in enumerate(text_lines):
            _put_text(
                body_bgr,
                line,
                (x0 + pad_x, y0 + pad_y + 15 + line_idx * line_h),
                scale=0.54,
                color=(245, 245, 245),
                thickness=1,
            )

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
        metric = f"gyro MAE {_finite_mae(pred, target):.6f}"
    if title:
        metric = f"{title}   {metric}" if metric else title
    _put_text(header, metric, (12, 28), scale=0.62)

    all_values = pred
    if target is not None:
        all_values = np.concatenate([pred, target], axis=0)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size:
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
    else:
        value_min = -1.0
        value_max = 1.0
    pad = max((value_max - value_min) * 0.1, 1e-4)
    value_min -= pad
    value_max += pad
    pred = np.nan_to_num(pred, nan=0.0, posinf=value_max, neginf=value_min)
    if target is not None:
        target = np.nan_to_num(target, nan=0.0, posinf=value_max, neginf=value_min)

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
    cmf = _to_numpy_cmf(motion_field)
    canvas = _draw_cmf_overlay(
        image,
        cmf,
        grid_step=grid_step,
        trajectory_scale=trajectory_scale,
    )

    header = np.full((44, canvas.shape[1], 3), 26, dtype=np.uint8)
    cmf_h, cmf_w, channels = cmf.shape
    label = f"CMF / paper V   shape={cmf_h}x{cmf_w}x{channels}"
    if title:
        label = f"{title}   {label}"
    _put_text(header, label, (12, 28), scale=0.58)
    return np.concatenate([header, canvas], axis=0)


def make_cmf_comparison(
    image,
    pred_motion_field,
    target_motion_field,
    title=None,
    max_panel_height=360,
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

    pred = _to_numpy_cmf(pred_motion_field)
    target = _to_numpy_cmf(target_motion_field)
    pred, target = _align_cmf_pair(pred, target)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    epe = _cmf_epe(diff)
    mean_epe = float(np.mean(epe))

    pred_panel = _draw_cmf_overlay(
        image,
        pred,
        grid_step=grid_step,
        trajectory_scale=trajectory_scale,
        line_color=(60, 230, 255),
    )
    target_panel = _draw_cmf_overlay(
        image,
        target,
        grid_step=grid_step,
        trajectory_scale=trajectory_scale,
        line_color=(90, 255, 120),
    )
    error_panel = _draw_cmf_error_heatmap(image, epe)

    h = min(panel.shape[0] for panel in (pred_panel, target_panel, error_panel))
    panel_w = min(panel.shape[1] for panel in (pred_panel, target_panel, error_panel))
    panels = [
        cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_AREA)
        for panel in (pred_panel, target_panel, error_panel)
    ]
    body = np.concatenate(panels, axis=1)

    labels = ["Pred CMF", "GT CMF", "Error heatmap"]
    for idx, label in enumerate(labels):
        x0 = idx * panel_w
        cv2.rectangle(body, (x0, 0), (x0 + 142, 28), (0, 0, 0), -1)
        _put_text(body, label, (x0 + 8, 20), scale=0.52)

    header = np.full((44, body.shape[1], 3), 26, dtype=np.uint8)
    metric = f"CMF MAE {mae:.6f} | RMSE {rmse:.6f} | EPE {mean_epe:.6f}"
    if title:
        metric = f"{title}   {metric}"
    _put_text(header, metric, (12, 28), scale=0.58)
    return np.concatenate([header, body], axis=0), {
        "cmf_mae": mae,
        "cmf_rmse": rmse,
        "cmf_epe": mean_epe,
    }


def _to_numpy_sequence(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    return value


def _finite_mae(pred, target):
    diff = np.abs(pred - target)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _draw_cmf_overlay(
    image_rgb,
    cmf,
    grid_step=48,
    trajectory_scale=1.0,
    line_color=(60, 230, 255),
):
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
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
                cv2.line(canvas, p0, p1, line_color, 1, cv2.LINE_AA)
            cv2.circle(canvas, points[0], 2, (40, 120, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, points[-1], 2, (80, 255, 120), -1, lineType=cv2.LINE_AA)
    return canvas


def _align_cmf_pair(pred, target):
    h = min(pred.shape[0], target.shape[0])
    w = min(pred.shape[1], target.shape[1])
    c = min(pred.shape[2], target.shape[2])
    c -= c % 2
    if c <= 0:
        raise ValueError(f"CMF channels must include at least one vector pair: {pred.shape}, {target.shape}")
    return pred[:h, :w, :c], target[:h, :w, :c]


def _cmf_epe(diff):
    vectors = diff.reshape(diff.shape[0], diff.shape[1], -1, 2)
    return np.sqrt(np.sum(vectors * vectors, axis=-1)).mean(axis=-1)


def _draw_cmf_error_heatmap(image_rgb, epe):
    heat = cv2.resize(epe, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    scale = np.percentile(heat, 99.0)
    scale = max(float(scale), 1e-8)
    heat_norm = np.clip(heat / scale, 0.0, 1.0)
    heat_uint8 = np.round(heat_norm * 255.0).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.45, heat_color, 0.55, 0.0)
    _put_text(overlay, f"p99 EPE {scale:.4g}", (8, overlay.shape[0] - 12), scale=0.48)
    return overlay


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
    if not np.isfinite(ratio):
        ratio = 0.5
    ratio = float(np.clip(ratio, 0.0, 1.0))
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
