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


def make_stage1_v_visualization(
    image,
    pred_v,
    target_v=None,
    title=None,
    mean=None,
    std=None,
    max_image_height=520,
):
    image = tensor_to_rgb_uint8(image, mean=mean, std=std)
    if image.shape[0] > max_image_height:
        image = _resize_to_height(image, max_image_height)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred = _to_numpy_v(pred_v)
    target = _to_numpy_v(target_v) if target_v is not None else None
    panel_h = 220
    panel = np.full((panel_h, image_bgr.shape[1], 3), 34, dtype=np.uint8)

    header = np.full((44, image_bgr.shape[1], 3), 26, dtype=np.uint8)
    metric = ""
    if target is not None:
        metric = f"V MAE {np.abs(pred - target).mean():.6f}"
    if title:
        metric = f"{title}   {metric}" if metric else title
    _put_text(header, metric, (12, 28), scale=0.62)

    all_xy = pred[:, :2]
    if target is not None:
        all_xy = np.concatenate([all_xy, target[:, :2]], axis=0)
    max_norm = max(float(np.linalg.norm(all_xy, axis=1).max()), 1e-6)
    arrow_scale = 64.0 / max_norm

    count = pred.shape[0]
    xs = np.linspace(52, image_bgr.shape[1] - 52, count)
    y0 = 92
    for idx, x in enumerate(xs):
        base = (int(round(x)), y0)
        cv2.circle(panel, base, 3, (210, 210, 210), -1)
        if target is not None:
            _draw_arrow(panel, base, target[idx, :2], arrow_scale, (230, 190, 80))
        _draw_arrow(panel, base, pred[idx, :2], arrow_scale, (80, 230, 120))
        z_text = f"{idx}: z {pred[idx, 2]:+.3g}"
        _put_text(panel, z_text, (max(4, base[0] - 35), 160), scale=0.38, color=(220, 220, 220))

    _put_text(panel, "pred", (12, 30), scale=0.5, color=(80, 230, 120))
    if target is not None:
        _put_text(panel, "target", (72, 30), scale=0.5, color=(230, 190, 80))

    return np.concatenate([header, image_bgr, panel], axis=0)


def _to_numpy_v(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    return value


def _draw_arrow(canvas, base, vector_xy, scale, color):
    dx = float(vector_xy[0]) * scale
    dy = float(vector_xy[1]) * scale
    end = (int(round(base[0] + dx)), int(round(base[1] - dy)))
    cv2.arrowedLine(canvas, base, end, color, 2, tipLength=0.28, line_type=cv2.LINE_AA)
