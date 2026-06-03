import torch
import torch.nn.functional as F


def batch_psnr(pred, target, eps=1e-8):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return (10.0 * torch.log10(1.0 / (mse + eps))).mean()


def batch_ssim(pred, target, window_size=11, eps=1e-8):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.mean()


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, epoch=0, max_batches=None):
    was_training = model.training
    model.eval()

    total = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        blur = batch["lq"].to(device, non_blocking=True).float()
        sharp = batch["gt"].to(device, non_blocking=True).float()
        motion_field = batch["motion_field"].to(device, non_blocking=True).float()
        pred = model(blur, motion_field, epoch)
        loss = criterion(pred, sharp)
        batch_size = blur.shape[0]

        total["loss"] += float(loss.detach().cpu()) * batch_size
        total["psnr"] += float(batch_psnr(pred, sharp).detach().cpu()) * batch_size
        total["ssim"] += float(batch_ssim(pred, sharp).detach().cpu()) * batch_size
        count += batch_size

    if was_training:
        model.train()

    if count == 0:
        return {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    return {key: value / count for key, value in total.items()}
