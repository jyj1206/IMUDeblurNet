import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.stage1_dataset import build_stage1_dataset
from models.stage1_model import build_stage1_model
from utils import Stage1AuxLoss, apply_dataset_overrides, load_eval_config
from utils.utils_eval import (
    GroupedMetricAverager,
    MetricAverager,
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    load_model_weights,
    normalize_motion_type,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_visualization import make_stage1_gyro_visualization, write_image


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stage1 gyro model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tensor_stats(values):
    if not values:
        return {"count": 0}
    tensor = torch.cat([value.reshape(-1).float() for value in values], dim=0)
    finite = torch.isfinite(tensor)
    tensor = tensor[finite]
    if tensor.numel() == 0:
        return {"count": 0}
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
    }


def _axis_alignment(pred_values, target_values):
    if not pred_values or not target_values:
        return {"count": 0}
    pred = torch.cat([value.reshape(-1, 3).float() for value in pred_values], dim=0)
    target = torch.cat([value.reshape(-1, 3).float() for value in target_values], dim=0)
    finite = torch.isfinite(pred).all(dim=1) & torch.isfinite(target).all(dim=1)
    pred = pred[finite]
    target = target[finite]
    if pred.shape[0] < 2:
        return {"count": int(pred.shape[0])}

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    pred_std = pred_centered.std(dim=0, unbiased=False).clamp_min(1e-12)
    target_std = target_centered.std(dim=0, unbiased=False).clamp_min(1e-12)
    corr = (pred_centered[:, :, None] * target_centered[:, None, :]).mean(dim=0)
    corr = corr / (pred_std[:, None] * target_std[None, :])

    axes = ["x", "y", "z"]
    best = []
    for pred_axis in range(3):
        target_axis = int(corr[pred_axis].abs().argmax().item())
        value = float(corr[pred_axis, target_axis].item())
        best.append(
            {
                "pred_axis": axes[pred_axis],
                "target_axis": axes[target_axis],
                "corr": value,
                "sign": 1 if value >= 0 else -1,
            }
        )

    return {
        "count": int(pred.shape[0]),
        "corr_pred_rows_target_cols": [
            [float(corr[row, col].item()) for col in range(3)] for row in range(3)
        ],
        "best_match_per_pred_axis": best,
    }


def _diagnostics(records):
    groups = {}
    scenes = {}
    for record in records:
        group = normalize_motion_type(record["type"])
        groups.setdefault(
            group, {"pred": [], "target": [], "target_norm": [], "pred_norm": []}
        )
        groups[group]["pred"].append(record["pred"])
        groups[group]["target"].append(record["target"])
        groups[group]["target_norm"].append(record["target"].norm(dim=-1))
        groups[group]["pred_norm"].append(record["pred"].norm(dim=-1))

        scene = record.get("scene_dir") or "unknown"
        scenes.setdefault(scene, {"type": group, "target_norm": []})
        scenes[scene]["target_norm"].append(record["target"].norm(dim=-1))

    by_type = {}
    for group, values in sorted(groups.items()):
        by_type[group] = {
            "target_norm": _tensor_stats(values["target_norm"]),
            "pred_norm": _tensor_stats(values["pred_norm"]),
            "axis_alignment": _axis_alignment(values["pred"], values["target"]),
        }

    by_scene = {}
    for scene, values in sorted(scenes.items()):
        by_scene[scene] = {
            "type": values["type"],
            "target_norm": _tensor_stats(values["target_norm"]),
        }
    return {"by_type": by_type, "by_scene": by_scene}


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg, config_source = load_eval_config(
        args.config,
        args.checkpoint,
        device=device,
        normalize=False,
    )
    cfg = apply_dataset_overrides(cfg, args)
    run_dir = create_run_dir(args.output_root, "stage1_validation")
    visual_dir = run_dir / "visuals"

    split = args.split or cfg.get("validation", {}).get("split") or "val"
    val_cfg = cfg.get("validation", {})
    dataset = build_stage1_dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(
            args.batch_size
            or val_cfg.get("batch_size", cfg["dataset"].get("batch_size", 8))
        ),
        shuffle=False,
        num_workers=int(
            args.num_workers
            if args.num_workers is not None
            else val_cfg.get("num_workers", 0)
        ),
        pin_memory=device.type == "cuda",
    )

    model = build_stage1_model(cfg).to(device).eval()
    load_report = load_model_weights(
        model, args.checkpoint, device=device, strict=not args.non_strict
    )
    loss_cfg = cfg.get("loss", {})
    criterion = Stage1AuxLoss(
        gyro_loss=loss_cfg.get(
            "gyro_loss", cfg.get("train", {}).get("loss", "smooth_l1")
        ),
        aux_loss=loss_cfg.get("aux_loss", "smooth_l1"),
        aux_weight=loss_cfg.get("aux_weight", 0.05),
        default_dt=cfg.get("time", {}).get("default_dt", 1.0 / 240.0),
        target_norm_weight=loss_cfg.get("target_norm_weight", 0.0),
        target_norm_reference=loss_cfg.get("target_norm_reference", 2.5),
        target_norm_max_weight=loss_cfg.get("target_norm_max_weight", 3.0),
    )
    image_cfg = cfg.get("image", {})

    metric_names = [
        "loss",
        "gyro_loss",
        "aux_loss",
        "mae",
        "gyro_x_mae",
        "gyro_y_mae",
        "gyro_z_mae",
        "rmse",
        "pose_omega_mae",
    ]
    overall = MetricAverager(metric_names)
    by_type = GroupedMetricAverager(metric_names)
    sample_rows = []
    diagnostic_records = []
    saved = 0

    max_batches = args.max_batches
    if max_batches is None:
        max_batches = val_cfg.get("max_batches")
    total = len(loader)
    if max_batches is not None:
        total = min(total, int(max_batches))
    progress = tqdm(loader, total=total, desc="stage1 validation")
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        image = batch["image"].to(device, non_blocking=True).float()
        target_gyro = batch["gyro"].to(device, non_blocking=True).float()
        timestamp_window = (
            batch["timestamp_window"].to(device, non_blocking=True).float()
        )
        focal_length = batch["focal_length"].to(device, non_blocking=True).float()
        outputs = model(image, focal_length=focal_length, return_aux=True)
        _, loss_metrics, omega_gt = criterion(outputs, target_gyro, timestamp_window)
        pred_gyro = outputs["gyro"]
        pose = outputs.get("pose")
        if pose is not None and omega_gt is not None:
            pose_omega_mae = (pose[:, :3] - omega_gt).abs().flatten(1).mean(dim=1)
        else:
            pose_omega_mae = torch.zeros(image.shape[0], device=device)
        per_item_loss = (pred_gyro - target_gyro).abs().flatten(1).mean(dim=1)
        per_item_axis_mae = (pred_gyro - target_gyro).abs().mean(dim=1)
        per_item_rmse = torch.sqrt(
            ((pred_gyro - target_gyro) ** 2).flatten(1).mean(dim=1)
        )

        batch_size = image.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        scenes = batch_meta_list(batch, "scene_dir", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)
        target_norm = target_gyro.norm(dim=-1).mean(dim=1)
        pred_norm = pred_gyro.norm(dim=-1).mean(dim=1)

        for idx in range(batch_size):
            metrics = {
                "loss": float(loss_metrics["loss"].detach().cpu()),
                "gyro_loss": float(loss_metrics["gyro_loss"].detach().cpu()),
                "aux_loss": float(loss_metrics["aux_loss"].detach().cpu()),
                "mae": float(per_item_loss[idx].detach().cpu()),
                "gyro_x_mae": float(per_item_axis_mae[idx, 0].detach().cpu()),
                "gyro_y_mae": float(per_item_axis_mae[idx, 1].detach().cpu()),
                "gyro_z_mae": float(per_item_axis_mae[idx, 2].detach().cpu()),
                "rmse": float(per_item_rmse[idx].detach().cpu()),
                "pose_omega_mae": float(pose_omega_mae[idx].detach().cpu()),
            }
            overall.update(metrics)
            by_type.update(types[idx], metrics)
            row = {
                "index": indices[idx],
                "type": types[idx],
                "scene_dir": scenes[idx],
                "stem": stems[idx],
                "target_gyro_norm": f"{float(target_norm[idx].detach().cpu()):.8f}",
                "pred_gyro_norm": f"{float(pred_norm[idx].detach().cpu()):.8f}",
                **{key: f"{value:.8f}" for key, value in metrics.items()},
            }
            sample_rows.append(row)
            diagnostic_records.append(
                {
                    "type": types[idx],
                    "scene_dir": scenes[idx],
                    "pred": pred_gyro[idx].detach().cpu(),
                    "target": target_gyro[idx].detach().cpu(),
                }
            )

            if args.save_limit is None or saved < int(args.save_limit):
                name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
                visual = make_stage1_gyro_visualization(
                    batch["image"][idx],
                    pred_gyro[idx].detach().cpu(),
                    target_gyro=target_gyro[idx].detach().cpu(),
                    title=f"Stage1 gyro | {types[idx]} / {stems[idx]}",
                    mean=image_cfg.get("mean"),
                    std=image_cfg.get("std"),
                )
                write_image(visual_dir / f"{name}.png", visual)
                saved += 1

        progress.set_postfix(overall.as_dict())

    metrics = {
        "overall": overall.as_dict(),
        "by_type": by_type.as_dict(),
        "config": str(Path(args.config)) if args.config else None,
        "config_source": config_source,
        "checkpoint": args.checkpoint,
        "dataset_root": cfg.get("dataset", {}).get("root"),
        "metadata_name": cfg.get("dataset", {}).get("metadata_name"),
        "split": split,
        "max_batches": max_batches,
        "load_report": load_report,
        "saved_visuals": saved,
        "diagnostics": _diagnostics(diagnostic_records),
    }
    save_json(run_dir / "metrics.json", metrics)
    save_csv(
        run_dir / "samples.csv",
        sample_rows,
        [
            "index",
            "type",
            "scene_dir",
            "stem",
            "target_gyro_norm",
            "pred_gyro_norm",
            *metric_names,
        ],
    )
    print(f"saved: {run_dir}")
    print(metrics["overall"])
    print(metrics["by_type"])


if __name__ == "__main__":
    main()
