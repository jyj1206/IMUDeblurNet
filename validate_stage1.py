import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.stage1_gyro_dataset import build_stage1_dataset
from models.stage1_gyro_estimation_model import build_stage1_model
from utils import apply_dataset_overrides, load_eval_config
from utils.utils_eval import (
    GroupedMetricAverager,
    MetricAverager,
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    load_model_weights,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_visualization import make_stage1_gyro_visualization, write_image


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 gyro validation by motion type.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loss(name):
    name = str(name).lower()
    if name == "mse":
        return torch.nn.MSELoss(reduction="none")
    if name in ("l1", "mae"):
        return torch.nn.L1Loss(reduction="none")
    if name in ("smooth_l1", "huber"):
        return torch.nn.SmoothL1Loss(reduction="none")
    raise ValueError(f"Unknown stage1 loss: {name}")


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
        batch_size=int(args.batch_size or val_cfg.get("batch_size", cfg["dataset"].get("batch_size", 8))),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else val_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_stage1_model(cfg).to(device).eval()
    load_report = load_model_weights(model, args.checkpoint, device=device, strict=not args.non_strict)
    criterion = build_loss(cfg.get("train", {}).get("loss", "smooth_l1"))
    image_cfg = cfg.get("image", {})

    overall = MetricAverager(["loss", "mae", "rmse"])
    by_type = GroupedMetricAverager(["loss", "mae", "rmse"])
    sample_rows = []
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
        pred_gyro = model(image)["gyro"]
        per_item_loss = criterion(pred_gyro, target_gyro).flatten(1).mean(dim=1)
        per_item_mae = (pred_gyro - target_gyro).abs().flatten(1).mean(dim=1)
        per_item_rmse = torch.sqrt(((pred_gyro - target_gyro) ** 2).flatten(1).mean(dim=1))

        batch_size = image.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            metrics = {
                "loss": float(per_item_loss[idx].detach().cpu()),
                "mae": float(per_item_mae[idx].detach().cpu()),
                "rmse": float(per_item_rmse[idx].detach().cpu()),
            }
            overall.update(metrics)
            by_type.update(types[idx], metrics)
            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
                **{key: f"{value:.8f}" for key, value in metrics.items()},
            }
            sample_rows.append(row)

            if args.save_limit is None or saved < int(args.save_limit):
                name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
                visual = make_stage1_gyro_visualization(
                    batch["image"][idx],
                    pred_gyro[idx].detach().cpu(),
                    target_gyro=target_gyro[idx].detach().cpu(),
                    title=f"{types[idx]} / {stems[idx]}",
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
    }
    save_json(run_dir / "metrics.json", metrics)
    save_csv(run_dir / "samples.csv", sample_rows, ["index", "type", "stem", "loss", "mae", "rmse"])
    print(f"saved: {run_dir}")
    print(metrics["overall"])
    print(metrics["by_type"])


if __name__ == "__main__":
    main()
