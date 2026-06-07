import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.stage1_dataset import build_stage1_dataset
from models.stage1_model import build_stage1_model
from utils import apply_dataset_overrides, load_eval_config
from utils.utils_eval import (
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
    parser = argparse.ArgumentParser(description="Stage1 gyro inference with gyro visualization.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    run_dir = create_run_dir(args.output_root, "stage1_inference")
    visual_dir = run_dir / "visuals"

    split = args.split or cfg.get("validation", {}).get("split") or cfg["dataset"].get("split", "val")
    dataset = build_stage1_dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )

    model = build_stage1_model(cfg).to(device).eval()
    load_report = load_model_weights(model, args.checkpoint, device=device, strict=not args.non_strict)
    image_cfg = cfg.get("image", {})

    rows = []
    saved = 0
    total_batches = len(loader)
    if args.limit is not None:
        total_batches = min(total_batches, math.ceil(int(args.limit) / max(1, int(args.batch_size))))
    for batch in tqdm(loader, total=total_batches, desc="stage1 inference"):
        image = batch["image"].to(device, non_blocking=True).float()
        target_gyro = batch.get("gyro")
        focal_length = batch.get("focal_length")
        if focal_length is not None:
            focal_length = focal_length.to(device, non_blocking=True).float()
        pred_gyro = model(image, focal_length=focal_length, return_aux=False)["gyro"].detach().cpu()
        batch_size = image.shape[0]

        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            if args.limit is not None and saved >= int(args.limit):
                break
            target = target_gyro[idx] if target_gyro is not None else None
            mae = float((pred_gyro[idx] - target).abs().mean()) if target is not None else None
            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            visual = make_stage1_gyro_visualization(
                batch["image"][idx],
                pred_gyro[idx],
                target_gyro=target,
                title=f"{types[idx]} / {stems[idx]}",
                mean=image_cfg.get("mean"),
                std=image_cfg.get("std"),
            )
            visual_path = visual_dir / f"{name}.png"
            write_image(visual_path, visual)

            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
                "visual_path": str(visual_path),
                "mae": "" if mae is None else f"{mae:.8f}",
            }
            for gyro_idx, vector in enumerate(pred_gyro[idx].numpy()):
                row[f"pred_gyro{gyro_idx}_x"] = f"{vector[0]:.8f}"
                row[f"pred_gyro{gyro_idx}_y"] = f"{vector[1]:.8f}"
                row[f"pred_gyro{gyro_idx}_z"] = f"{vector[2]:.8f}"
            rows.append(row)
            saved += 1

        if args.limit is not None and saved >= int(args.limit):
            break

    fieldnames = list(rows[0].keys()) if rows else ["index", "type", "stem", "visual_path", "mae"]
    save_csv(run_dir / "predictions.csv", rows, fieldnames)
    save_json(
        run_dir / "summary.json",
        {
            "config": str(Path(args.config)) if args.config else None,
            "config_source": config_source,
            "checkpoint": args.checkpoint,
            "dataset_root": cfg.get("dataset", {}).get("root"),
            "metadata_name": cfg.get("dataset", {}).get("metadata_name"),
            "split": split,
            "saved": saved,
            "load_report": load_report,
        },
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
