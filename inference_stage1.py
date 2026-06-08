import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.image_dataset_common import load_image
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


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage1 gyro inference with gyro visualization."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--input",
        default=None,
        help="Optional image file or directory for direct image inference.",
    )
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


def _as_hw(image_size):
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    return int(image_size[0]), int(image_size[1])


def _resize_image(image, image_size):
    hw = _as_hw(image_size)
    if hw is None:
        return image
    return F.interpolate(
        image.unsqueeze(0), size=hw, mode="bilinear", align_corners=False
    ).squeeze(0)


def _normalize_image(image, mean, std):
    if mean is None or std is None:
        return image
    mean_t = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean_t) / std_t


def _image_paths(input_path, limit=None):
    input_path = Path(input_path)
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTS
        )
    else:
        raise FileNotFoundError(f"Missing input image or directory: {input_path}")
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"No image files found: {input_path}")
    return paths


def _scaled_focal_length(config):
    camera = config.get("camera", {}) or {}
    focal_length = camera.get("focal_length")
    if focal_length is not None:
        return float(focal_length)
    image_size = _as_hw(config.get("image", {}).get("size", (224, 320)))
    native_size = camera.get("native_size", [1080, 1920])
    height, width = (
        image_size
        if image_size is not None
        else (int(native_size[0]), int(native_size[1]))
    )
    native_h, native_w = float(native_size[0]), float(native_size[1])
    fx = float(camera.get("fx", 923.7181693)) * (float(width) / native_w)
    fy = float(camera.get("fy", 924.51235192)) * (float(height) / native_h)
    return 0.5 * (fx + fy)


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

    model = build_stage1_model(cfg).to(device).eval()
    load_report = load_model_weights(
        model, args.checkpoint, device=device, strict=not args.non_strict
    )
    image_cfg = cfg.get("image", {})

    if args.input:
        rows = []
        paths = _image_paths(args.input, limit=args.limit)
        for index, image_path in enumerate(tqdm(paths, desc="stage1 image inference")):
            raw_image = load_image(image_path)
            image = _normalize_image(
                _resize_image(raw_image, image_cfg.get("size", (224, 320))),
                image_cfg.get("mean"),
                image_cfg.get("std"),
            )
            focal_length = torch.tensor(
                [_scaled_focal_length(cfg)], device=device, dtype=torch.float32
            )
            pred_gyro = (
                model(
                    image.unsqueeze(0).to(device=device, dtype=torch.float32),
                    focal_length=focal_length,
                    return_aux=False,
                )["gyro"]
                .detach()
                .cpu()[0]
            )
            name = safe_name(f"{index:06d}", image_path.stem)
            visual_path = visual_dir / f"{name}.png"
            write_image(
                visual_path,
                make_stage1_gyro_visualization(
                    image,
                    pred_gyro,
                    title=image_path.name,
                    mean=image_cfg.get("mean"),
                    std=image_cfg.get("std"),
                ),
            )
            row = {
                "index": index,
                "input_path": str(image_path),
                "visual_path": str(visual_path),
            }
            for gyro_idx, vector in enumerate(pred_gyro.numpy()):
                row[f"pred_gyro{gyro_idx}_x"] = f"{vector[0]:.8f}"
                row[f"pred_gyro{gyro_idx}_y"] = f"{vector[1]:.8f}"
                row[f"pred_gyro{gyro_idx}_z"] = f"{vector[2]:.8f}"
            rows.append(row)

        fieldnames = (
            list(rows[0].keys()) if rows else ["index", "input_path", "visual_path"]
        )
        save_csv(run_dir / "predictions.csv", rows, fieldnames)
        save_json(
            run_dir / "summary.json",
            {
                "config": str(Path(args.config)) if args.config else None,
                "config_source": config_source,
                "checkpoint": args.checkpoint,
                "input": args.input,
                "saved": len(rows),
                "load_report": load_report,
            },
        )
        print(f"saved: {run_dir}")
        return

    split = (
        args.split
        or cfg.get("validation", {}).get("split")
        or cfg["dataset"].get("split", "val")
    )
    dataset = build_stage1_dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )

    rows = []
    saved = 0
    total_batches = len(loader)
    if args.limit is not None:
        total_batches = min(
            total_batches, math.ceil(int(args.limit) / max(1, int(args.batch_size)))
        )
    for batch in tqdm(loader, total=total_batches, desc="stage1 inference"):
        image = batch["image"].to(device, non_blocking=True).float()
        target_gyro = batch.get("gyro")
        focal_length = batch.get("focal_length")
        if focal_length is not None:
            focal_length = focal_length.to(device, non_blocking=True).float()
        pred_gyro = (
            model(image, focal_length=focal_length, return_aux=False)["gyro"]
            .detach()
            .cpu()
        )
        batch_size = image.shape[0]

        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            if args.limit is not None and saved >= int(args.limit):
                break
            target = target_gyro[idx] if target_gyro is not None else None
            mae = (
                float((pred_gyro[idx] - target).abs().mean())
                if target is not None
                else None
            )
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

    fieldnames = (
        list(rows[0].keys())
        if rows
        else ["index", "type", "stem", "visual_path", "mae"]
    )
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
