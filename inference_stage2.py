import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.image_dataset_common import load_image
from datasets import build_stage2_dataset
from models.stage2_deblur_model import build_model
from utils import (
    apply_dataset_overrides,
    configure_stage2_motion_loading,
    load_eval_config,
    stage2_forward,
)
from utils.utils_eval import (
    MetricAverager,
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    load_model_weights,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_visualization import (
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage2 deblur inference visualization."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--input",
        default=None,
        help="Optional image file or directory for direct image inference.",
    )
    parser.add_argument(
        "--motion-input",
        default=None,
        help="Optional CMF .npy/.npz file or directory for direct image inference.",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--motion-field-root", default=None)
    parser.add_argument("--motion-field-dir", default=None)
    parser.add_argument("--motion-field-ext", default=None)
    parser.add_argument("--motion-downsample", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Run a Stage2 model with use_motion=False without loading CMF files.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _load_motion_field(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            motion_field = data["motion_field"].astype(np.float32)
    else:
        motion_field = np.load(path).astype(np.float32)
    if motion_field.ndim != 3:
        raise ValueError(
            f"motion field must be HWC or CHW, got {motion_field.shape}: {path}"
        )
    if motion_field.shape[0] <= 64 and motion_field.shape[0] < motion_field.shape[-1]:
        return torch.from_numpy(np.ascontiguousarray(motion_field))
    return torch.from_numpy(np.ascontiguousarray(motion_field.transpose(2, 0, 1)))


def _motion_path_for(image_path, motion_input):
    if not motion_input:
        return None
    motion_input = Path(motion_input)
    if motion_input.is_file():
        return motion_input
    if not motion_input.is_dir():
        raise FileNotFoundError(
            f"Missing motion input file or directory: {motion_input}"
        )
    for suffix in (".npy", ".npz"):
        candidate = motion_input / f"{Path(image_path).stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing CMF for {image_path}: {motion_input / (Path(image_path).stem + '.npy')}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg, config_source = load_eval_config(
        args.config,
        args.checkpoint,
        device=device,
        normalize=True,
    )
    cfg = apply_dataset_overrides(cfg, args, include_motion=True)
    use_motion = configure_stage2_motion_loading(cfg, force_image_only=args.image_only)
    run_dir = create_run_dir(args.output_root, "stage2_inference")
    visual_dir = run_dir / "visuals"
    output_dir = run_dir / "outputs"

    model = build_model(cfg).to(device).eval()
    load_report = load_model_weights(
        model, args.checkpoint, device=device, strict=not args.non_strict
    )

    if args.input:
        if use_motion and not args.motion_input:
            raise ValueError(
                "Stage2 motion-guided inference needs --motion-input, or use --image-only with an image-only model."
            )
        rows = []
        paths = _image_paths(args.input, limit=args.limit)
        for index, image_path in enumerate(tqdm(paths, desc="stage2 image inference")):
            blur = (
                load_image(image_path)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            batch = {}
            if use_motion:
                motion_path = _motion_path_for(image_path, args.motion_input)
                batch["motion_field"] = _load_motion_field(motion_path).unsqueeze(0)
            else:
                motion_path = None
            pred = stage2_forward(
                model, blur, batch, device, use_motion=use_motion
            ).clamp(0.0, 1.0)
            output_rgb = tensor_to_rgb_uint8(pred[0].detach().cpu())
            name = safe_name(f"{index:06d}", image_path.stem)
            visual_path = visual_dir / f"{name}.png"
            output_path = output_dir / f"{name}_deblur.png"
            write_image(
                visual_path,
                make_stage2_comparison(
                    blur[0].detach().cpu(),
                    pred[0].detach().cpu(),
                    title=image_path.name,
                ),
            )
            write_image(output_path, output_rgb[:, :, ::-1].copy())
            rows.append(
                {
                    "index": index,
                    "input_path": str(image_path),
                    "motion_path": "" if motion_path is None else str(motion_path),
                    "visual_path": str(visual_path),
                    "output_path": str(output_path),
                }
            )

        save_csv(
            run_dir / "predictions.csv",
            rows,
            ["index", "input_path", "motion_path", "visual_path", "output_path"],
        )
        save_json(
            run_dir / "summary.json",
            {
                "config": str(Path(args.config)) if args.config else None,
                "config_source": config_source,
                "checkpoint": args.checkpoint,
                "input": args.input,
                "motion_input": args.motion_input,
                "saved": len(rows),
                "use_motion": use_motion,
                "load_report": load_report,
            },
        )
        print(f"saved: {run_dir}")
        return

    split = args.split or cfg.get("validation", {}).get("split") or "val"
    dataset = build_stage2_dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )

    summary = MetricAverager(["psnr", "ssim"])
    rows = []
    saved = 0

    total_batches = len(loader)
    if args.limit is not None:
        total_batches = min(
            total_batches, math.ceil(int(args.limit) / max(1, int(args.batch_size)))
        )
    for batch in tqdm(loader, total=total_batches, desc="stage2 inference"):
        blur = batch["lq"].to(device, non_blocking=True).float()
        sharp = batch["gt"].to(device, non_blocking=True).float()
        pred = stage2_forward(model, blur, batch, device, use_motion=use_motion).clamp(
            0.0, 1.0
        )
        psnr_values = sample_psnr(pred, sharp).detach().cpu()
        ssim_values = sample_ssim(pred, sharp).detach().cpu()

        batch_size = blur.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            if args.limit is not None and saved >= int(args.limit):
                break
            metrics = {"psnr": float(psnr_values[idx]), "ssim": float(ssim_values[idx])}
            summary.update(metrics)
            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            visual = make_stage2_comparison(
                blur[idx].detach().cpu(),
                pred[idx].detach().cpu(),
                sharp[idx].detach().cpu(),
                psnr=metrics["psnr"],
                ssim=metrics["ssim"],
                title=f"{types[idx]} / {stems[idx]}",
            )
            output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
            visual_path = visual_dir / f"{name}.png"
            output_path = output_dir / f"{name}_deblur.png"
            write_image(visual_path, visual)
            write_image(output_path, output_rgb[:, :, ::-1].copy())
            rows.append(
                {
                    "index": indices[idx],
                    "type": types[idx],
                    "stem": stems[idx],
                    "psnr": f"{metrics['psnr']:.6f}",
                    "ssim": f"{metrics['ssim']:.8f}",
                    "visual_path": str(visual_path),
                    "output_path": str(output_path),
                }
            )
            saved += 1

        if args.limit is not None and saved >= int(args.limit):
            break

    save_csv(
        run_dir / "predictions.csv",
        rows,
        ["index", "type", "stem", "psnr", "ssim", "visual_path", "output_path"],
    )
    save_json(
        run_dir / "summary.json",
        {
            "overall": summary.as_dict(),
            "config": str(Path(args.config)) if args.config else None,
            "config_source": config_source,
            "checkpoint": args.checkpoint,
            "dataset_root": cfg.get("dataset", {}).get("root"),
            "metadata_name": cfg.get("dataset", {}).get("metadata_name"),
            "split": split,
            "saved": saved,
            "use_motion": use_motion,
            "load_report": load_report,
        },
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
