import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from models.stage2_deblur_model import build_model
from utils import load_config, normalize_config
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
from utils.utils_visualization import make_stage2_comparison, tensor_to_rgb_uint8, write_image


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 deblur inference visualization.")
    parser.add_argument("--config", default="config/stage2_deblur.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--limit", type=int, default=32)
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
    cfg = normalize_config(load_config(args.config))
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.output_root, "stage2_inference")
    visual_dir = run_dir / "visuals"
    output_dir = run_dir / "outputs"

    split = args.split or cfg.get("validation", {}).get("split") or "val"
    dataset = build_dataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device).eval()
    load_report = load_model_weights(model, args.checkpoint, device=device, strict=not args.non_strict)
    summary = MetricAverager(["psnr", "ssim"])
    rows = []
    saved = 0

    total_batches = len(loader)
    if args.limit is not None:
        total_batches = min(total_batches, math.ceil(int(args.limit) / max(1, int(args.batch_size))))
    for batch in tqdm(loader, total=total_batches, desc="stage2 inference"):
        blur = batch["lq"].to(device, non_blocking=True).float()
        sharp = batch["gt"].to(device, non_blocking=True).float()
        motion = batch["motion_field"].to(device, non_blocking=True).float()
        pred = model(blur, motion).clamp(0.0, 1.0)
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
            "config": str(Path(args.config)),
            "checkpoint": args.checkpoint,
            "split": split,
            "saved": saved,
            "load_report": load_report,
        },
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
