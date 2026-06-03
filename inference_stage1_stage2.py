import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from utils import (
    build_stage1_stage2_loader,
    load_config,
    load_stage1_stage2_models,
    normalize_config,
    resolve_device,
    run_stage1_stage2_batch,
)
from utils.utils_eval import (
    MetricAverager,
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_visualization import (
    make_stage1_v_visualization,
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference: B -> V, V + B -> S.")
    parser.add_argument("--stage1-config", default="config/stage1_v.yaml")
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--stage2-config", default="config/stage2_deblur.yaml")
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--default-dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--non-strict-stage1", action="store_true")
    parser.add_argument("--non-strict-stage2", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    stage1_cfg = load_config(args.stage1_config)
    stage2_cfg = normalize_config(load_config(args.stage2_config))
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.output_root, "stage1_stage2_inference")
    visual_dir = run_dir / "visuals"
    stage1_visual_dir = visual_dir / "stage1_v"
    stage2_visual_dir = visual_dir / "stage2"
    output_dir = run_dir / "outputs"

    split = args.split or stage2_cfg.get("validation", {}).get("split") or "val"
    _, loader = build_stage1_stage2_loader(
        stage1_cfg,
        stage2_cfg,
        split=split,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        device=device,
        load_target_v=False,
        default_dt=args.default_dt,
    )
    stage1_model, stage2_model, load_report = load_stage1_stage2_models(
        stage1_cfg,
        stage2_cfg,
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        device=device,
        strict_stage1=not args.non_strict_stage1,
        strict_stage2=not args.non_strict_stage2,
    )

    summary = MetricAverager(["psnr", "ssim"])
    rows = []
    saved = 0
    total_batches = len(loader)
    if args.limit is not None:
        total_batches = min(total_batches, math.ceil(int(args.limit) / max(1, int(args.batch_size))))

    image_cfg = stage1_cfg.get("image", {})
    for batch in tqdm(loader, total=total_batches, desc="stage1->stage2 inference"):
        sharp = batch["gt"].to(device, non_blocking=True).float()
        result = run_stage1_stage2_batch(
            stage1_model,
            stage2_model,
            batch,
            stage2_cfg,
            device,
            default_dt=args.default_dt,
        )
        pred = result["pred"]
        pred_v = result["pred_v"].detach().cpu()
        psnr_values = sample_psnr(pred, sharp).detach().cpu()
        ssim_values = sample_ssim(pred, sharp).detach().cpu()

        batch_size = pred.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            if args.limit is not None and saved >= int(args.limit):
                break

            metrics = {
                "psnr": float(psnr_values[idx]),
                "ssim": float(ssim_values[idx]),
            }
            summary.update(metrics)
            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            stage1_visual = make_stage1_v_visualization(
                batch["stage1_image"][idx],
                pred_v[idx],
                title=f"B -> V | {types[idx]} / {stems[idx]}",
                mean=image_cfg.get("mean"),
                std=image_cfg.get("std"),
            )
            stage2_visual = make_stage2_comparison(
                batch["lq"][idx],
                pred[idx].detach().cpu(),
                batch["gt"][idx],
                psnr=metrics["psnr"],
                ssim=metrics["ssim"],
                title=f"V + B -> S | {types[idx]} / {stems[idx]}",
            )
            output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
            stage1_visual_path = stage1_visual_dir / f"{name}.png"
            stage2_visual_path = stage2_visual_dir / f"{name}.png"
            output_path = output_dir / f"{name}_deblur.png"
            write_image(stage1_visual_path, stage1_visual)
            write_image(stage2_visual_path, stage2_visual)
            write_image(output_path, output_rgb[:, :, ::-1].copy())

            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
                "psnr": f"{metrics['psnr']:.6f}",
                "ssim": f"{metrics['ssim']:.8f}",
                "stage1_visual_path": str(stage1_visual_path),
                "stage2_visual_path": str(stage2_visual_path),
                "output_path": str(output_path),
            }
            for v_idx, vector in enumerate(pred_v[idx].numpy()):
                row[f"pred_v{v_idx}_x"] = f"{vector[0]:.8f}"
                row[f"pred_v{v_idx}_y"] = f"{vector[1]:.8f}"
                row[f"pred_v{v_idx}_z"] = f"{vector[2]:.8f}"
            rows.append(row)
            saved += 1

        if args.limit is not None and saved >= int(args.limit):
            break

    fieldnames = list(rows[0].keys()) if rows else [
        "index",
        "type",
        "stem",
        "psnr",
        "ssim",
        "stage1_visual_path",
        "stage2_visual_path",
        "output_path",
    ]
    save_csv(run_dir / "predictions.csv", rows, fieldnames)
    save_json(
        run_dir / "summary.json",
        {
            "overall": summary.as_dict(),
            "stage1_config": str(Path(args.stage1_config)),
            "stage1_checkpoint": args.stage1_checkpoint,
            "stage2_config": str(Path(args.stage2_config)),
            "stage2_checkpoint": args.stage2_checkpoint,
            "split": split,
            "saved": saved,
            "load_report": load_report,
        },
    )
    print(f"saved: {run_dir}")
    print(summary.as_dict())


if __name__ == "__main__":
    main()
