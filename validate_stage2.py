import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_stage2_dataset
from models.stage2_deblur_model import build_model
from utils import (
    apply_dataset_overrides,
    build_criterion,
    configure_stage2_motion_loading,
    load_eval_config,
    stage2_forward,
)
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
from utils.utils_iqa import Stage2IqaMetrics, normalize_iqa_metric_names
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_visualization import make_stage2_comparison, tensor_to_rgb_uint8, write_image


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 deblur validation with PSNR/SSIM by type.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--motion-field-root", default=None)
    parser.add_argument("--motion-field-dir", default=None)
    parser.add_argument("--motion-field-ext", default=None)
    parser.add_argument("--motion-downsample", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Allow metadata rows without sharp_path and report only no-reference metrics.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Evaluate a Stage2 model with use_motion=False without loading CMF files.",
    )
    parser.add_argument(
        "--extra-metrics",
        nargs="*",
        default=[],
        help="Optional standalone validation metrics: lpips niqe topiq/topiq_fr topiq_nr.",
    )
    parser.add_argument(
        "--realblur-metrics",
        action="store_true",
        help="Shortcut for RealBlur-style evaluation: LPIPS, NIQE, and TOPIQ-FR.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def format_metric(name, value):
    if name == "psnr":
        return f"{float(value):.6f}"
    return f"{float(value):.8f}"


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
    if args.allow_missing_gt or args.realblur_metrics:
        cfg.setdefault("dataset", {})["allow_missing_gt"] = True
    use_motion = configure_stage2_motion_loading(cfg, force_image_only=args.image_only)
    run_dir = create_run_dir(args.output_root, "stage2_validation")
    visual_dir = run_dir / "visuals"
    output_dir = run_dir / "outputs"

    split = args.split or cfg.get("validation", {}).get("split") or "val"
    val_cfg = cfg.get("validation", {})
    dataset = build_stage2_dataset(cfg, split=split)
    has_gt = bool(getattr(dataset, "has_gt", True))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or val_cfg.get("batch_size", cfg["dataset"].get("batch_size", 1))),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else val_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device).eval()
    load_report = load_model_weights(model, args.checkpoint, device=device, strict=not args.non_strict)
    criterion = build_criterion(cfg.get("train", {}).get("loss", "psnr")).to(device)
    extra_metric_names = normalize_iqa_metric_names(
        args.extra_metrics,
        realblur_preset=args.realblur_metrics,
        has_target=has_gt,
    )
    iqa_metrics = Stage2IqaMetrics(extra_metric_names, device) if extra_metric_names else None
    reference_metric_names = ["loss", "psnr", "ssim"] if has_gt else []
    metric_names = [*reference_metric_names, *extra_metric_names]
    overall = MetricAverager(metric_names)
    by_type = GroupedMetricAverager(metric_names)
    sample_rows = []
    saved = 0

    max_batches = args.max_batches
    if max_batches is None:
        max_batches = val_cfg.get("max_batches")
    total = len(loader)
    if max_batches is not None:
        total = min(total, int(max_batches))
    progress = tqdm(loader, total=total, desc="stage2 validation")
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        blur = batch["lq"].to(device, non_blocking=True).float()
        sharp = batch["gt"].to(device, non_blocking=True).float() if has_gt else None
        pred_raw = stage2_forward(model, blur, batch, device, use_motion=use_motion)
        pred = pred_raw.clamp(0.0, 1.0)
        psnr_values = sample_psnr(pred, sharp).detach().cpu() if has_gt else None
        ssim_values = sample_ssim(pred, sharp).detach().cpu() if has_gt else None
        extra_values = iqa_metrics(pred, sharp) if iqa_metrics is not None else {}
        if has_gt:
            if criterion.__class__.__name__.lower().startswith("psnr"):
                mse = ((pred_raw - sharp) ** 2).flatten(1).mean(dim=1)
                loss_values = (10.0 * torch.log10(mse + 1e-8)).detach().cpu()
            else:
                loss_values = ((pred_raw - sharp).abs().flatten(1).mean(dim=1)).detach().cpu()
        else:
            loss_values = None

        batch_size = blur.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)

        for idx in range(batch_size):
            metrics = {}
            if has_gt:
                metrics.update(
                    {
                        "loss": float(loss_values[idx]),
                        "psnr": float(psnr_values[idx]),
                        "ssim": float(ssim_values[idx]),
                    }
                )
            for metric_name in extra_metric_names:
                metrics[metric_name] = float(extra_values[metric_name][idx])
            overall.update(metrics)
            by_type.update(types[idx], metrics)
            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
            }
            for metric_name in metric_names:
                row[metric_name] = format_metric(metric_name, metrics[metric_name])
            sample_rows.append(row)

            if args.save_limit is None or saved < int(args.save_limit):
                name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
                visual = make_stage2_comparison(
                    blur[idx].detach().cpu(),
                    pred[idx].detach().cpu(),
                    sharp[idx].detach().cpu() if has_gt else None,
                    psnr=metrics.get("psnr"),
                    ssim=metrics.get("ssim"),
                    title=f"{types[idx]} / {stems[idx]}",
                )
                output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
                write_image(visual_dir / f"{name}.png", visual)
                write_image(output_dir / f"{name}_deblur.png", output_rgb[:, :, ::-1].copy())
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
        "extra_metrics": extra_metric_names,
        "has_gt": has_gt,
        "use_motion": use_motion,
        "load_report": load_report,
        "saved_visuals": saved,
    }
    save_json(run_dir / "metrics.json", metrics)
    save_csv(run_dir / "samples.csv", sample_rows, ["index", "type", "stem", *metric_names])
    print(f"saved: {run_dir}")
    print(metrics["overall"])
    print(metrics["by_type"])


if __name__ == "__main__":
    main()
