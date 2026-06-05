import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import build_stage1_stage2_loader
from utils import apply_dataset_overrides, camera_matrix_from_config, load_eval_config
from utils.utils_eval import (
    GroupedMetricAverager,
    MetricAverager,
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_iaai_stage_pipeline import (
    load_stage1_iaai_stage2_models,
    resolve_device,
    run_stage1_iaai_stage2_batch,
)
from utils.utils_iqa import Stage2IqaMetrics, normalize_iqa_metric_names
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_visualization import (
    make_cmf_comparison,
    make_cmf_visualization,
    make_stage1_gyro_visualization,
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)
from validate_stage1_stage2 import (
    finite_issue_label,
    format_metric,
    load_motion_field,
    sample_stage2_loss,
    tensor_finite_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end validation with Stage1 IAAI auxiliary gyro model.")
    parser.add_argument("--stage1-config", default=None)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--stage2-config", default=None)
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--motion-downsample", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--default-dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--camera-fx", type=float, default=None)
    parser.add_argument("--camera-fy", type=float, default=None)
    parser.add_argument("--camera-cx", type=float, default=None)
    parser.add_argument("--camera-cy", type=float, default=None)
    parser.add_argument("--load-target-gyro", action="store_true")
    parser.add_argument("--non-strict-stage1", action="store_true")
    parser.add_argument("--non-strict-stage2", action="store_true")
    parser.add_argument("--extra-metrics", nargs="*", default=[])
    parser.add_argument("--realblur-metrics", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    stage1_cfg, stage1_config_source = load_eval_config(
        args.stage1_config,
        args.stage1_checkpoint,
        device=device,
        normalize=False,
    )
    stage2_cfg, stage2_config_source = load_eval_config(
        args.stage2_config,
        args.stage2_checkpoint,
        device=device,
        normalize=True,
    )
    stage2_cfg = apply_dataset_overrides(stage2_cfg, args, include_motion=True)
    camera_cfg = {
        "camera": {
            **(stage1_cfg.get("camera") or {}),
            **(stage2_cfg.get("camera") or {}),
        }
    }
    camera_matrix = camera_matrix_from_config(
        camera_cfg,
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy,
    )
    run_dir = create_run_dir(args.output_root, "stage1_iaai_stage2_validation")
    visual_dir = run_dir / "visuals"
    gyro_visual_dir = visual_dir / "gyro"
    cmf_visual_dir = visual_dir / "cmf"
    cmf_compare_visual_dir = visual_dir / "cmf_compare"
    stage2_visual_dir = visual_dir / "stage2"
    output_dir = run_dir / "outputs"

    split = args.split or stage2_cfg.get("validation", {}).get("split") or "val"
    _, loader = build_stage1_stage2_loader(
        stage1_cfg,
        stage2_cfg,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        load_target_gyro=args.load_target_gyro,
        default_dt=args.default_dt,
    )
    stage1_model, stage2_model, load_report = load_stage1_iaai_stage2_models(
        stage1_cfg,
        stage2_cfg,
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        device=device,
        strict_stage1=not args.non_strict_stage1,
        strict_stage2=not args.non_strict_stage2,
    )
    extra_metric_names = normalize_iqa_metric_names(
        args.extra_metrics,
        realblur_preset=args.realblur_metrics,
    )
    iqa_metrics = Stage2IqaMetrics(extra_metric_names, device) if extra_metric_names else None
    metric_names = ["loss", "psnr", "ssim"]
    if args.load_target_gyro:
        metric_names.extend(["gyro_mae", "gyro_rmse"])
    metric_names.extend(extra_metric_names)
    overall = MetricAverager(metric_names)
    by_type = GroupedMetricAverager(metric_names)
    rows = []
    finite_issue_counts = {
        "pred_gyro": 0,
        "target_gyro": 0,
        "pred_cmf": 0,
        "pred_image": 0,
    }
    saved = 0
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = stage2_cfg.get("validation", {}).get("max_batches")
    total = len(loader)
    if max_batches is not None:
        total = min(total, int(max_batches))

    progress = tqdm(loader, total=total, desc="stage1 IAAI -> stage2 validation")
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        sharp = batch["gt"].to(device, non_blocking=True).float()
        result = run_stage1_iaai_stage2_batch(
            stage1_model,
            stage2_model,
            batch,
            stage2_cfg,
            device,
            default_dt=args.default_dt,
            camera_matrix=camera_matrix,
            return_aux=False,
        )
        pred = result["pred"]
        pred_raw = result["pred_raw"]
        pred_gyro = result["pred_gyro"].detach().cpu()
        target_gyro = batch["gyro"] if args.load_target_gyro else None
        cmf = result["cmf"].detach().cpu()
        pred_cpu = pred.detach().cpu()

        psnr_values = sample_psnr(pred, sharp).detach().cpu()
        ssim_values = sample_ssim(pred, sharp).detach().cpu()
        extra_values = iqa_metrics(pred, sharp) if iqa_metrics is not None else {}
        loss_values = sample_stage2_loss(
            pred_raw,
            sharp,
            stage2_cfg.get("train", {}).get("loss", "psnr"),
        )
        if args.load_target_gyro:
            gyro_mae_values = (pred_gyro - target_gyro).abs().flatten(1).mean(dim=1)
            gyro_rmse_values = torch.sqrt(((pred_gyro - target_gyro) ** 2).flatten(1).mean(dim=1))

        batch_size = pred.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)
        motion_field_paths = batch_meta_list(batch, "motion_field_path", batch_size, "")
        image_cfg = stage1_cfg.get("image", {})

        for idx in range(batch_size):
            finite_summaries = {
                "pred_gyro": tensor_finite_summary(pred_gyro[idx]),
                "target_gyro": tensor_finite_summary(target_gyro[idx] if args.load_target_gyro else None),
                "pred_cmf": tensor_finite_summary(cmf[idx]),
                "pred_image": tensor_finite_summary(pred_cpu[idx]),
            }
            for key, summary in finite_summaries.items():
                if not summary["finite"]:
                    finite_issue_counts[key] += 1
            finite_issues = finite_issue_label(**finite_summaries)
            metrics = {
                "loss": float(loss_values[idx]),
                "psnr": float(psnr_values[idx]),
                "ssim": float(ssim_values[idx]),
            }
            if args.load_target_gyro:
                metrics["gyro_mae"] = float(gyro_mae_values[idx])
                metrics["gyro_rmse"] = float(gyro_rmse_values[idx])
            for metric_name in extra_metric_names:
                metrics[metric_name] = float(extra_values[metric_name][idx])
            overall.update(metrics)
            by_type.update(types[idx], metrics)

            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
                "finite_issues": finite_issues,
            }
            for source_name, summary in finite_summaries.items():
                row[f"{source_name}_nan"] = summary["nan"]
                row[f"{source_name}_inf"] = summary["inf"]
            for metric_name in metric_names:
                row[metric_name] = format_metric(metric_name, metrics[metric_name])

            if args.save_limit is None or saved < int(args.save_limit):
                gyro_visual = make_stage1_gyro_visualization(
                    batch["stage1_image"][idx],
                    pred_gyro[idx],
                    target_gyro=target_gyro[idx] if args.load_target_gyro else None,
                    title=f"IAAI B -> gyro | {types[idx]} / {stems[idx]}",
                    mean=image_cfg.get("mean"),
                    std=image_cfg.get("std"),
                )
                cmf_visual = make_cmf_visualization(
                    batch["lq"][idx],
                    cmf[idx],
                    title=f"IAAI gyro -> CMF | {types[idx]} / {stems[idx]}",
                )
                cmf_compare_visual = None
                cmf_compare_path = None
                target_cmf = load_motion_field(motion_field_paths[idx]) if args.load_target_gyro else None
                if target_cmf is not None:
                    cmf_compare_visual, cmf_metrics = make_cmf_comparison(
                        batch["lq"][idx],
                        cmf[idx],
                        target_cmf,
                        title=f"IAAI Pred CMF vs GT CMF | {types[idx]} / {stems[idx]}",
                    )
                    for metric_name, metric_value in cmf_metrics.items():
                        row[metric_name] = format_metric(metric_name, metric_value)
                stage2_visual = make_stage2_comparison(
                    batch["lq"][idx],
                    pred[idx].detach().cpu(),
                    batch["gt"][idx],
                    psnr=metrics["psnr"],
                    ssim=metrics["ssim"],
                    title=f"IAAI gyro + B -> S | {types[idx]} / {stems[idx]}",
                )
                output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
                gyro_visual_path = gyro_visual_dir / f"{name}.png"
                cmf_visual_path = cmf_visual_dir / f"{name}.png"
                if cmf_compare_visual is not None:
                    cmf_compare_path = cmf_compare_visual_dir / f"{name}.png"
                stage2_visual_path = stage2_visual_dir / f"{name}.png"
                output_path = output_dir / f"{name}_deblur.png"
                write_image(gyro_visual_path, gyro_visual)
                write_image(cmf_visual_path, cmf_visual)
                if cmf_compare_visual is not None:
                    write_image(cmf_compare_path, cmf_compare_visual)
                write_image(stage2_visual_path, stage2_visual)
                write_image(output_path, output_rgb[:, :, ::-1].copy())
                row.update(
                    {
                        "gyro_visual_path": str(gyro_visual_path),
                        "cmf_visual_path": str(cmf_visual_path),
                        "cmf_compare_visual_path": "" if cmf_compare_path is None else str(cmf_compare_path),
                        "stage2_visual_path": str(stage2_visual_path),
                        "output_path": str(output_path),
                    }
                )
                saved += 1
            rows.append(row)

        progress.set_postfix(overall.as_dict())

    fieldnames = [
        "index",
        "type",
        "stem",
        "finite_issues",
        "pred_gyro_nan",
        "pred_gyro_inf",
        "target_gyro_nan",
        "target_gyro_inf",
        "pred_cmf_nan",
        "pred_cmf_inf",
        "pred_image_nan",
        "pred_image_inf",
        *metric_names,
        "gyro_visual_path",
        "cmf_visual_path",
        "cmf_mae",
        "cmf_rmse",
        "cmf_epe",
        "cmf_compare_visual_path",
        "stage2_visual_path",
        "output_path",
    ]
    save_csv(run_dir / "samples.csv", rows, fieldnames)
    save_json(
        run_dir / "metrics.json",
        {
            "overall": overall.as_dict(),
            "by_type": by_type.as_dict(),
            "stage1_config": str(Path(args.stage1_config)) if args.stage1_config else None,
            "stage1_config_source": stage1_config_source,
            "stage1_checkpoint": args.stage1_checkpoint,
            "stage2_config": str(Path(args.stage2_config)) if args.stage2_config else None,
            "stage2_config_source": stage2_config_source,
            "stage2_checkpoint": args.stage2_checkpoint,
            "dataset_root": stage2_cfg.get("dataset", {}).get("root"),
            "metadata_name": stage2_cfg.get("dataset", {}).get("metadata_name"),
            "split": split,
            "max_batches": max_batches,
            "extra_metrics": extra_metric_names,
            "load_target_gyro": args.load_target_gyro,
            "load_report": load_report,
            "camera_matrix": camera_matrix.tolist(),
            "saved_visuals": saved,
            "finite_issue_counts": finite_issue_counts,
        },
    )
    print(f"saved: {run_dir}")
    print(overall.as_dict())
    print(by_type.as_dict())


if __name__ == "__main__":
    main()
