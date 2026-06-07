import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.stage1_stage2_finetune_dataset import build_stage1_stage2_finetune_dataset
from models.stage1_stage2_finetune_model import build_stage1_stage2_finetune_model
from utils import load_config
from utils.utils_eval_config import upgrade_stage1_config_names
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
from utils.utils_iqa import Stage2IqaMetrics, normalize_iqa_metric_names
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_loss import (
    build_stage1_stage2_finetune_loss,
    cmf_epe,
)
from utils.utils_stage_pipeline import camera_matrix_from_config, resolve_device
from utils.utils_torch_load import torch_load_checkpoint
from utils.utils_visualization import (
    make_cmf_comparison,
    make_cmf_visualization,
    make_stage1_gyro_visualization,
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a fine-tuned Stage1->CMF->Stage2 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-missing-gt", action="store_true")
    parser.add_argument("--camera-fx", type=float, default=None)
    parser.add_argument("--camera-fy", type=float, default=None)
    parser.add_argument("--camera-cx", type=float, default=None)
    parser.add_argument("--camera-cy", type=float, default=None)
    parser.add_argument("--load-target-gyro", action="store_true")
    parser.add_argument("--extra-metrics", nargs="*", default=[])
    parser.add_argument("--realblur-metrics", action="store_true")
    return parser.parse_args()


def _load_checkpoint(path, device):
    checkpoint = torch_load_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(f"Fine-tune checkpoint must contain a combined model state: {path}")
    return checkpoint


def _config_from_args(args, checkpoint):
    cfg = load_config(args.config) if args.config else checkpoint.get("config")
    if cfg is None:
        raise ValueError("Missing config. Pass --config or use a checkpoint saved by train_stage1_stage2_finetune.py.")
    cfg = upgrade_stage1_config_names(cfg)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.metadata_name:
        cfg.setdefault("dataset", {})["metadata_name"] = args.metadata_name
    if args.split:
        cfg.setdefault("validation", {})["split"] = args.split
    if args.allow_missing_gt or args.realblur_metrics:
        cfg.setdefault("dataset", {})["allow_missing_gt"] = True
        cfg.setdefault("dataset", {})["load_target_cmf"] = False
    cfg.setdefault("dataset", {})["load_target_gyro"] = bool(args.load_target_gyro)
    return cfg


def _load_motion_field(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".npz":
        with np.load(path) as data:
            motion_field = data["motion_field"].astype(np.float32)
    else:
        motion_field = np.load(path).astype(np.float32)
    if motion_field.shape[0] <= 64 and motion_field.shape[0] < motion_field.shape[-1]:
        return torch.from_numpy(np.array(motion_field, dtype=np.float32, copy=True))
    return torch.from_numpy(np.array(motion_field.transpose(2, 0, 1), dtype=np.float32, copy=True))


def _format(value, digits=8):
    return f"{float(value):.{digits}f}"


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    cfg = _config_from_args(args, checkpoint)
    stage1_cfg = cfg["stage1_resolved_config"]
    stage2_cfg = cfg["stage2_resolved_config"]
    camera_matrix = camera_matrix_from_config(
        cfg,
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy,
    )

    split = args.split or cfg.get("validation", {}).get("split") or "val"
    dataset = build_stage1_stage2_finetune_dataset(
        cfg,
        stage1_cfg,
        split=split,
        is_train=False,
    )
    has_gt = bool(getattr(dataset, "has_gt", True))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_stage1_stage2_finetune_model(
        stage1_config=stage1_cfg,
        stage2_config=stage2_cfg,
        motion_downsample=cfg.get("dataset", {}).get("motion_downsample", 2),
        default_dt=cfg.get("time", {}).get("default_dt", 1.0 / 240.0),
        camera_matrix=camera_matrix,
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    criterion = build_stage1_stage2_finetune_loss(cfg).to(device)

    extra_metric_names = normalize_iqa_metric_names(
        args.extra_metrics,
        realblur_preset=args.realblur_metrics,
        has_target=has_gt,
    )
    iqa_metrics = Stage2IqaMetrics(extra_metric_names, device) if extra_metric_names else None
    load_target_gyro = bool(cfg.get("dataset", {}).get("load_target_gyro", False))
    load_target_cmf = bool(cfg.get("dataset", {}).get("load_target_cmf", False))
    metric_names = ["loss", "image_loss", "gyro_loss", "cmf_loss", "psnr", "ssim"] if has_gt else []
    if load_target_gyro:
        metric_names.extend(["gyro_mae", "gyro_rmse"])
    if load_target_cmf:
        metric_names.append("cmf_epe")
    metric_names.extend(extra_metric_names)
    overall = MetricAverager(metric_names)
    by_type = GroupedMetricAverager(metric_names)
    rows = []
    run_dir = create_run_dir(args.output_root, "stage1_stage2_finetune_validation")
    visual_dir = run_dir / "visuals"
    output_dir = run_dir / "outputs"
    saved = 0

    max_batches = args.max_batches
    total = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    progress = tqdm(loader, total=total, desc="stage1->stage2 finetune validation")
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        tensor_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(
            tensor_batch["stage1_image"].float(),
            tensor_batch["lq"].float(),
            tensor_batch["timestamp_window"].float(),
            crop_origin_yx=tensor_batch.get("crop_origin_yx"),
        )
        pred = outputs["pred"]
        sharp = tensor_batch["gt"].float() if has_gt else None
        target_gyro = tensor_batch["gyro"].float() if load_target_gyro else None
        target_cmf = tensor_batch.get("target_cmf")

        loss_values = criterion(outputs, tensor_batch) if has_gt else None
        psnr_values = sample_psnr(pred, sharp).detach().cpu() if has_gt else None
        ssim_values = sample_ssim(pred, sharp).detach().cpu() if has_gt else None
        extra_values = iqa_metrics(pred, sharp) if iqa_metrics is not None else {}
        if load_target_gyro:
            gyro_mae_values = (outputs["pred_gyro"] - target_gyro).abs().flatten(1).mean(dim=1).detach().cpu()
            gyro_rmse_values = torch.sqrt(((outputs["pred_gyro"] - target_gyro) ** 2).flatten(1).mean(dim=1)).detach().cpu()
        else:
            gyro_mae_values = None
            gyro_rmse_values = None
        cmf_epe_value = cmf_epe(outputs["cmf"], target_cmf.float()).detach().cpu() if target_cmf is not None else None

        batch_size = pred.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)
        motion_paths = batch_meta_list(batch, "motion_field_path", batch_size, "")
        image_cfg = stage1_cfg.get("image", {})

        batch_metrics = {
        }
        if load_target_gyro:
            batch_metrics["gyro_mae"] = float(gyro_mae_values.mean())
            batch_metrics["gyro_rmse"] = float(gyro_rmse_values.mean())
        if target_cmf is not None:
            batch_metrics["cmf_epe"] = float(cmf_epe_value)
        if has_gt:
            batch_metrics.update(
                {
                    "loss": float(loss_values["loss"].detach().cpu()),
                    "image_loss": float(loss_values["image_loss"].detach().cpu()),
                    "gyro_loss": float(loss_values["gyro_loss"].detach().cpu()),
                    "cmf_loss": float(loss_values["cmf_loss"].detach().cpu()),
                    "psnr": float(psnr_values.mean()),
                    "ssim": float(ssim_values.mean()),
                }
            )
        for metric_name in extra_metric_names:
            batch_metrics[metric_name] = float(extra_values[metric_name].mean())
        progress.set_postfix(
            psnr=batch_metrics.get("psnr", 0.0),
            gyro=batch_metrics.get("gyro_mae", 0.0),
            cmf=batch_metrics.get("cmf_epe", 0.0),
        )

        for idx in range(batch_size):
            metrics = dict(batch_metrics)
            if has_gt:
                metrics["psnr"] = float(psnr_values[idx])
                metrics["ssim"] = float(ssim_values[idx])
            if load_target_gyro:
                metrics["gyro_mae"] = float(gyro_mae_values[idx])
                metrics["gyro_rmse"] = float(gyro_rmse_values[idx])
            for metric_name in extra_metric_names:
                metrics[metric_name] = float(extra_values[metric_name][idx])
            overall.update(metrics)
            by_type.update(types[idx], metrics)
            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            row = {"index": indices[idx], "type": types[idx], "stem": stems[idx]}
            for metric_name in metric_names:
                row[metric_name] = _format(metrics[metric_name], digits=6 if metric_name == "psnr" else 8)

            if args.save_limit is None or saved < int(args.save_limit):
                target_cmf_cpu = _load_motion_field(motion_paths[idx])
                gyro_visual = make_stage1_gyro_visualization(
                    batch["stage1_image"][idx],
                    outputs["pred_gyro"][idx].detach().cpu(),
                    target_gyro=batch["gyro"][idx] if load_target_gyro else None,
                    title=f"B -> gyro | {types[idx]} / {stems[idx]}",
                    mean=image_cfg.get("mean"),
                    std=image_cfg.get("std"),
                )
                cmf_visual = make_cmf_visualization(
                    batch["lq"][idx],
                    outputs["cmf"][idx].detach().cpu(),
                    title=f"differentiable gyro -> CMF | {types[idx]} / {stems[idx]}",
                )
                cmf_compare_path = ""
                if target_cmf_cpu is not None:
                    cmf_compare, cmf_metrics = make_cmf_comparison(
                        batch["lq"][idx],
                        outputs["cmf"][idx].detach().cpu(),
                        target_cmf_cpu,
                        title=f"Pred CMF vs GT CMF | {types[idx]} / {stems[idx]}",
                    )
                    cmf_compare_path = visual_dir / "cmf_compare" / f"{name}.png"
                    write_image(cmf_compare_path, cmf_compare)
                    for key, value in cmf_metrics.items():
                        row[key] = _format(value)
                stage2_visual = make_stage2_comparison(
                    batch["lq"][idx],
                    pred[idx].detach().cpu(),
                    batch["gt"][idx] if has_gt else None,
                    psnr=metrics.get("psnr"),
                    ssim=metrics.get("ssim"),
                    title=f"E2E fine-tuned | {types[idx]} / {stems[idx]}",
                )
                output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
                gyro_path = visual_dir / "gyro" / f"{name}.png"
                cmf_path = visual_dir / "cmf" / f"{name}.png"
                stage2_path = visual_dir / "stage2" / f"{name}.png"
                output_path = output_dir / f"{name}_deblur.png"
                write_image(gyro_path, gyro_visual)
                write_image(cmf_path, cmf_visual)
                write_image(stage2_path, stage2_visual)
                write_image(output_path, output_rgb[:, :, ::-1].copy())
                row.update(
                    {
                        "gyro_visual_path": str(gyro_path),
                        "cmf_visual_path": str(cmf_path),
                        "cmf_compare_visual_path": str(cmf_compare_path),
                        "stage2_visual_path": str(stage2_path),
                        "output_path": str(output_path),
                    }
                )
                saved += 1
            rows.append(row)

    fieldnames = [
        "index",
        "type",
        "stem",
        *metric_names,
        "cmf_mae",
        "cmf_rmse",
        "gyro_visual_path",
        "cmf_visual_path",
        "cmf_compare_visual_path",
        "stage2_visual_path",
        "output_path",
    ]
    metrics = {
        "overall": overall.as_dict(),
        "by_type": by_type.as_dict(),
        "checkpoint": args.checkpoint,
        "config": str(Path(args.config)) if args.config else None,
        "dataset_root": cfg.get("dataset", {}).get("root"),
        "split": split,
        "saved_visuals": saved,
        "extra_metrics": extra_metric_names,
        "has_gt": has_gt,
        "load_target_gyro": load_target_gyro,
        "load_target_cmf": load_target_cmf,
    }
    save_json(run_dir / "metrics.json", metrics)
    save_csv(run_dir / "samples.csv", rows, fieldnames)
    print(f"saved: {run_dir}")
    print(metrics["overall"])
    print(metrics["by_type"])


if __name__ == "__main__":
    main()
