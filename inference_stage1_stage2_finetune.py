import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.stage1_stage2_finetune_dataset import build_stage1_stage2_finetune_dataset
from models.stage1_stage2_finetune_model import build_stage1_stage2_finetune_model
from utils import load_config
from utils.utils_eval import (
    batch_meta_int_list,
    batch_meta_list,
    create_run_dir,
    safe_name,
    save_csv,
    save_json,
)
from utils.utils_eval_config import upgrade_stage1_config_names
from utils.utils_metrics import sample_psnr, sample_ssim
from utils.utils_stage_pipeline import camera_matrix_from_config, resolve_device
from utils.utils_torch_load import torch_load_checkpoint
from utils.utils_visualization import (
    make_cmf_visualization,
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned Stage1->CMF->Stage2 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def _load_checkpoint(path, device):
    checkpoint = torch_load_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(
            f"Fine-tune checkpoint must contain a combined model state: {path}"
        )
    return checkpoint


def _config_from_args(args, checkpoint):
    cfg = load_config(args.config) if args.config else checkpoint.get("config")
    if cfg is None:
        raise ValueError(
            "Missing config. Pass --config or use a checkpoint saved by train_stage1_stage2_finetune.py."
        )
    cfg = upgrade_stage1_config_names(cfg)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    if args.metadata_name:
        cfg.setdefault("dataset", {})["metadata_name"] = args.metadata_name
    if args.split:
        cfg.setdefault("validation", {})["split"] = args.split
    cfg.setdefault("dataset", {})["allow_missing_gt"] = True
    cfg.setdefault("dataset", {})["load_target_cmf"] = False
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    cfg = _config_from_args(args, checkpoint)
    stage1_cfg = cfg["stage1_resolved_config"]
    stage2_cfg = cfg["stage2_resolved_config"]
    camera_matrix = camera_matrix_from_config(cfg)
    split = args.split or cfg.get("validation", {}).get("split") or "val"

    dataset = build_stage1_stage2_finetune_dataset(
        cfg, stage1_cfg, split=split, is_train=False
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = (
        build_stage1_stage2_finetune_model(
            stage1_config=stage1_cfg,
            stage2_config=stage2_cfg,
            motion_downsample=cfg.get("dataset", {}).get("motion_downsample", 2),
            default_dt=cfg.get("time", {}).get("default_dt", 1.0 / 240.0),
            camera_matrix=camera_matrix,
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    run_dir = create_run_dir(args.output_root, "stage1_stage2_finetune_inference")
    output_dir = run_dir / "outputs"
    visual_dir = run_dir / "visuals"
    rows = []
    total = len(loader) if args.limit is None else min(len(loader), int(args.limit))
    progress = tqdm(loader, total=total, desc="stage1->stage2 finetune inference")
    seen = 0
    for batch in progress:
        if args.limit is not None and seen >= int(args.limit):
            break
        tensor_batch = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        outputs = model(
            tensor_batch["stage1_image"].float(),
            tensor_batch["lq"].float(),
            tensor_batch["timestamp_window"].float(),
            crop_origin_yx=tensor_batch.get("crop_origin_yx"),
        )
        pred = outputs["pred"]
        batch_size = pred.shape[0]
        stems = batch_meta_list(batch, "stem", batch_size, "sample")
        types = batch_meta_list(batch, "type", batch_size, "unknown")
        indices = batch_meta_int_list(batch, "index", batch_size, 0)
        has_gt = "gt" in batch
        psnr_values = (
            sample_psnr(pred, tensor_batch["gt"].float()).detach().cpu()
            if has_gt
            else None
        )
        ssim_values = (
            sample_ssim(pred, tensor_batch["gt"].float()).detach().cpu()
            if has_gt
            else None
        )

        for idx in range(batch_size):
            if args.limit is not None and seen >= int(args.limit):
                break
            name = safe_name(f"{indices[idx]:06d}", types[idx], stems[idx])
            output_rgb = tensor_to_rgb_uint8(pred[idx].detach().cpu())
            output_path = output_dir / f"{name}_deblur.png"
            write_image(output_path, output_rgb[:, :, ::-1].copy())
            row = {
                "index": indices[idx],
                "type": types[idx],
                "stem": stems[idx],
                "output_path": str(output_path),
            }
            if has_gt:
                row["psnr"] = f"{float(psnr_values[idx]):.6f}"
                row["ssim"] = f"{float(ssim_values[idx]):.8f}"
            if args.save_visuals:
                cmf_visual = make_cmf_visualization(
                    batch["lq"][idx],
                    outputs["cmf"][idx].detach().cpu(),
                    title=f"differentiable gyro -> CMF | {types[idx]} / {stems[idx]}",
                )
                stage2_visual = make_stage2_comparison(
                    batch["lq"][idx],
                    pred[idx].detach().cpu(),
                    batch["gt"][idx] if has_gt else None,
                    psnr=float(psnr_values[idx]) if has_gt else None,
                    ssim=float(ssim_values[idx]) if has_gt else None,
                    title=f"E2E fine-tuned | {types[idx]} / {stems[idx]}",
                )
                cmf_path = visual_dir / "cmf" / f"{name}.png"
                stage2_path = visual_dir / "stage2" / f"{name}.png"
                write_image(cmf_path, cmf_visual)
                write_image(stage2_path, stage2_visual)
                row["cmf_visual_path"] = str(cmf_path)
                row["stage2_visual_path"] = str(stage2_path)
            rows.append(row)
            seen += 1

    fieldnames = [
        "index",
        "type",
        "stem",
        "psnr",
        "ssim",
        "cmf_visual_path",
        "stage2_visual_path",
        "output_path",
    ]
    save_csv(run_dir / "samples.csv", rows, fieldnames)
    save_json(
        run_dir / "summary.json",
        {
            "checkpoint": args.checkpoint,
            "config": str(Path(args.config)) if args.config else None,
            "dataset_root": cfg.get("dataset", {}).get("root"),
            "split": split,
            "count": len(rows),
        },
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
