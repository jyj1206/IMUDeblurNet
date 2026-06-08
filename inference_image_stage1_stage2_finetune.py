import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.image_dataset_common import load_image
from models.stage1_stage2_finetune_model import build_stage1_stage2_finetune_model
from utils import load_config
from utils.utils_eval import create_run_dir, safe_name, save_csv, save_json
from utils.utils_eval_config import upgrade_stage1_config_names
from utils.utils_stage_pipeline import camera_matrix_from_config, resolve_device
from utils.utils_torch_load import torch_load_checkpoint
from utils.utils_visualization import (
    make_cmf_visualization,
    make_stage2_comparison,
    tensor_to_rgb_uint8,
    write_image,
)


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fine-tuned Stage1->CMF->Stage2 inference on image files."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
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
    return upgrade_stage1_config_names(cfg)


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


def _timestamp_window(num_vectors, default_dt, device):
    return torch.arange(int(num_vectors), device=device, dtype=torch.float32).unsqueeze(
        0
    ) * float(default_dt)


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    cfg = _config_from_args(args, checkpoint)
    stage1_cfg = cfg["stage1_resolved_config"]
    stage2_cfg = cfg["stage2_resolved_config"]
    camera_matrix = camera_matrix_from_config(cfg)

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

    run_dir = (
        Path(args.output_dir)
        if args.output_dir
        else create_run_dir("runs", "image_inference")
    )
    output_dir = run_dir / "outputs"
    visual_dir = run_dir / "visuals"
    image_cfg = stage1_cfg.get("image", {})
    target_cfg = stage1_cfg.get("target", {})
    num_vectors = target_cfg.get("num_vectors", 7)
    default_dt = cfg.get("time", {}).get("default_dt", 1.0 / 240.0)
    paths = _image_paths(args.input, limit=args.limit)

    rows = []
    for index, image_path in enumerate(tqdm(paths, desc="image inference")):
        blur = load_image(image_path)
        stage1_image = _normalize_image(
            _resize_image(blur, image_cfg.get("size", (224, 320))),
            image_cfg.get("mean"),
            image_cfg.get("std"),
        )
        tensor_blur = blur.unsqueeze(0).to(device=device, dtype=torch.float32)
        tensor_stage1 = stage1_image.unsqueeze(0).to(device=device, dtype=torch.float32)
        timestamp = _timestamp_window(num_vectors, default_dt, device)
        crop_origin_yx = torch.zeros((1, 2), device=device, dtype=torch.float32)

        outputs = model(
            tensor_stage1, tensor_blur, timestamp, crop_origin_yx=crop_origin_yx
        )
        pred = outputs["pred"][0].detach().cpu()
        name = safe_name(f"{index:06d}", image_path.stem)
        output_path = output_dir / f"{name}_deblur.png"
        output_rgb = tensor_to_rgb_uint8(pred)
        write_image(output_path, output_rgb[:, :, ::-1].copy())

        row = {
            "index": index,
            "input_path": str(image_path),
            "output_path": str(output_path),
        }
        if args.save_visuals:
            cmf_path = visual_dir / "cmf" / f"{name}.png"
            stage2_path = visual_dir / "stage2" / f"{name}.png"
            write_image(
                cmf_path,
                make_cmf_visualization(
                    blur,
                    outputs["cmf"][0].detach().cpu(),
                    title=f"gyro -> CMF | {image_path.name}",
                ),
            )
            write_image(
                stage2_path,
                make_stage2_comparison(
                    blur, pred, title=f"Fine-tuned inference | {image_path.name}"
                ),
            )
            row["cmf_visual_path"] = str(cmf_path)
            row["stage2_visual_path"] = str(stage2_path)
        rows.append(row)

    save_csv(
        run_dir / "samples.csv",
        rows,
        ["index", "input_path", "output_path", "cmf_visual_path", "stage2_visual_path"],
    )
    save_json(
        run_dir / "summary.json",
        {
            "checkpoint": args.checkpoint,
            "config": str(Path(args.config)) if args.config else None,
            "input": args.input,
            "count": len(rows),
        },
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
