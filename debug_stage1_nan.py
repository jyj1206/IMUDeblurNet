import argparse
from pathlib import Path

import torch

from datasets.stage1_stage2_dataset import Stage1Stage2Dataset
from models.stage1_gyro_estimation_model import build_stage1_model
from utils.utils_eval import load_model_weights
from utils.utils_eval_config import load_eval_config


def tensor_summary(value):
    if isinstance(value, (list, tuple)):
        summaries = [tensor_summary(item) for item in value]
        bad = [item for item in summaries if item and not item["finite"]]
        return bad[0] if bad else (summaries[0] if summaries else None)
    if isinstance(value, dict):
        summaries = [tensor_summary(item) for item in value.values()]
        bad = [item for item in summaries if item and not item["finite"]]
        return bad[0] if bad else (summaries[0] if summaries else None)
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().float()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    return {
        "shape": tuple(tensor.shape),
        "finite": bool(finite.all().item()),
        "nan": int(torch.isnan(tensor).sum().item()),
        "inf": int(torch.isinf(tensor).sum().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else float("nan"),
        "max": float(finite_values.max().item()) if finite_values.numel() else float("nan"),
        "mean_abs": float(finite_values.abs().mean().item()) if finite_values.numel() else float("nan"),
    }


def format_summary(summary):
    if summary is None:
        return "n/a"
    return (
        f"shape={summary['shape']} finite={summary['finite']} "
        f"nan={summary['nan']} inf={summary['inf']} "
        f"min={summary['min']:.6g} max={summary['max']:.6g} "
        f"mean_abs={summary['mean_abs']:.6g}"
    )


def check_state(model):
    issues = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not torch.is_floating_point(tensor):
            continue
        summary = tensor_summary(tensor)
        if summary and not summary["finite"]:
            issues.append((name, summary))
    return issues


def add_hooks(model, traces):
    hook_names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if (
            name == "backbone"
            or name.startswith("backbone.patch_embed")
            or name.startswith("backbone.norm")
            or name.startswith("backbone.block")
            or name.startswith("gyro_head")
        ):
            hook_names.append(name)

    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            traces.append((name, tensor_summary(output)))

        return hook

    for name, module in model.named_modules():
        if name in hook_names:
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def print_first_bad(traces):
    bad = [(name, summary) for name, summary in traces if summary and not summary["finite"]]
    if not bad:
        print("first_bad_layer: none")
        return
    name, summary = bad[0]
    print(f"first_bad_layer: {name} | {format_summary(summary)}")
    print("bad_layers:")
    for name, summary in bad[:20]:
        print(f"  {name}: {format_summary(summary)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("weights/best_stage1.pt"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--metadata-name", default="metadata.csv")
    parser.add_argument("--indices", nargs="+", type=int, default=[361, 362])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    stage1_config, config_source = load_eval_config(
        config_path=args.config,
        checkpoint_path=args.stage1_checkpoint,
    )
    image_cfg = stage1_config.get("image", {})
    target_cfg = stage1_config.get("target", {})

    dataset = Stage1Stage2Dataset(
        dataset_root=args.dataset_root,
        split=args.split,
        metadata_name=args.metadata_name,
        stage1_image_size=image_cfg.get("size", (224, 320)),
        stage1_mean=image_cfg.get("mean"),
        stage1_std=image_cfg.get("std"),
        num_vectors=target_cfg.get("num_vectors", 7),
        vector_start=target_cfg.get("vector_start", 0),
        vector_dim=target_cfg.get("vector_dim", 3),
        load_target_gyro=True,
    )

    model = build_stage1_model(stage1_config).to(device).eval()
    load_report = load_model_weights(model, args.stage1_checkpoint, device=device, strict=True)

    print(f"config_source: {config_source}")
    print(f"load_report: {load_report}")
    state_issues = check_state(model)
    if state_issues:
        print("state_nonfinite:")
        for name, summary in state_issues[:20]:
            print(f"  {name}: {format_summary(summary)}")
    else:
        print("state_nonfinite: none")

    for index in args.indices:
        sample = dataset[index]
        meta = sample["meta"]
        image = sample["stage1_image"].unsqueeze(0).to(device).float()
        print("")
        print(f"sample index={index} type={meta['type']} scene={meta['scene_dir']} stem={meta['stem']}")
        print(f"stage1_input: {format_summary(tensor_summary(image))}")
        preprocessed = model.backbone._preprocess(image)
        print(f"backbone_preprocess: {format_summary(tensor_summary(preprocessed))}")

        traces = []
        handles = add_hooks(model, traces)
        output = model(image)["gyro"]
        for handle in handles:
            handle.remove()
        print(f"pred_gyro: {format_summary(tensor_summary(output))}")
        print(f"target_gyro: {format_summary(tensor_summary(sample['gyro']))}")
        print_first_bad(traces)


if __name__ == "__main__":
    main()
