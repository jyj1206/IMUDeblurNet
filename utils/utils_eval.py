import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from .utils_torch_load import torch_load_checkpoint


class MetricAverager:
    def __init__(self, metric_names):
        self.metric_names = list(metric_names)
        self.total = {name: 0.0 for name in self.metric_names}
        self.count = 0

    def update(self, metrics, n=1):
        n = int(n)
        for name in self.metric_names:
            self.total[name] += float(metrics[name]) * n
        self.count += n

    def as_dict(self):
        if self.count <= 0:
            return {**{name: 0.0 for name in self.metric_names}, "count": 0}
        return {
            **{name: self.total[name] / self.count for name in self.metric_names},
            "count": self.count,
        }


class GroupedMetricAverager:
    def __init__(self, metric_names):
        self.metric_names = list(metric_names)
        self.groups = {}

    def update(self, group, metrics, n=1):
        group = normalize_motion_type(group)
        if group not in self.groups:
            self.groups[group] = MetricAverager(self.metric_names)
        self.groups[group].update(metrics, n=n)

    def as_dict(self):
        return {group: avg.as_dict() for group, avg in sorted(self.groups.items())}


def normalize_motion_type(value):
    text = str(value or "unknown").strip().lower()
    if not text:
        return "unknown"
    if "run" in text:
        return "run"
    if "walk" in text:
        return "walk"
    if "turn" in text:
        return "turn"
    return text


def create_run_dir(root="runs", prefix="eval"):
    root = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{prefix}_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_model_weights(model, checkpoint_path, device="cpu", strict=True):
    if not checkpoint_path:
        return {"path": None, "loaded": False, "missing": 0, "unexpected": 0}

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model")
            or checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
        )
    else:
        state = checkpoint
    if state is None:
        raise KeyError(f"checkpoint does not contain model weights: {checkpoint_path}")

    clean_state = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    missing, unexpected = model.load_state_dict(clean_state, strict=strict)
    return {
        "path": str(checkpoint_path),
        "loaded": True,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def batch_meta_list(batch, key, batch_size, default="unknown"):
    meta = batch.get("meta") or {}
    values = meta.get(key, default)
    if isinstance(values, torch.Tensor):
        return [str(v.item()) for v in values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)] * batch_size


def batch_meta_int_list(batch, key, batch_size, default=0):
    meta = batch.get("meta") or {}
    values = meta.get(key, default)
    if isinstance(values, torch.Tensor):
        return [int(v.item()) for v in values]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return [int(values)] * batch_size


def safe_name(*parts):
    text = "_".join(str(part) for part in parts if str(part))
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"
