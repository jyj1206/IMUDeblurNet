from copy import deepcopy
from pathlib import Path

from .utils_config import normalize_config
from .utils_torch_load import torch_load_checkpoint
from .utils_yaml import load_config


_LEGACY_STAGE1_MODEL_NAMES = {
    "stage1_iaai_gyro",
    "stage1_iaai_aux_gyro",
    "iaai_gyro",
}
_LEGACY_BASELINE_STAGE1_MODEL_NAMES = {
    "stage1_gyro_estimation",
    "blur_to_gyro",
    "blur_to_gyro_net",
}


def _replace_legacy_stage1_text(value):
    if isinstance(value, str):
        return value.replace("stage1_iaai", "stage1")
    return value


def upgrade_stage1_config_names(config):
    if not isinstance(config, dict):
        return config

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        name = str(model_cfg.get("name", "")).lower()
        if name in _LEGACY_STAGE1_MODEL_NAMES:
            model_cfg["name"] = "stage1"
        elif name in _LEGACY_BASELINE_STAGE1_MODEL_NAMES:
            model_cfg["name"] = "stage1"
            args_cfg = model_cfg.setdefault("args", {})
            if isinstance(args_cfg, dict):
                args_cfg.setdefault("use_aux_branch", False)

    exp_cfg = config.get("experiment")
    if isinstance(exp_cfg, dict):
        for key in ("name", "run_prefix"):
            if key in exp_cfg:
                exp_cfg[key] = _replace_legacy_stage1_text(exp_cfg[key])

    for value in config.values():
        if isinstance(value, dict):
            upgrade_stage1_config_names(value)
        elif isinstance(value, list):
            for item in value:
                upgrade_stage1_config_names(item)
    return config


_upgrade_stage1_config_names = upgrade_stage1_config_names


def checkpoint_config(checkpoint_path, device="cpu"):
    if not checkpoint_path:
        return None
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get("config")


def load_eval_config(
    config_path=None, checkpoint_path=None, device="cpu", normalize=False
):
    if config_path:
        config = load_config(config_path)
        source = str(Path(config_path))
    else:
        config = checkpoint_config(checkpoint_path, device=device)
        source = f"{checkpoint_path}:config" if config is not None else None

    if config is None:
        raise ValueError(
            "Missing config. Pass --config, or pass a checkpoint that contains a saved config."
        )

    config = deepcopy(config)
    config = upgrade_stage1_config_names(config)
    if normalize:
        config = normalize_config(config)
    return config, source


def apply_dataset_overrides(config, args, include_motion=False):
    config = deepcopy(config)
    dataset_cfg = config.setdefault("dataset", {})
    validation_cfg = config.setdefault("validation", {})

    if getattr(args, "dataset_root", None):
        dataset_cfg["root"] = args.dataset_root
    if getattr(args, "split", None):
        dataset_cfg["split"] = args.split
        validation_cfg["split"] = args.split
    if getattr(args, "metadata_name", None):
        dataset_cfg["metadata_name"] = args.metadata_name
    if getattr(args, "batch_size", None) is not None:
        validation_cfg["batch_size"] = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        validation_cfg["num_workers"] = int(args.num_workers)

    if include_motion:
        if getattr(args, "motion_field_root", None) is not None:
            dataset_cfg["motion_field_root"] = args.motion_field_root
        if getattr(args, "motion_field_dir", None) is not None:
            dataset_cfg["motion_field_dir"] = args.motion_field_dir
        if getattr(args, "motion_field_ext", None) is not None:
            dataset_cfg["motion_field_ext"] = args.motion_field_ext
        if getattr(args, "motion_downsample", None) is not None:
            dataset_cfg["motion_downsample"] = int(args.motion_downsample)

    return config
