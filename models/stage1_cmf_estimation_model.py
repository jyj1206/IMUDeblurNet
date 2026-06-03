from pathlib import Path

import torch
from torch import nn

from .modules.stage1_cmf_head import GlobalVHead
from .modules.stage1_mscan import MSCAN


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def _strip_prefix(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _load_mscan_backbone(backbone, weights_path):
    if not weights_path:
        return {"path": None, "loaded": 0, "skipped": 0, "missing": 0}

    weights_path = Path(weights_path)
    checkpoint = torch.load(weights_path, map_location="cpu")
    source_state = _state_dict_from_checkpoint(checkpoint)
    target_state = backbone.state_dict()

    loadable = {}
    skipped = 0
    for key, value in source_state.items():
        clean_key = _strip_prefix(
            key,
            (
                "module.backbone.",
                "model.backbone.",
                "backbone.",
                "module.",
            ),
        )
        if clean_key in target_state and target_state[clean_key].shape == value.shape:
            loadable[clean_key] = value
        else:
            skipped += 1

    missing, unexpected = backbone.load_state_dict(loadable, strict=False)
    return {
        "path": str(weights_path),
        "loaded": len(loadable),
        "skipped": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


class Stage1CMFEstimationNet(nn.Module):
    def __init__(
        self,
        head_hidden=512,
        num_vectors=7,
        vector_dim=3,
        dropout=0.0,
        backbone_weights=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = MSCAN()
        self.v_head = GlobalVHead(
            in_channels=512,
            hidden_channels=head_hidden,
            num_vectors=num_vectors,
            vector_dim=vector_dim,
            dropout=dropout,
        )
        self.pretrained_report = _load_mscan_backbone(self.backbone, backbone_weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, blur):
        features = self.backbone({"image": blur})["features"]
        return {"v": self.v_head(features[-1])}


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return vars(value)


def _model_config(config):
    cfg = _as_dict(config)
    return _as_dict(cfg.get("model", cfg))


def _model_args(model_cfg):
    aliases = {
        "num_outputs": "num_vectors",
        "output_dim": "vector_dim",
        "weights": "backbone_weights",
        "pretrained": "backbone_weights",
    }
    allowed = {
        "head_hidden",
        "num_vectors",
        "vector_dim",
        "dropout",
        "backbone_weights",
        "freeze_backbone",
    }

    kwargs = dict(model_cfg.get("args") or {})
    for key in allowed:
        if key in model_cfg and key not in kwargs:
            kwargs[key] = model_cfg[key]
    for source, target in aliases.items():
        if source in model_cfg and target not in kwargs:
            kwargs[target] = model_cfg[source]
        if source in kwargs and target not in kwargs:
            kwargs[target] = kwargs.pop(source)

    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"Unknown Stage1CMFEstimationNet args: {unknown}")
    return kwargs


def build_stage1_model(config=None):
    model_cfg = _model_config(config)
    name = model_cfg.get("name", "stage1_cmf_estimation").lower()
    if name not in ("stage1_cmf_estimation", "blur_to_v", "blur_to_v_net"):
        raise ValueError(f"Unknown stage1 model.name: {name}")
    return Stage1CMFEstimationNet(**_model_args(model_cfg))


if __name__ == "__main__":
    model = Stage1CMFEstimationNet()
    blur = torch.randn(2, 3, 224, 320)
    out = model(blur)
    print(out["v"].shape)
