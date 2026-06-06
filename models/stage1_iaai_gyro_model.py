from pathlib import Path

import torch
import torch.nn as nn

from utils.utils_torch_load import torch_load_checkpoint

from .modules.iaai_decoders import DepthDecoder, FlowDecoder
from .modules.iaai_pose_solver import DifferentiablePoseSolver
from .modules.stage1_gyro_head import GlobalGyroHead
from .modules.stage1_mscan import MSCAN
from .stage1_gyro_estimation_model import _load_mscan_backbone, _state_dict_from_checkpoint, _strip_prefix


def _load_decoder_weights(model, weights_path):
    if not weights_path:
        return {"path": None, "loaded": 0, "skipped": 0, "missing": 0, "unexpected": 0}

    weights_path = Path(weights_path)
    checkpoint = torch_load_checkpoint(weights_path, map_location="cpu")
    source_state = _state_dict_from_checkpoint(checkpoint)
    target_state = model.state_dict()
    loadable = {}
    skipped = 0
    prefixes = (
        "module.",
        "model.",
    )
    for key, value in source_state.items():
        clean_key = _strip_prefix(key, prefixes)
        if clean_key in target_state and target_state[clean_key].shape == value.shape:
            loadable[clean_key] = value
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {
        "path": str(weights_path),
        "loaded": len(loadable),
        "skipped": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


class Stage1IAAIGyroNet(nn.Module):
    def __init__(
        self,
        head_hidden=512,
        num_vectors=7,
        vector_dim=3,
        dropout=0.0,
        backbone_weights=None,
        decoder_weights=None,
        freeze_backbone=False,
        freeze_decoders=False,
        freeze_flow_decoder=False,
        freeze_depth_decoder=False,
        decoder_hidden=128,
        use_aux_branch=True,
        pose_ridge=1e-4,
        pose_max_points=4096,
        bgr255_input=True,
    ):
        super().__init__()
        self.use_aux_branch = bool(use_aux_branch)
        self.backbone = MSCAN(bgr255_input=bgr255_input)
        self.gyro_head = GlobalGyroHead(
            in_channels=512,
            hidden_channels=head_hidden,
            num_vectors=num_vectors,
            vector_dim=vector_dim,
            dropout=dropout,
        )

        if self.use_aux_branch:
            self.flow_decoder = FlowDecoder(hidden_channels=decoder_hidden)
            self.depth_decoder = DepthDecoder(hidden_channels=decoder_hidden)
            self.pose_solver = DifferentiablePoseSolver(
                ridge=pose_ridge,
                max_points=pose_max_points,
            )
        else:
            self.flow_decoder = None
            self.depth_decoder = None
            self.pose_solver = None

        self.pretrained_report = {
            "backbone": _load_mscan_backbone(self.backbone, backbone_weights),
            "decoders": _load_decoder_weights(self, decoder_weights or backbone_weights)
            if self.use_aux_branch
            else {"path": None, "loaded": 0, "skipped": 0, "missing": 0, "unexpected": 0},
        }

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if self.use_aux_branch:
            if freeze_decoders:
                freeze_flow_decoder = True
                freeze_depth_decoder = True
            modules_to_freeze = []
            if freeze_flow_decoder:
                modules_to_freeze.append(self.flow_decoder)
            if freeze_depth_decoder:
                modules_to_freeze.append(self.depth_decoder)
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, blur, focal_length=None, return_aux=True):
        features = self.backbone({"image": blur})["features"]
        gyro = self.gyro_head(features[-1])
        outputs = {"gyro": gyro}

        if self.use_aux_branch and return_aux:
            flow = self.flow_decoder(features)
            depth = self.depth_decoder(features)
            pose = self.pose_solver(flow, depth, focal_length=focal_length)
            outputs.update(
                {
                    "flow": flow,
                    "depth": depth,
                    "pose": pose,
                }
            )
        return outputs


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
        "weights": "backbone_weights",
        "pretrained": "backbone_weights",
    }
    allowed = {
        "head_hidden",
        "num_vectors",
        "vector_dim",
        "dropout",
        "backbone_weights",
        "decoder_weights",
        "freeze_backbone",
        "freeze_decoders",
        "freeze_flow_decoder",
        "freeze_depth_decoder",
        "decoder_hidden",
        "use_aux_branch",
        "pose_ridge",
        "pose_max_points",
        "bgr255_input",
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
        raise ValueError(f"Unknown Stage1IAAIGyroNet args: {unknown}")
    return kwargs


def build_stage1_iaai_model(config=None):
    model_cfg = _model_config(config)
    name = model_cfg.get("name", "stage1_iaai_gyro").lower()
    if name not in ("stage1_iaai_gyro", "stage1_iaai_aux_gyro", "iaai_gyro"):
        raise ValueError(f"Unknown stage1 IAAI model.name: {name}")
    return Stage1IAAIGyroNet(**_model_args(model_cfg))


if __name__ == "__main__":
    model = Stage1IAAIGyroNet()
    image = torch.randn(2, 3, 360, 480)
    focal = torch.tensor([230.0, 230.0])
    output = model(image, focal_length=focal)
    print("gyro:", output["gyro"].shape)
    print("flow:", output["flow"].shape)
    print("depth:", output["depth"].shape)
    print("pose:", output["pose"].shape)
