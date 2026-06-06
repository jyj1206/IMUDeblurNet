import torch
from torch import nn

from .stage2_deblur_model import build_model as build_stage2_model
from .modules.torch_camera_motion_field import make_camera_motion_field_torch


def build_stage1_for_finetune(config):
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    name = str(model_cfg.get("name", "stage1_gyro_estimation")).lower()
    if name == "stage1_iaai_gyro":
        from .stage1_iaai_gyro_model import build_stage1_iaai_model

        return build_stage1_iaai_model(config)

    from .stage1_gyro_estimation_model import build_stage1_model

    return build_stage1_model(config)


class Stage1Stage2FinetuneNet(nn.Module):
    def __init__(
        self,
        stage1_model,
        stage2_model,
        motion_downsample=2,
        default_dt=1.0 / 240.0,
        camera_matrix=None,
    ):
        super().__init__()
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.motion_downsample = int(motion_downsample)
        self.default_dt = float(default_dt)
        if camera_matrix is None:
            self.register_buffer("camera_matrix", None, persistent=False)
        else:
            self.register_buffer(
                "camera_matrix",
                torch.as_tensor(camera_matrix, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, stage1_image, blur, timestamp_window, crop_origin_yx=None):
        try:
            stage1_out = self.stage1(stage1_image, return_aux=False)
        except TypeError:
            stage1_out = self.stage1(stage1_image)
        pred_gyro = stage1_out["gyro"]
        cmf = make_camera_motion_field_torch(
            gyro_window=pred_gyro,
            timestamp_window=timestamp_window,
            height=blur.shape[-2],
            width=blur.shape[-1],
            downsample=self.motion_downsample,
            default_dt=self.default_dt,
            camera_matrix=self.camera_matrix,
            origin_yx=crop_origin_yx,
        )
        pred_raw = self.stage2(blur, cmf)
        return {
            "pred_gyro": pred_gyro,
            "cmf": cmf,
            "motion_field": cmf,
            "pred_raw": pred_raw,
            "pred": pred_raw.clamp(0.0, 1.0),
            "stage1": stage1_out,
        }


def build_stage1_stage2_finetune_model(
    stage1_config,
    stage2_config,
    motion_downsample=2,
    default_dt=1.0 / 240.0,
    camera_matrix=None,
):
    stage1_model = build_stage1_for_finetune(stage1_config)
    stage2_model = build_stage2_model(stage2_config)
    return Stage1Stage2FinetuneNet(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        motion_downsample=motion_downsample,
        default_dt=default_dt,
        camera_matrix=camera_matrix,
    )
