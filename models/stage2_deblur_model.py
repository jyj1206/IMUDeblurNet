import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class ImageOnlyDeblurNet(nn.Module):
    def __init__(self, width=32, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, blur, motion_field=None, epoch=0):
        feat = self.body(self.head(blur))
        return blur + self.tail(feat)


class MotionFieldDeblurNet(nn.Module):
    def __init__(self, motion_channels=12, width=32, num_blocks=4):
        super().__init__()
        self.img_head = nn.Conv2d(3, width, 3, padding=1)
        self.img_enc = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)

        self.motion_head = nn.Sequential(
            nn.Conv2d(motion_channels, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(width * 2),
        )
        self.fuse = nn.Conv2d(width * 4, width * 2, 1)

        self.mid = nn.Sequential(*[ResBlock(width * 2) for _ in range(num_blocks)])
        self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.dec = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, blur, motion_field, epoch=0):
        if motion_field is None:
            raise ValueError("MotionFieldDeblurNet needs motion_field input.")

        _, _, original_h, original_w = blur.shape
        pad_h = (2 - original_h % 2) % 2
        pad_w = (2 - original_w % 2) % 2
        if pad_h or pad_w:
            blur = F.pad(blur, (0, pad_w, 0, pad_h))

        image_feat = self.img_enc(self.img_head(blur))
        image_down = self.down1(image_feat)

        if motion_field.shape[-2:] != image_down.shape[-2:]:
            raise ValueError(
                "motion_field shape must match the image feature resolution. "
                f"got motion={tuple(motion_field.shape[-2:])}, expected={tuple(image_down.shape[-2:])}"
            )
        motion_feat = self.motion_head(motion_field)

        fused = self.fuse(F.relu(torch.cat((image_down, motion_feat), dim=1)))
        fused = self.mid(fused)
        decoded = self.up1(fused) + image_feat
        decoded = self.dec(decoded)
        output = blur + self.tail(decoded)
        return output[:, :, :original_h, :original_w]


def build_model(config):
    model_cfg = config["model"]
    name = model_cfg.get("name", "motion_field_deblur")
    width = int(model_cfg.get("width", 32))
    num_blocks = int(model_cfg.get("num_blocks", 4))

    if name == "image_only":
        return ImageOnlyDeblurNet(width=width, num_blocks=num_blocks)

    if name == "motion_field_deblur":
        return MotionFieldDeblurNet(
            motion_channels=int(model_cfg.get("motion_channels", 12)),
            width=width,
            num_blocks=num_blocks,
        )

    raise ValueError(f"Unknown model.name: {name}")
