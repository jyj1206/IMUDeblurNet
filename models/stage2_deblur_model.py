import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.nafblock import NAFBlock
from .modules.motion_guided_block import (
    MotionGuidedDeblurringBlock,
    MotionRefinementBlock,
)


class MotionGuidedDeblurNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        motion_channel=16,
        width=32,
        middle_blk_num=16,
        enc_blk_nums=(2, 2, 2),
        dec_blk_nums=(1, 1, 1),
        use_motion=True,
    ):
        super().__init__()
        self.use_motion = bool(use_motion)

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        if self.use_motion:
            self.intro_motion = nn.Conv2d(
                in_channels=motion_channel,
                out_channels=width * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )

        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        if self.use_motion:
            self.motion_refine_blks = nn.ModuleList()

        self.middle_blks = nn.ModuleList()
        if self.use_motion:
            self.motion_deblur_blks = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = width

        for idx, num_blocks in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num_blocks)])
            )

            if self.use_motion:
                if idx == 0:
                    self.motion_refine_blks.append(nn.Identity())
                else:
                    self.motion_refine_blks.append(
                        MotionRefinementBlock(
                            motion_channels=chan,
                            blur_channels=chan,
                        )
                    )

            self.downs.append(
                nn.Conv2d(
                    in_channels=chan,
                    out_channels=chan * 2,
                    kernel_size=2,
                    stride=2,
                )
            )

            chan = chan * 2

        num_motion_blocks = 4
        blocks_per_group = middle_blk_num // num_motion_blocks

        for _ in range(num_motion_blocks):
            self.middle_blks.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(blocks_per_group)])
            )

            if self.use_motion:
                self.motion_deblur_blks.append(MotionGuidedDeblurringBlock(chan))

        remain_blocks = middle_blk_num % num_motion_blocks
        if remain_blocks > 0:
            self.middle_blks.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(remain_blocks)])
            )

            if self.use_motion:
                self.motion_deblur_blks.append(MotionGuidedDeblurringBlock(chan))

        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=chan,
                        out_channels=chan * 2,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.PixelShuffle(2),
                )
            )

            chan = chan // 2

            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num_blocks)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, blur, motion=None):
        _, _, h, w = blur.shape

        blur = self.check_image_size(blur)
        x = self.intro(blur)

        motion_feat = None
        if self.use_motion:
            if motion is None:
                raise ValueError("motion is required when use_motion=True")
            motion = self.check_motion_size(
                motion,
                target_h=blur.shape[-2] // 2,
                target_w=blur.shape[-1] // 2,
            )
            motion_feat = self.intro_motion(motion)

        encs = []

        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            encs.append(x)

            if self.use_motion and idx != 0:
                motion_feat = self.motion_refine_blks[idx](x, motion_feat)
            x = down(x)

        for idx, middle_blk in enumerate(self.middle_blks):
            x = middle_blk(x)
            if self.use_motion:
                x, motion_feat = self.motion_deblur_blks[idx](x, motion_feat)

        for decoder, up, enc_skip in zip(
            self.decoders,
            self.ups,
            encs[::-1],
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        x = x + blur

        return x[:, :, :h, :w]

    def check_image_size(self, x):
        _, _, h, w = x.size()

        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def check_motion_size(self, motion, target_h, target_w):
        _, _, h, w = motion.size()
        if h > target_h or w > target_w:
            raise ValueError(
                "motion must be half-resolution relative to padded blur. "
                f"got motion={(h, w)}, expected at most={(target_h, target_w)}"
            )

        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h or pad_w:
            motion = F.pad(motion, (0, pad_w, 0, pad_h))
        return motion


def _model_config(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args.get("model", args)
    return vars(args)


def _model_args(model_cfg):
    aliases = {
        "img_channels": "img_channel",
        "image_channels": "img_channel",
        "motion_channels": "motion_channel",
    }
    allowed = {
        "img_channel",
        "motion_channel",
        "width",
        "middle_blk_num",
        "enc_blk_nums",
        "dec_blk_nums",
        "use_motion",
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
        raise ValueError(f"Unknown MotionGuidedDeblurNet args: {unknown}")
    return kwargs


def build_model(args=None):
    model_cfg = _model_config(args)
    name = model_cfg.get("name", "motion_guided_deblur").lower()
    if name not in ("motion_guided_deblur", "motion_guided_deblurnet"):
        raise ValueError(f"Unknown model.name: {name}")
    return MotionGuidedDeblurNet(**_model_args(model_cfg))


if __name__ == "__main__":
    img_channel = 3
    motion_channel = 16
    width = 32

    enc_blks = (2, 2, 2)
    middle_blk_num = 16
    dec_blks = (1, 1, 1)

    net = MotionGuidedDeblurNet(
        img_channel=img_channel,
        motion_channel=motion_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
        use_motion=True,
    ).cuda()

    blur = torch.randn(1, 3, 256, 256).cuda()
    motion = torch.randn(1, motion_channel, 128, 128).cuda()

    out = net(blur, motion)

    print("params:", sum(p.numel() for p in net.parameters()))
    print("out:", out.shape)
