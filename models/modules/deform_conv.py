import torch
import torchvision
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels_blur,
        in_channels_motion,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        in_channels = in_channels_blur + in_channels_motion
        kh, kw = kernel_size

        self.offset_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kh * kw,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.modulator_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=kh * kw,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels_blur,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

    def forward(self, feat_blur, feat_motion):
        guide = torch.cat([feat_blur, feat_motion], dim=1)

        offset = self.offset_conv(guide)
        mask = 2.0 * torch.sigmoid(self.modulator_conv(guide))

        out = torchvision.ops.deform_conv2d(
            input=feat_blur,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=mask,
            stride=self.stride,
            dilation=self.dilation,
        )

        return out