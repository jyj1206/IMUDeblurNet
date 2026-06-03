import torch
import torch.nn as nn

from .layer_norm import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()

        dw_channels = channels * dw_expand

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=dw_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.conv2 = nn.Conv2d(
            in_channels=dw_channels,
            out_channels=dw_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channels,
            bias=True,
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channels // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channels // 2,
                out_channels=dw_channels // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        self.sg = SimpleGate()

        ffn_channels = channels * ffn_expand

        self.conv4 = nn.Conv2d(
            in_channels=channels,
            out_channels=ffn_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.conv5 = nn.Conv2d(
            in_channels=ffn_channels // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma