import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRefineBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class IAAIFpnDecoder(nn.Module):
    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        hidden_channels=128,
        out_channels=2,
        final_activation=None,
    ):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, hidden_channels, 1) for channels in in_channels]
        )
        self.refine = nn.ModuleList(
            [ConvRefineBlock(hidden_channels, hidden_channels) for _ in in_channels]
        )
        self.out = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.final_activation = final_activation

    def forward(self, features):
        if len(features) != len(self.lateral):
            raise ValueError(
                f"expected {len(self.lateral)} features, got {len(features)}"
            )

        x = self.lateral[-1](features[-1])
        x = self.refine[-1](x)
        for idx in range(len(features) - 2, -1, -1):
            x = F.interpolate(
                x,
                size=features[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = x + self.lateral[idx](features[idx])
            x = self.refine[idx](x)

        x = self.out(x)
        if self.final_activation == "softplus":
            x = F.softplus(x) + 1e-3
        elif self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class FlowDecoder(IAAIFpnDecoder):
    def __init__(self, in_channels=(64, 128, 320, 512), hidden_channels=128):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            final_activation=None,
        )


class DepthDecoder(IAAIFpnDecoder):
    def __init__(self, in_channels=(64, 128, 320, 512), hidden_channels=128):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            final_activation="softplus",
        )
