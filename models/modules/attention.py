import torch
import torch.nn as nn


class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_pool = torch.max(x, dim=1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, dim=1).unsqueeze(1)
        return torch.cat([max_pool, mean_pool], dim=1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = ChannelPool()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=1,
                eps=1e-5,
                momentum=0.01,
                affine=True,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = self.pool(x)
        scale = self.sigmoid(self.conv(pooled))
        return x * scale