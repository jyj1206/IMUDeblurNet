from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.utils import _pair


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, height, width


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2)
        return attn * residual


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        x = x.permute(0, 2, 1).view(batch, channels, height, width)
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x.view(batch, channels, tokens).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = _pair(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, height, width


class MSCAN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=None,
        mlp_ratios=None,
        depths=None,
        drop_rate=0.0,
        bgr255_input=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dims = list(embed_dims or [64, 128, 320, 512])
        self.mlp_ratios = list(mlp_ratios or [8, 8, 4, 4])
        self.depths = list(depths or [3, 3, 12, 3])
        self.drop_rate = float(drop_rate)
        self.num_stages = len(self.embed_dims)
        self.bgr255_input = bool(bgr255_input)

        if not (len(self.embed_dims) == len(self.mlp_ratios) == len(self.depths)):
            raise ValueError("embed_dims, mlp_ratios, and depths must have the same length.")

        for idx in range(self.num_stages):
            if idx == 0:
                patch_embed = StemConv(self.in_channels, self.embed_dims[idx])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[idx - 1],
                    embed_dim=self.embed_dims[idx],
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=self.embed_dims[idx],
                        mlp_ratio=self.mlp_ratios[idx],
                        drop=self.drop_rate,
                    )
                    for _ in range(self.depths[idx])
                ]
            )
            norm = nn.LayerNorm(self.embed_dims[idx])

            setattr(self, f"patch_embed{idx + 1}", patch_embed)
            setattr(self, f"block{idx + 1}", block)
            setattr(self, f"norm{idx + 1}", norm)

    def _preprocess(self, image):
        if self.bgr255_input:
            return image[:, [2, 1, 0], :, :] * 255.0
        return image

    def forward(self, image: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if isinstance(image, dict):
            image = image["image"]
        x = self._preprocess(image)
        batch = x.shape[0]

        features: List[torch.Tensor] = []
        for idx in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{idx + 1}")
            block = getattr(self, f"block{idx + 1}")
            norm = getattr(self, f"norm{idx + 1}")

            x, height, width = patch_embed(x)
            for blk in block:
                x = blk(x, height, width)
            x = norm(x)
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)

        return {"features": features}

