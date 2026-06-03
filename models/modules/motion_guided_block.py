import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialAttention
from .deform_conv import DeformableConv2d
from .nafblock import NAFBlock


class MotionRefinementBlock(nn.Module):
    def __init__(self, motion_channels, blur_channels):
        super().__init__()

        self.conv_ca_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=motion_channels + blur_channels,
                out_channels=motion_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.conv_motion = nn.Conv2d(
            in_channels=motion_channels,
            out_channels=motion_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_down = nn.Conv2d(
            in_channels=motion_channels,
            out_channels=2 * motion_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, feat_blur, feat_motion):
        ca_weight = self.conv_ca_weight(torch.cat([feat_blur, feat_motion], dim=1))
        feat_motion = feat_motion * ca_weight
        feat_motion = F.relu(self.conv_motion(feat_motion), inplace=True)
        feat_motion = self.conv_down(feat_motion)

        return feat_motion


class MotionBlock(nn.Module):
    def __init__(self, blur_channels, motion_channels):
        super().__init__()

        self.conv_deform = DeformableConv2d(
            in_channels_blur=blur_channels,
            in_channels_motion=motion_channels,
            out_channels=blur_channels,
        )

    def forward(self, feat_blur, feat_motion):
        feat_blur = self.conv_deform(feat_blur, feat_motion)
        return feat_blur, feat_motion


class MotionGuidedDeblurringBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.motion_block = MotionBlock(
            blur_channels=channels,
            motion_channels=channels,
        )

        self.spatial_attn = SpatialAttention()
        self.naf_block = NAFBlock(channels)

        self.conv = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, feat_blur, feat_motion):
        feat_deform, feat_motion = self.motion_block(feat_blur, feat_motion)

        feat_attn = self.spatial_attn(feat_deform)
        feat_refined = self.naf_block(feat_attn)

        feat = torch.cat([feat_deform, feat_refined], dim=1)
        feat = self.conv(feat)

        return feat, feat_motion