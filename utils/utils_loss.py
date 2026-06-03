import math

import torch
from torch import nn


class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 10.0 / math.log(10.0)

    def forward(self, pred, target):
        return self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def build_criterion(name):
    name = name.lower()
    if name == "psnr":
        return PSNRLoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown train.loss: {name}")
