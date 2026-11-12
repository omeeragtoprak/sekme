"""The U-Net shared by SEKME's own heatmap networks (ball, court keypoints) and their loss.

Encoder levels of two 3x3 conv + batch norm + ReLU with 2x2 max pooling, a decoder with bilinear
up-sampling and skip connections, a 1x1 head. The head bias starts at a small prior probability
(RetinaNet) because the targets are a few pixels in a large map.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int, widths: tuple[int, ...], prior: float = 0.01) -> None:
        super().__init__()
        self.enc = nn.ModuleList()
        for w in widths:
            self.enc.append(_block(cin, w))
            cin = w
        self.dec = nn.ModuleList(_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1)))
        self.head = nn.Conv2d(widths[0], cout, 1)
        nn.init.constant_(self.head.bias, -math.log((1 - prior) / prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, block in enumerate(self.enc):
            x = block(x)
            if k < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def disk_targets(points: list[tuple[float, float] | None], w: int, h: int, radius: float) -> np.ndarray:
    """(len(points), h, w) binary maps: a disk at each point (network pixels), empty for None."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((len(points), h, w), np.float32)
    for k, p in enumerate(points):
        if p is not None:
            out[k] = ((xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= radius ** 2).astype(np.float32)
    return out


def wbce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted BCE of TrackNetV2, normalised by the number of target pixels.

    Each term is down-weighted by how confident the network already is. Targets cover a few dozen of
    ~150 000 pixels; averaged over all pixels their signal vanishes and the network settles on a flat
    map (first ball run: peak 0.12 everywhere, argmax 168 px off), so the sum is divided by the target
    pixel count instead, as in CenterNet-style keypoint losses.
    """
    p = torch.sigmoid(logits)
    pos = (1 - p) ** 2 * target * F.logsigmoid(logits)
    neg = p ** 2 * (1 - target) * F.logsigmoid(-logits)
    return -(pos + neg).sum() / target.sum().clamp(min=1.0)
