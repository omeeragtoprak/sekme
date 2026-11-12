"""SEKME ball network: a compact U-Net trained in this project on the TrackNet tennis labels.

Input: three consecutive frames, each resized to 512x288, RGB, ImageNet-normalised and stacked on the
channel axis (9 x 288 x 512). Output: one logit heatmap per input frame (3 x 288 x 512); the ball is
where sigmoid(logit) > 0.5. This is the same contract as the WASB network it can replace, so the
streaming detector (perception/ball.py) and its post-processing work unchanged.

Architecture: four encoder levels (32-64-128-256 channels, two 3x3 conv + BN + ReLU each, 2x2 max
pooling), a decoder with bilinear up-sampling and skip connections, a 1x1 head. ~1.95 M parameters.
Training: scripts/train_ball.py (loss: weighted BCE of TrackNetV2, binary disk targets).
"""

from __future__ import annotations

import numpy as np

from courtvision.models.unet import UNet, wbce_loss
from courtvision.models.unet import disk_targets as _disk_targets

ARCH = "sekme-ball-unet"
IN_W, IN_H = 512, 288
FRAMES = 3
TARGET_RADIUS = 2.5  # px at 512x288: a tennis ball is ~3-6 px across at this scale
BALL_PRIOR = 0.01

__all__ = ["ARCH", "FRAMES", "IN_H", "IN_W", "BallUNet", "disk_targets", "wbce_loss"]


class BallUNet(UNet):
    """Three stacked frames in (9 channels), one heatmap per frame out."""

    def __init__(self, widths: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__(3 * FRAMES, FRAMES, widths, BALL_PRIOR)


def disk_targets(points: list[tuple[float, float] | None], w: int = IN_W, h: int = IN_H,
                 radius: float = TARGET_RADIUS) -> np.ndarray:
    """Ball targets at the network's input size."""
    return _disk_targets(points, w, h, radius)
