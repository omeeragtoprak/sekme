"""SEKME court-keypoint network: a U-Net trained in this project on the TennisCourtDetector labels.

Input: one frame resized to 640x360, BGR in [0, 1] (the contract of perception/court.py, shared with
the TennisCourtDetector network it can replace). Output: 14 logit heatmaps, one per court keypoint in
the TennisCourtDetector order (geometry/court.py: CourtModel.keypoints). Five encoder levels
(32-64-128-256-256) reach a 22x40 bottleneck, wide enough to tell which corner is which.
Training: scripts/train_court.py.
"""

from __future__ import annotations

import numpy as np

from courtvision.models.unet import UNet
from courtvision.models.unet import disk_targets as _disk_targets

ARCH = "sekme-court-unet"
IN_W, IN_H = 640, 360
KEYPOINTS = 14
TARGET_RADIUS = 4.0  # px at 640x360
# Keypoint index after a horizontal flip: left and right swap, the centre-line T points stay.
FLIP = (1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13)


class CourtUNet(UNet):
    def __init__(self, widths: tuple[int, ...] = (32, 64, 128, 256, 256)) -> None:
        super().__init__(3, KEYPOINTS, widths, prior=0.01)


def court_targets(points: list[tuple[float, float] | None]) -> np.ndarray:
    """(14, 360, 640) disk targets; None (out of frame) gives an empty map."""
    return _disk_targets(points, IN_W, IN_H, TARGET_RADIUS)


def flip_points(points: list[tuple[float, float] | None]) -> list[tuple[float, float] | None]:
    """Keypoints of the horizontally mirrored frame, re-ordered so each index keeps its meaning."""
    mirrored = [None if p is None else (IN_W - 1 - p[0], p[1]) for p in points]
    return [mirrored[FLIP[i]] for i in range(KEYPOINTS)]
