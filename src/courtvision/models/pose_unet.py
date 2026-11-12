"""SEKME pose network: 17 COCO keypoints of one player crop, on the shared U-Net (top-down).

Input: the player's crop (box widened for an extended arm, as perception/pose.py cuts it) resized to
128x192 (w x h), BGR in [0, 1]. Output: 17 logit heatmaps in the COCO order; a keypoint is the centroid
of its heatmap blob, its confidence the blob's peak. Trained in this project (scripts/train_pose.py) by
distillation from YOLO26-s-pose run on the same crops; it does not see human keypoint labels.
"""

from __future__ import annotations

import numpy as np

from courtvision.models.unet import UNet
from courtvision.models.unet import disk_targets as _disk_targets

ARCH = "sekme-pose-unet"
IN_W, IN_H = 128, 192
KEYPOINTS = 17
TARGET_RADIUS = 2.5
# COCO order: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees,
# 15/16 ankles; a horizontal flip swaps every left/right pair.
FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


class PoseUNet(UNet):
    def __init__(self, widths: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__(3, KEYPOINTS, widths, prior=0.01)


def pose_targets(points: list[tuple[float, float] | None]) -> np.ndarray:
    return _disk_targets(points, IN_W, IN_H, TARGET_RADIUS)


def crop_box(box: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    """The crop around a person box used for pose (room for an extended arm), or None if too small."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - 0.6 * w)), int(max(0, y1 - 0.35 * h))
    X2, Y2 = int(min(frame_w, x2 + 0.6 * w)), int(min(frame_h, y2 + 0.1 * h))
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None
    return X1, Y1, X2, Y2
