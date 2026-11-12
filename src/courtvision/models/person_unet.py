"""SEKME person network: a centre-heatmap detector (CenterNet-style) on the shared U-Net.

Input: one frame resized to 512x288, BGR in [0, 1]. Output, per pixel: a person-centre logit and the
box size as log(width), log(height) in network pixels. Persons are decoded as blobs of the centre
heatmap; their size is read at the blob's centre. Trained in this project (scripts/train_players.py)
by distillation: the labels are the boxes a teacher detector (Ultralytics YOLO26-s) finds in broadcast
frames, so this network learns to reproduce them; it does not see human box labels.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from courtvision.models.unet import UNet, wbce_loss

ARCH = "sekme-person-unet"
IN_W, IN_H = 512, 288
CENTRE_RADIUS = 2.0  # px at 512x288
SIZE_WEIGHT = 0.5


class PersonUNet(UNet):
    def __init__(self, widths: tuple[int, ...] = (24, 48, 96, 192, 192)) -> None:
        super().__init__(3, 3, widths, prior=0.01)
        with torch.no_grad():  # size channels start at a typical player box (log px), not at the prior
            self.head.bias[1] = math.log(12.0)
            self.head.bias[2] = math.log(30.0)


def person_targets(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """boxes (n, 4) x1 y1 x2 y2 in network pixels -> centre map (H, W), log-size map (2, H, W), mask (H, W)."""
    yy, xx = np.mgrid[0:IN_H, 0:IN_W]
    centre = np.zeros((IN_H, IN_W), np.float32)
    size = np.zeros((2, IN_H, IN_W), np.float32)
    # smaller boxes last so that overlapping centres keep the smaller person's size
    for x1, y1, x2, y2 in sorted(boxes.tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= CENTRE_RADIUS ** 2
        centre[disk] = 1.0
        size[0][disk] = math.log(max(x2 - x1, 1.0))
        size[1][disk] = math.log(max(y2 - y1, 1.0))
    return centre, size, centre.copy()


def person_loss(out: torch.Tensor, centre: torch.Tensor, size: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    heat = wbce_loss(out[:, 0], centre)
    reg = (F.l1_loss(out[:, 1:], size, reduction="none") * mask[:, None]).sum() / mask.sum().clamp(min=1.0)
    return heat + SIZE_WEIGHT * reg


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def decode(out: np.ndarray, threshold: float = 0.3, nms_iou: float = 0.5) -> list[tuple[np.ndarray, float]]:
    """(3, H, W) network output (centre logit, log w, log h) -> [(box x1 y1 x2 y2, score)] in network pixels.

    A large person can light up more than one centre blob; boxes overlapping a higher-scoring one by more
    than ``nms_iou`` are dropped (non-maximum suppression).
    """
    heat = 1.0 / (1.0 + np.exp(-out[0]))
    if heat.max() <= threshold:
        return []
    n, labels = cv2.connectedComponents((heat > threshold).astype(np.uint8))
    dets = []
    for m in range(1, n):
        ys, xs = np.nonzero(labels == m)
        w = heat[ys, xs]
        cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
        iy, ix = ys[int(np.argmax(w))], xs[int(np.argmax(w))]
        bw, bh = math.exp(float(out[1, iy, ix])), math.exp(float(out[2, iy, ix]))
        dets.append((np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), float(w.max())))
    kept: list[tuple[np.ndarray, float]] = []
    for box, score in sorted(dets, key=lambda d: -d[1]):
        if all(_overlap(box, k[0]) <= nms_iou for k in kept):
            kept.append((box, score))
    return kept
