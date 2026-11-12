"""Court keypoint heatmap network (TrackNet-style VGG encoder/decoder).

Independent implementation whose layer names match the public
TennisCourtDetector checkpoint (14 keypoints + 1 centre channel, 640x360
BGR input in [0, 1]). That checkpoint ships without a licence, so it is
treated as research-only; see docs/LICENSES.md.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


_SPEC: list[tuple] = [
    ("conv1", 3, 64), ("conv2", 64, 64), ("pool",),
    ("conv3", 64, 128), ("conv4", 128, 128), ("pool",),
    ("conv5", 128, 256), ("conv6", 256, 256), ("conv7", 256, 256), ("pool",),
    ("conv8", 256, 512), ("conv9", 512, 512), ("conv10", 512, 512), ("up",),
    ("conv11", 512, 256), ("conv12", 256, 256), ("conv13", 256, 256), ("up",),
    ("conv14", 256, 128), ("conv15", 128, 128), ("up",),
    ("conv16", 128, 64), ("conv17", 64, 64), ("conv18", 64, None),
]


class CourtKeypointNet(nn.Module):
    INPUT_W = 640
    INPUT_H = 360

    def __init__(self, out_channels: int = 15) -> None:
        super().__init__()
        self._order: list[str | nn.Module] = []
        for item in _SPEC:
            if item[0] == "pool":
                self._order.append(nn.MaxPool2d(2, 2))
            elif item[0] == "up":
                self._order.append(nn.Upsample(scale_factor=2))
            else:
                name, cin, cout = item
                setattr(self, name, _ConvBlock(cin, cout if cout is not None else out_channels))
                self._order.append(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step in self._order:
            x = getattr(self, step)(x) if isinstance(step, str) else step(x)
        return x
