"""Which player is which across end changes, from clothing colour (candidate, docs/ARCHITECTURE.md).

The pipeline knows the near and the far player, not who they are. Names from the broadcast graphic map ends to
players through the number of end changes read off the score, which a misread digit or a highlight edit flips.
Here the two players are told apart by what they wear: in every stretch of play (between camera cuts and long
pauses, where nobody changes ends) one of two people is near and the other far, and their clothing colours decide
which. Names, where the graphic gives them, are then attached to the two people by majority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from courtvision.io.video import open_video

CROP_W, CROP_H = 24, 48  # every player crop is resized to this before its colours are read
OFFSETS = (0, 4, 8)  # frames after a flight's start sampled for both players
STRETCH_GAP_S = 20.0  # a pause between flights this long can hide a change of ends (no camera cut needed)
MIN_MARGIN = 0.15  # Hellinger cost difference below which a stretch's assignment is left undecided
_H_BINS = 12
DESCRIPTOR = "clothing/2:24x48:edge18:sat60:hue12c"  # part of the cache key: change it with the descriptor


def crops_at(video: str | Path, requests: Sequence[tuple[int, np.ndarray]]) -> tuple[list[np.ndarray | None], bool]:
    """A CROP_W x CROP_H crop for each (frame, box) request, decoding the video once, in order (frames that are
    not needed are skipped without being converted), and whether every requested frame was reached (False when
    the video ends or fails to decode early)."""
    out: list[np.ndarray | None] = [None] * len(requests)
    wanted: dict[int, list[int]] = {}
    for k, (frame, box) in enumerate(requests):
        if np.all(np.isfinite(box)):
            wanted.setdefault(int(frame), []).append(k)
    if not wanted:
        return out, True
    cap = open_video(video)
    if not cap.isOpened():
        return out, False
    last = max(wanted)
    i = 0
    try:
        while i <= last and cap.grab():
            if i in wanted:
                ok, img = cap.retrieve()
                if ok:
                    h, w = img.shape[:2]
                    for k in wanted[i]:
                        x1, y1, x2, y2 = requests[k][1]
                        x1, y1 = int(round(max(x1, 0))), int(round(max(y1, 0)))
                        x2, y2 = int(round(min(x2, w))), int(round(min(y2, h)))
                        if x2 - x1 >= 4 and y2 - y1 >= 8:
                            out[k] = cv2.resize(img[y1:y2, x1:x2], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
            i += 1
    finally:
        cap.release()
    return out, i > last


def clothing(crop: np.ndarray) -> np.ndarray | None:
    """Normalised colour histogram of the torso of one player crop: pixels like the crop's left and right edges
    (the court behind the player) left out; saturated pixels binned by hue, the rest as dark / grey / light.
    None when too little of the torso differs from the background."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(float)
    edge = np.concatenate([lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])[::3]
    body = lab[8:36, 5:19].reshape(-1, 3)  # shoulders to hips, the middle columns
    dist = np.min(np.linalg.norm(body[:, None, :] - edge[None, :, :], axis=2), axis=1)
    keep = dist > 18
    if keep.sum() < 20:
        return None
    h, s, v = hsv[8:36, 5:19].reshape(-1, 3)[keep].T
    sat = s > 60
    hist = np.zeros(_H_BINS + 3)
    np.add.at(hist, (h[sat] / 180 * _H_BINS + 0.5).astype(int) % _H_BINS, 1)  # bin 0 centred on red (hue 0/180)
    grey = ~sat
    hist[_H_BINS] = np.sum(grey & (v < 70))
    hist[_H_BINS + 1] = np.sum(grey & (v >= 70) & (v < 170))
    hist[_H_BINS + 2] = np.sum(grey & (v >= 170))
    return hist / hist.sum()


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


def play_stretches(starts: Sequence[int], cuts: Sequence[int], fps: float, gap_s: float = STRETCH_GAP_S) -> list[int]:
    """Stretch number of each flight (by its start frame, in the given order): a new stretch after a camera cut or
    after a pause longer than `gap_s` between flight starts. Players change ends only between stretches."""
    order = sorted(range(len(starts)), key=lambda k: starts[k])
    out = [0] * len(starts)
    stretch, prev = 0, None
    for k in order:
        s = starts[k]
        if prev is not None and (s - prev > gap_s * fps or any(prev < c <= s for c in cuts)):
            stretch += 1
        out[k] = stretch
        prev = s
    return out


@dataclass
class Assignment:
    near_is_a: dict[int, bool] = field(default_factory=dict)  # per stretch
    margin: dict[int, float] = field(default_factory=dict)  # cost difference of the two ways round (0..2)
    a: np.ndarray | None = None  # mean clothing histogram of person A
    b: np.ndarray | None = None

    def decided(self, stretch: int) -> bool:
        return self.margin.get(stretch, 0.0) >= MIN_MARGIN


def assign(near: dict[int, list[np.ndarray]], far: dict[int, list[np.ndarray]], iterations: int = 20) -> Assignment:
    """Two people, A and B: in each stretch one is near and the other far. A starts as the near player of the
    stretch with most samples; then each stretch takes the cheaper way round, and A and B are re-estimated from the
    decided stretches (all of them while none is), weighted by their samples. The returned sides and margins are
    those against the final A and B."""
    keys = [k for k in sorted(near) if near[k] and far.get(k)]
    if not keys:
        return Assignment()
    n = {k: np.mean(near[k], axis=0) for k in keys}
    f = {k: np.mean(far[k], axis=0) for k in keys}
    w = {k: len(near[k]) + len(far[k]) for k in keys}
    big = max(keys, key=lambda k: (w[k], -k))
    a, b = n[big].copy(), f[big].copy()

    def sides(a: np.ndarray, b: np.ndarray) -> tuple[dict[int, bool], dict[int, float]]:
        side, margin = {}, {}
        for k in keys:
            c0 = hellinger(n[k], a) + hellinger(f[k], b)
            c1 = hellinger(n[k], b) + hellinger(f[k], a)
            side[k], margin[k] = c0 <= c1, abs(c0 - c1)
        return side, margin

    prev: dict[int, bool] | None = None
    for _ in range(iterations):
        side, margin = sides(a, b)
        use = [k for k in keys if margin[k] >= MIN_MARGIN] or keys
        wt = [w[k] for k in use]
        a = np.average([n[k] if side[k] else f[k] for k in use], axis=0, weights=wt)
        b = np.average([f[k] if side[k] else n[k] for k in use], axis=0, weights=wt)
        if side == prev:
            break
        prev = side
    side, margin = sides(a, b)
    return Assignment(near_is_a=side, margin=margin, a=a, b=b)
