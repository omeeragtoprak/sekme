"""Ball detection with a three-frame heatmap network plus track cleaning.

Two networks share one contract (three consecutive 512x288 frames in, one heatmap per frame out):
our own SEKME U-Net (models/ball_unet.py, trained by scripts/train_ball.py) and WASB (Tarashima et
al., BMVC 2023, MIT). The checkpoint says which one it is. We slide the window one frame at a time
and average the (up to three) heatmaps every frame receives.

The model also fires on small white court marks (the baseline centre marks);
with a calibrated frame these are recognised because they stay on one spot of
the court, which a ball in play never does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from courtvision.calib.camera import apply_h
from courtvision.models.ball_unet import ARCH as UNET_ARCH
from courtvision.models.ball_unet import BallUNet
from courtvision.models.wasb_hrnet import WASB_CONFIG, HRNet

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class BallCandidate:
    x: float
    y: float
    score: float


def ball_arch(weights: str) -> str:
    """Which network a ball checkpoint holds: ``sekme-ball-unet`` (ours) or ``wasb-hrnet``."""
    return torch.load(weights, map_location="cpu", weights_only=False).get("arch", "wasb-hrnet")


def load_ball_net(weights: str) -> tuple[torch.nn.Module, str]:
    """The network in a ball checkpoint and its name: our U-Net (has ``arch``) or WASB's HRNet."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if ckpt.get("arch") == UNET_ARCH:
        net: torch.nn.Module = BallUNet()
        net.load_state_dict(ckpt["model"], strict=True)
        return net, UNET_ARCH
    net = HRNet(WASB_CONFIG)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    return net, "wasb-hrnet"


class BallDetector:
    IN_W, IN_H = WASB_CONFIG["inp_width"], WASB_CONFIG["inp_height"]

    def __init__(self, weights: str, device: torch.device, threshold: float = 0.5, half: bool = False) -> None:
        # fp16 on GPU/MPS: 72 -> 46 ms/frame for WASB on the US Open clip, same detections, top-1 shift <= 0.72 px.
        self.device = device
        self.half = half and device.type != "cpu"
        self.net, self.arch = load_ball_net(weights)
        self.net.eval().to(device)
        if self.half:
            self.net.half()
        self.threshold = threshold

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(frame, (self.IN_W, self.IN_H), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)

    @torch.inference_mode()
    def heatmaps(self, prepped: list[np.ndarray]) -> np.ndarray:
        """Three prepped frames (3, H, W each) -> sigmoid heatmaps (3, H, W)."""
        x = torch.from_numpy(np.concatenate(prepped, axis=0)[None]).to(self.device)
        if self.half:
            x = x.half()
        out = self.net(x)
        logits = out[0] if isinstance(out, (dict, list, tuple)) else out  # WASB returns {0: logits}
        return torch.sigmoid(logits.float()).cpu().numpy()[0]

    def candidates(self, hm: np.ndarray, scale: tuple[float, float]) -> list[BallCandidate]:
        """Connected blobs above threshold -> heatmap-weighted centroids (WASB 'concomp')."""
        if hm.max() <= self.threshold:
            return []
        mask = (hm > self.threshold).astype(np.uint8)
        n, labels = cv2.connectedComponents(mask)
        # Pixel-centre decoding for both networks. For our U-Net (trained on labels x * scale) it was chosen on
        # the selection games 6-7 against decoding x * scale: F1 at 4 px 0.827 vs 0.780 (the resize's own
        # half-pixel shift makes the ball appear where the centre convention puts it).
        off = 0.5
        out = []
        for m in range(1, n):
            ys, xs = np.nonzero(labels == m)
            w = hm[ys, xs]
            cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
            out.append(BallCandidate((cx + off) * scale[0] - off, (cy + off) * scale[1] - off, float(w.sum())))
        return sorted(out, key=lambda c: -c.score)

    def start(self, frame_size: tuple[int, int], stride: int = 1) -> None:
        """Begin a streaming pass; feed frames in order with :meth:`push`, then :meth:`finish`.

        ``stride`` spaces the three frames of a window (t, t+s, t+2s). Both networks were trained on 30 fps
        video, so 60 fps footage uses s = 2 to show them the motion they learned; every frame still gets
        one inference and up to three averaged heatmaps.
        """
        self._scale = (frame_size[0] / self.IN_W, frame_size[1] / self.IN_H)
        self._stride = max(1, int(stride))
        self._buf: deque[tuple[int, np.ndarray]] = deque(maxlen=2 * self._stride + 1)
        self._acc: dict[int, np.ndarray] = {}
        self._cnt: dict[int, int] = {}
        self._out: dict[int, list[BallCandidate]] = {}

    def push(self, idx: int, frame: np.ndarray) -> None:
        self._buf.append((idx, self._prep(frame)))
        if len(self._buf) == self._buf.maxlen:
            window = [self._buf[k] for k in range(0, len(self._buf), self._stride)]
            hms = self.heatmaps([p for _, p in window])
            for (j, _), hm in zip(window, hms, strict=True):
                self._acc[j] = self._acc.get(j, 0.0) + hm
                self._cnt[j] = self._cnt.get(j, 0) + 1
        # A frame is final once it can no longer appear in a window (keeps memory bounded).
        done = [j for j in self._acc if j <= idx - 2 * self._stride]
        for j in done:
            self._out[j] = self.candidates(self._acc.pop(j) / self._cnt.pop(j), self._scale)

    def finish(self, n_frames: int) -> list[list[BallCandidate]]:
        for j in list(self._acc):
            self._out[j] = self.candidates(self._acc.pop(j) / self._cnt.pop(j), self._scale)
        return [self._out.get(j, []) for j in range(n_frames)]

    def run(self, frames_iter, n_frames: int, frame_size: tuple[int, int],
            stride: int = 1) -> list[list[BallCandidate]]:
        """Sliding-window inference over an iterator of (index, frame); candidates per frame."""
        self.start(frame_size, stride)
        for idx, frame in frames_iter:
            self.push(idx, frame)
        return self.finish(n_frames)


@dataclass
class BallTrack:
    xy: np.ndarray  # (N, 2) pixels, NaN where unknown
    score: np.ndarray  # (N,)
    source: np.ndarray  # (N,) 0 none, 1 detected, 2 interpolated


def drop_court_fixed(cands: list[list[BallCandidate]], ground_h: list[np.ndarray | None], radius_m: float = 0.25,
                     win: int = 15, min_frames: int = 10) -> list[list[BallCandidate]]:
    """Remove candidates that stay on one spot of the court (painted marks, a ball lying on the court).

    A ball in play never stays within ``radius_m`` of one ground point in ``min_frames`` of the
    ``2 * win`` neighbouring frames; a toss projects onto a long stretch of the camera ray.
    ``ground_h`` holds each frame's image -> court homography (None where uncalibrated: kept as is).
    """
    n = len(cands)
    g = [apply_h(H, np.array([[c.x, c.y] for c in cs])) if H is not None and cs else None
         for cs, H in zip(cands, ground_h, strict=True)]
    out: list[list[BallCandidate]] = []
    for i in range(n):
        if g[i] is None:
            out.append(cands[i])
            continue
        nb = [g[j] for j in range(max(0, i - win), min(n, i + win + 1)) if j != i and g[j] is not None]
        keep = [c for c, p in zip(cands[i], g[i], strict=True)
                if sum(bool(np.min(np.linalg.norm(q - p, axis=1)) < radius_m) for q in nb) < min_frames]
        out.append(keep)
    return out


def track_ball(cands: list[list[BallCandidate]], frame_w: int, max_gap: int = 6) -> BallTrack:
    """Pick one candidate per frame with a constant-acceleration predictor, then clean and fill gaps."""
    n = len(cands)
    max_disp = 300.0 * frame_w / 1280.0
    xy = np.full((n, 2), np.nan)
    score = np.zeros(n)
    seen: list[int] = []  # frames with a selection
    for i in range(n):
        prev = [xy[i - k] for k in (1, 2, 3) if i - k >= 0]
        pred = None
        if len(prev) == 3 and all(np.isfinite(p).all() for p in prev):
            p1, p2, p3 = prev
            acc = (p1 - p2) - (p2 - p3)
            vel = (p1 - p2) + acc  # the next step under constant acceleration
            pred = p1 + vel
        # Across a short miss keep gating and predicting from the last seen position, so a
        # single missed detection does not hand the track to a court mark or a spectator.
        last = seen[-1] if seen and i - seen[-1] <= max_gap else None
        if pred is None and last is not None:
            pred = xy[last].copy()
            if len(seen) > 1 and last - seen[-2] <= max_gap:
                pred += (xy[last] - xy[seen[-2]]) / (last - seen[-2]) * (i - last)
        best, best_s = None, -np.inf
        for c in cands[i]:
            p = np.array([c.x, c.y])
            if last is not None and np.linalg.norm(p - xy[last]) > max_disp * (i - last):
                continue
            s = c.score - (np.linalg.norm(pred - p) if pred is not None else 0.0)
            if s > best_s:
                best, best_s = p, s
                score[i] = c.score
        if best is not None:
            xy[i] = best
            seen.append(i)
    xy = _remove_outliers(xy)
    source = np.where(np.isfinite(xy[:, 0]), 1, 0)
    xy, source = _fill_gaps(xy, source, max_gap)
    score[source != 1] = 0.0
    return BallTrack(xy, score, source)


def _remove_outliers(xy: np.ndarray, win: int = 3, k: float = 4.0, floor_px: float = 12.0) -> np.ndarray:
    """Drop isolated points and points far from a local quadratic fit of their neighbours, worst first.

    A removed point is never a neighbour again, and the remaining points near it are re-tested. (The first
    version tested points in order against neighbours it had already removed: NaN fits that never count as
    too far, so a lone wild point could survive while the good points beside it were dropped.)
    """
    xy = xy.copy()
    n = len(xy)
    ok = np.isfinite(xy[:, 0])

    def ratio(i: int) -> float:  # residual / threshold; > 1 means too far, inf means isolated
        nb = [j for j in range(max(0, i - win), min(n, i + win + 1)) if j != i and ok[j]]
        if not nb:
            return np.inf
        if len(nb) < 4:
            return 0.0
        t = np.array(nb, dtype=float)
        pred = [np.polyval(np.polyfit(t, xy[nb, d], 2), i) for d in (0, 1)]
        resid = np.hypot(xy[i, 0] - pred[0], xy[i, 1] - pred[1])
        spread = np.median(np.hypot(*np.diff(xy[nb], axis=0).T))
        return float(resid / max(k * spread, floor_px * 4))

    r = np.full(n, -np.inf)
    for i in np.flatnonzero(ok):
        r[i] = ratio(i)
    while n:
        i = int(np.argmax(r))
        if r[i] <= 1.0:
            break
        ok[i], r[i] = False, -np.inf
        xy[i] = np.nan
        for j in range(max(0, i - win), min(n, i + win + 1)):
            if ok[j]:
                r[j] = ratio(j)
    return xy


def _fill_gaps(xy: np.ndarray, source: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill short gaps with a quadratic through the neighbouring detections (marked as interpolated)."""
    xy, source = xy.copy(), source.copy()
    ok = np.flatnonzero(np.isfinite(xy[:, 0]))
    for a, b in zip(ok[:-1], ok[1:], strict=True):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            idx = [j for j in ok if a - 3 <= j <= b + 3]
            t = np.array(idx, dtype=float)
            deg = 2 if len(idx) >= 4 else 1
            for j in range(a + 1, b):
                xy[j] = [np.polyval(np.polyfit(t, xy[idx, d], deg), j) for d in (0, 1)]
                source[j] = 2
    return xy, source
