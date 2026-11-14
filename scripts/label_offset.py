"""Where is the ball in the image relative to its TrackNet label? A check that uses no network.

    .venv/bin/python scripts/label_offset.py            (selection clips: every fifth clip of games 1-7)
    .venv/bin/python scripts/label_offset.py --test     (all clips of games 8-10)
    .venv/bin/python scripts/label_offset.py --per-game (each game: its selection clips for 1-7, all clips for 8-10)

For a labelled frame t, the moving ball shows in both |f_t - f_(t-3)| and |f_t - f_(t+3)|, while the ball at
t-3 or t+3 shows in only one of them, so their per-pixel minimum isolates the ball at t. Its centroid within
8 px of the label (pixels above 40 % of the window's peak; intensity-weighted, and unweighted as a check that
lighting does not drive it) is compared with the label, in 1280x720 index coordinates. Frames where the ball
moved less than 16 px over either leg (t-3 to t, or t to t+3) are skipped (it would overlap itself in that
difference), which keeps somewhat faster balls than a test on t-3 to t+3 alone would.

The offset depends on the ball's speed (labels can lead or trail the ball in the image), so each axis is also
fitted as offset = constant + slope * velocity (velocity from the labels at t-3 and t+3, px per frame). Only the
constant moves a bounce reference: a point shifted along its own track stays on the same straight line, so the
meeting point of the incoming and outgoing tracks does not move; the constant is what GAME_LABEL_OFFSET_PX holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_ball import SRC_H, SRC_W, clips, labels, to_src  # noqa: E402

R, GAP = 8, 3


def offsets(clip_dirs: list[Path]) -> np.ndarray:
    """(n, 6): weighted dx, dy, unweighted dx, dy of the image centroid minus the label; label velocity vx, vy."""
    out = []
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    for clip in clip_dirs:
        names, net_pts = labels(clip)
        pts = [None if p is None else to_src(*p, centred=False) for p in net_pts]
        cache: dict[int, np.ndarray] = {}

        def gray(i: int, names=names, clip=clip, cache=cache) -> np.ndarray:
            if i not in cache:
                cache[i] = cv2.imread(str(clip / names[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return cache[i]

        for i in range(GAP, len(pts) - GAP):
            p, a, b = pts[i], pts[i - GAP], pts[i + GAP]
            if p is None or a is None or b is None:
                continue
            if min(np.hypot(*np.subtract(p, a)), np.hypot(*np.subtract(p, b))) < 2 * R:
                continue
            x0, y0 = int(round(p[0])), int(round(p[1]))
            if not (R < x0 < SRC_W - R - 1 and R < y0 < SRC_H - R - 1):
                continue
            win = (slice(y0 - R, y0 + R + 1), slice(x0 - R, x0 + R + 1))
            m = np.minimum(np.abs(gray(i)[win] - gray(i - GAP)[win]), np.abs(gray(i)[win] - gray(i + GAP)[win]))
            m = np.where(m > max(12.0, 0.4 * m.max()), m, 0.0)
            if m.sum() < 50:
                continue
            u = (m > 0).astype(np.float32)
            out.append(((m * xx).sum() / m.sum() + x0 - p[0], (m * yy).sum() / m.sum() + y0 - p[1],
                        (u * xx).sum() / u.sum() + x0 - p[0], (u * yy).sum() / u.sum() + y0 - p[1],
                        (b[0] - a[0]) / (2 * GAP), (b[1] - a[1]) / (2 * GAP)))
    return np.array(out)


def constant_and_slope(offset: np.ndarray, velocity: np.ndarray) -> tuple[float, float, float]:
    """Least squares offset = c + k * velocity; returns c, its standard error and k."""
    a = np.stack([np.ones_like(velocity), velocity], axis=1)
    coef, *_ = np.linalg.lstsq(a, offset, rcond=None)
    resid = offset - a @ coef
    cov = np.linalg.inv(a.T @ a) * resid.var(ddof=2)
    return float(coef[0]), float(np.sqrt(cov[0, 0])), float(coef[1])


def describe(name: str, d: np.ndarray) -> str:
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    (cx, sx, kx), (cy, sy, ky) = constant_and_slope(d[:, 0], d[:, 4]), constant_and_slope(d[:, 1], d[:, 5])
    return (f"{name}: n={len(d)}  image centre - label (weighted)  mean dx {d[:, 0].mean():+.2f} (SE {se[0]:.2f}) "
            f"dy {d[:, 1].mean():+.2f} (SE {se[1]:.2f})  |  constant dx {cx:+.2f} (SE {sx:.2f}) dy {cy:+.2f} "
            f"(SE {sy:.2f}) px, slope x {kx:+.3f} y {ky:+.3f} frame")


def main() -> None:
    if "--per-game" in sys.argv[1:]:
        for g in range(1, 11):
            print(describe(f"game{g}", offsets(clips(range(g, g + 1)) if g >= 8 else clips(range(g, g + 1))[::5])),
                  flush=True)
        return
    d = offsets(clips(range(8, 11)) if "--test" in sys.argv[1:] else clips(range(1, 8))[::5])
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    for name, k in (("intensity-weighted", 0), ("unweighted", 2)):
        print(f"{name}: n={len(d)}  image centre - label  dx {d[:, k].mean():+.2f} (SE {se[k]:.2f}, median "
              f"{np.median(d[:, k]):+.2f})  dy {d[:, k + 1].mean():+.2f} (SE {se[k + 1]:.2f}, median "
              f"{np.median(d[:, k + 1]):+.2f}) px")


if __name__ == "__main__":
    main()
