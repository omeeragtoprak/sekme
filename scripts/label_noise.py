"""How noisy is each ball position: the TrackNet label, the image centroid, a network's detection?

    .venv/bin/python scripts/label_noise.py runs_ball_v3 8 9 10     (runs dir under data/tracknet, then games)

Three-cornered hat: on the same frames, three independent measurements of the ball's position — the label,
the ball's centroid in the image (no network, as scripts/label_offset.py finds it) and the network's
detection from the run's cached candidates. After removing each pair's per-game constant + speed term
(systematic, see label_offset.py), the variance of each pairwise difference is the sum of two noise
variances, which gives all three. Robust (MAD) variances, so a few wrong matches do not dominate. The
label noise is what calibration_check.py feeds into the reference (1 px).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from label_offset import GAP, R  # noqa: E402
from train_ball import SRC_H, SRC_W, clips, labels, to_src  # noqa: E402

DATA = ROOT / "data" / "tracknet"


def samples(runs: str, games: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Game name per row and (label xy, centroid xy, detection xy, label velocity) per row."""
    rows, game = [], []
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    for clip in [c for g in games for c in clips(range(g, g + 1))]:
        cache = DATA / runs / f"{clip.parent.name}_{clip.name}" / "perception.pkl"
        if not cache.exists():
            continue
        cands = pickle.loads(cache.read_bytes())["cands"]
        names, net_pts = labels(clip)
        pts = [None if p is None else to_src(*p, centred=False) for p in net_pts]
        frames: dict[int, np.ndarray] = {}

        def gray(i: int, names=names, clip=clip, frames=frames) -> np.ndarray:
            if i not in frames:
                frames[i] = cv2.imread(str(clip / names[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return frames[i]

        for i in range(GAP, min(len(pts), len(cands)) - GAP):
            p, a, b = pts[i], pts[i - GAP], pts[i + GAP]
            if p is None or a is None or b is None or not cands[i]:
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
            centroid = ((m * xx).sum() / m.sum() + x0, (m * yy).sum() / m.sum() + y0)
            det = np.array([[k.x, k.y] for k in cands[i]])
            j = int(np.argmin(np.hypot(*(det - p).T)))
            if np.hypot(*(det[j] - p)) > 10:
                continue
            rows.append((*p, *centroid, *det[j], *((np.array(b) - np.array(a)) / (2 * GAP))))
            game.append(clip.parent.name)
    return np.array(game), np.array(rows, float)


def residual(diff: np.ndarray, game: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Remove a per-game constant + speed term from each axis of a pairwise difference."""
    out = np.empty_like(diff)
    for g in np.unique(game):
        s = game == g
        for ax in (0, 1):
            a = np.column_stack([np.ones(s.sum()), vel[s, ax]])
            coef, *_ = np.linalg.lstsq(a, diff[s, ax], rcond=None)
            out[s, ax] = diff[s, ax] - a @ coef
    return out


def robust_var(x: np.ndarray) -> float:
    return float((1.4826 * np.median(np.abs(x - np.median(x)))) ** 2)


def main() -> None:
    runs, games = sys.argv[1], [int(g) for g in sys.argv[2:]] or [8, 9, 10]
    game, a = samples(runs, games)
    lab, cen, det, vel = a[:, 0:2], a[:, 2:4], a[:, 4:6], a[:, 6:8]
    for ax, name in ((0, "x"), (1, "y")):
        lc, ld, cd = (robust_var(residual(u - w, game, vel)[:, ax]) for u, w in ((lab, cen), (lab, det), (cen, det)))
        s_lab, s_cen, s_det = (np.sqrt(max(0.0, (p + q - r) / 2)) for p, q, r in
                               ((lc, ld, cd), (lc, cd, ld), (ld, cd, lc)))
        print(f"{runs} games {games} n={len(a)} {name}: noise SD  label {s_lab:.2f}  centroid {s_cen:.2f}  "
              f"detection {s_det:.2f} px  (pairwise SD label-centroid {np.sqrt(lc):.2f}, label-detection "
              f"{np.sqrt(ld):.2f}, centroid-detection {np.sqrt(cd):.2f})")


if __name__ == "__main__":
    main()
