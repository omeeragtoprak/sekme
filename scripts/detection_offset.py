"""Where does a ball network put the ball relative to the TrackNet label, split into a constant and a speed term?

    .venv/bin/python scripts/detection_offset.py runs runs_court runs_ball_v2 runs_ball_v3     (under data/tracknet)

Uses the cached ball candidates of each run (perception.pkl, 1280x720 index coordinates) on games 8-10: per
labelled frame, the candidate nearest the label within 10 px. Each axis is fitted as detection - label =
constant + slope * velocity (velocity from the labels at t-3 and t+3, px per frame), as scripts/label_offset.py
does for the ball's image centre. Subtracting that script's constant gives the network's constant offset from
the ball's centre, the part that moves a bounce; a speed term only shifts points along their own track.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from label_offset import GAP, constant_and_slope  # noqa: E402

from courtvision.eval.tracknet import GAME_LABEL_OFFSET_PX, load_labels  # noqa: E402

DATA = ROOT / "data" / "tracknet"
GAMES = ("game8", "game9", "game10")


def rows(runs: str, game: str) -> np.ndarray:
    """(n, 4): detection - label dx, dy; label velocity vx, vy."""
    out = []
    for clip in sorted((DATA / "Dataset" / game).glob("Clip*"), key=lambda p: int(p.name[4:])):
        cache = DATA / runs / f"{game}_{clip.name}" / "perception.pkl"
        if not cache.exists():
            continue
        cands = pickle.loads(cache.read_bytes())["cands"]
        lab = load_labels(clip / "Label.csv")
        for i in range(GAP, min(len(cands), len(lab.xy)) - GAP):
            if not (lab.visible[i] and lab.visible[i - GAP] and lab.visible[i + GAP]) or not cands[i]:
                continue
            d = np.array([[c.x, c.y] for c in cands[i]]) - lab.xy[i]
            j = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))
            if np.hypot(*d[j]) <= 10:
                out.append((*d[j], *((lab.xy[i + GAP] - lab.xy[i - GAP]) / (2 * GAP))))
    return np.array(out)


def main() -> None:
    for runs in sys.argv[1:]:
        for game in GAMES:
            r = rows(runs, game)
            (cx, sx, kx), (cy, sy, ky) = constant_and_slope(r[:, 0], r[:, 2]), constant_and_slope(r[:, 1], r[:, 3])
            ox, oy = GAME_LABEL_OFFSET_PX[game]
            print(f"{runs} {game}: n={len(r)}  detection - label  mean dx {r[:, 0].mean():+.2f} "
                  f"dy {r[:, 1].mean():+.2f}"
                  f"  |  constant dx {cx:+.2f} (SE {sx:.2f}) dy {cy:+.2f} (SE {sy:.2f}), slope x {kx:+.3f} y {ky:+.3f}"
                  f"  |  constant from the ball's centre dx {cx + ox:+.2f} dy {cy + oy:+.2f} px", flush=True)


if __name__ == "__main__":
    main()
