"""Final test of ball networks on TrackNet games 8-10 (never used for training or selection).

    .venv/bin/python scripts/eval_ball.py models/sekme_ball.pt models/wasb_tennis_best.pth.tar

Each checkpoint runs through the production detector (perception/ball.py: sliding three-frame window,
averaged heatmaps, connected-component centroids); the highest-scoring candidate per frame is scored
against the label at 4 and 10 px in 1280x720 pixels. Speed is measured on the same frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_ball import SRC_H, SRC_W, clips, detection_scores, labels  # noqa: E402

from courtvision.models.ball_unet import IN_H, IN_W  # noqa: E402
from courtvision.perception.ball import BallDetector  # noqa: E402

TEST = range(8, 11)


def run(weights: str, dev: torch.device, games: range = TEST, shifts: tuple[float, ...] = (0.0,)) -> dict:
    det = BallDetector(weights, dev, half=dev.type != "cpu")
    pred, truth, n, secs = [], [], 0, 0.0
    for clip in clips(games):
        names, pts = labels(clip)
        frames = [cv2.imread(str(clip / name)) for name in names]
        det.start((SRC_W, SRC_H))
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            det.push(i, f)
        out = det.finish(len(frames))
        if dev.type == "mps":
            torch.mps.synchronize()
        secs += time.perf_counter() - t0
        n += len(frames)
        for cands, p in zip(out, pts, strict=True):
            pred.append((cands[0].x, cands[0].y) if cands else None)
            truth.append(None if p is None else (p[0] * SRC_W / IN_W, p[1] * SRC_H / IN_H))
    out = {"arch": det.arch, "frames": n, "ms_per_frame": round(1000 * secs / n, 1)}
    for sh in shifts:  # a constant shift of the decoded position, to compare pixel conventions
        moved = [None if p is None else (p[0] + sh, p[1] + sh) for p in pred]
        out[f"shift {sh:g}px" if len(shifts) > 1 else "scores"] = detection_scores(moved, truth)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--games", nargs=2, type=int, default=(8, 10), help="first and last TrackNet game")
    ap.add_argument("--shifts", nargs="*", type=float, default=[0.0])
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for w in args.weights:
        print(Path(w).name, json.dumps(run(w, dev, range(args.games[0], args.games[1] + 1), tuple(args.shifts))),
              flush=True)


if __name__ == "__main__":
    main()
