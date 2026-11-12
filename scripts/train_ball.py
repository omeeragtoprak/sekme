"""Train the SEKME ball network (models/ball_unet.py) on the TrackNet tennis labels.

    .venv/bin/python scripts/train_ball.py [--epochs 12] [--batch 8] [--max-steps N]

Split (fixed before training): games 1-5 train, games 6-7 validation (epoch selection), games 8-10
untouched here (final test: scripts/eval_ball.py). Frames are resized once to 512x288 into
data/tracknet/cache/. Output: models/sekme_ball.pt with the split and validation scores inside.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.models.ball_unet import ARCH, FRAMES, IN_H, IN_W, BallUNet, disk_targets, wbce_loss  # noqa: E402

DATA = ROOT / "data" / "tracknet"
CACHE = DATA / "cache"
OUT = ROOT / "models" / "sekme_ball.pt"
TRAIN, VAL = range(1, 6), range(6, 8)
SRC_W, SRC_H = 1280, 720
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def clips(games: range) -> list[Path]:
    return [c for g in games
            for c in sorted((DATA / "Dataset" / f"game{g}").glob("Clip*"), key=lambda p: int(p.name[4:]))]


def build_cache(all_clips: list[Path]) -> None:
    def one(src: Path) -> None:
        dst = CACHE / src.parent.parent.name / src.parent.name / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.resize(cv2.imread(str(src)), (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    files = [f for c in all_clips for f in sorted(c.glob("*.jpg"))]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, files))


def labels(clip: Path) -> tuple[list[str], list[tuple[float, float] | None]]:
    """Frame names and ball positions in network-input pixels (None: not visible)."""
    names, pts = [], []
    for r in csv.DictReader((clip / "Label.csv").open()):
        names.append(r["file name"])
        v = (r.get("visibility") or "0").strip()
        ok = v not in ("", "0") and (r.get("x-coordinate") or "") != ""
        pts.append((float(r["x-coordinate"]) * IN_W / SRC_W, float(r["y-coordinate"]) * IN_H / SRC_H) if ok else None)
    return names, pts


class Windows(Dataset):
    """Every run of three consecutive labelled frames inside one clip."""

    def __init__(self, clip_dirs: list[Path], augment: bool, stride: int = 1) -> None:
        self.augment = augment
        self.items = []
        for c in clip_dirs:
            names, pts = labels(c)
            folder = CACHE / c.parent.name / c.name
            for t in range(0, len(names) - FRAMES + 1, stride):
                self.items.append(([folder / n for n in names[t:t + FRAMES]], pts[t:t + FRAMES]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        paths, pts = self.items[i]
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in paths]
        pts = list(pts)
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                imgs = [im[:, ::-1] for im in imgs]
                pts = [None if p is None else (IN_W - 1 - p[0], p[1]) for p in pts]
            gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-0.08, 0.08)
            imgs = [np.clip(im * gain + bias, 0, 1) for im in imgs]
        x = np.concatenate([((im - MEAN) / STD).transpose(2, 0, 1) for im in imgs], axis=0)
        truth = torch.tensor([[-1.0, -1.0] if p is None else [p[0], p[1]] for p in pts])
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(disk_targets(pts)), truth


def detection_scores(pred: list[tuple[float, float] | None], truth: list[tuple[float, float] | None],
                     radii: tuple[float, ...] = (4.0, 10.0)) -> dict:
    """Per-frame top-1 scoring in source pixels: a prediction counts only within ``r`` of a visible ball."""
    out = {}
    for r in radii:
        tp = fp = fn = 0
        for p, t in zip(pred, truth, strict=True):
            hit = p is not None and t is not None and np.hypot(p[0] - t[0], p[1] - t[1]) <= r
            tp += hit
            fp += p is not None and not hit
            fn += t is not None and not hit
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"{r:g}px"] = {"precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}
    return out


def validate(net: torch.nn.Module, loader: DataLoader, dev: torch.device, thr: float = 0.5) -> dict:
    """Middle frame of each window: argmax of the heatmap if above ``thr``, in 1280x720 pixels."""
    pred, truth = [], []
    net.eval()
    with torch.inference_mode():
        for x, _, pts in loader:
            hm = torch.sigmoid(net(x.to(dev))[:, 1]).float().cpu().numpy()
            for b in range(len(hm)):
                t = pts[b, 1].numpy()
                truth.append((t[0] * SRC_W / IN_W, t[1] * SRC_H / IN_H) if t[0] >= 0 else None)
                if hm[b].max() > thr:
                    y, x_ = np.unravel_index(int(hm[b].argmax()), hm[b].shape)
                    pred.append(((x_ + 0.5) * SRC_W / IN_W - 0.5, (y + 0.5) * SRC_H / IN_H - 0.5))
                else:
                    pred.append(None)
    net.train()
    return detection_scores(pred, truth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (timing / smoke test)")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    build_cache(clips(TRAIN) + clips(VAL))
    print(f"cache ready in {time.time() - t0:.0f} s", flush=True)
    train = DataLoader(Windows(clips(TRAIN), augment=True), batch_size=args.batch, shuffle=True, num_workers=2,
                       persistent_workers=True, drop_last=True)
    val = DataLoader(Windows(clips(VAL), augment=False, stride=4), batch_size=16, num_workers=2,
                     persistent_workers=True)
    net = BallUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train),
                                                pct_start=0.1)
    best, step, history = -1.0, 0, []
    for epoch in range(args.epochs):
        t1, losses = time.time(), []
        for x, y, _ in train:
            loss = wbce_loss(net(x.to(dev)), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"{step} steps, {(time.time() - t1) / step:.3f} s/step, {len(train)} steps/epoch, "
                      f"loss {np.mean(losses):.5f}  val {json.dumps(validate(net, val, dev))}")
                return
        scores = validate(net, val, dev)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **scores})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.5f}  val {json.dumps(scores)}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if scores["4px"]["f1"] > best:
            best = scores["4px"]["f1"]
            OUT.parent.mkdir(exist_ok=True)
            torch.save({"arch": ARCH, "model": net.state_dict(), "epoch": epoch + 1, "val": scores,
                        "history": history, "trained_on": "TrackNet tennis games 1-5",
                        "selected_on": "TrackNet tennis games 6-7 (F1 at 4 px, 1280x720)", "args": vars(args)}, OUT)
    print("best val F1@4px", best, "->", OUT)


if __name__ == "__main__":
    main()
