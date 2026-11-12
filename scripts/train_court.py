"""Train the SEKME court-keypoint network (models/court_unet.py) on the TennisCourtDetector labels.

    .venv/bin/python scripts/train_court.py [--epochs 12] [--max-steps N]
    .venv/bin/python scripts/train_court.py --test models/sekme_court.pt models/court_kps_yastrebksv.pt

Data: data/court/ (8 841 broadcast frames, 14 keypoints each, resized once to 640x360; split files
data_train.json / data_val.json from the dataset). The official split is by frame, so validation
frames share videos with training: 10 % of the *training videos* are held out here to choose the epoch,
and data_val.json is scored once with --test (both networks trained on data_train.json).
Metric (as the dataset's authors report it): a keypoint is correct within 7 px at 1280x720.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.models.court_unet import ARCH, IN_H, IN_W, KEYPOINTS, CourtUNet, court_targets, flip_points  # noqa: E402
from courtvision.models.unet import wbce_loss  # noqa: E402
from courtvision.perception.court import CourtKeypointDetector, heatmap_peak  # noqa: E402

DATA = ROOT / "data" / "court"
OUT = ROOT / "models" / "sekme_court.pt"
SCALE = 1280 / IN_W  # label pixels per network pixel
TOL_PX = 7.0  # at 1280x720


def records(split: str) -> list[dict]:
    return json.loads((DATA / f"data_{split}.json").read_text())


def held_out(video: str) -> bool:
    return int(hashlib.sha1(video.encode()).hexdigest(), 16) % 10 == 0


def to_net(kps: list[list[float]]) -> list[tuple[float, float] | None]:
    out = []
    for x, y in kps:
        inside = 0 <= x < 1280 and 0 <= y < 720
        out.append((x / SCALE, y / SCALE) if inside else None)
    return out


class Frames(Dataset):
    def __init__(self, recs: list[dict], augment: bool) -> None:
        self.recs, self.augment = recs, augment

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        r = self.recs[i]
        img = cv2.imread(str(DATA / "images" / f"{r['id']}.jpg"))
        pts = to_net(r["kps"])
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1]
                pts = flip_points(pts)
            # small zoom / shift, as broadcast framing varies between tournaments
            s, tx, ty = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05) * IN_W, rng.uniform(-0.05, 0.05) * IN_H
            m = np.array([[s, 0, (1 - s) * IN_W / 2 + tx], [0, s, (1 - s) * IN_H / 2 + ty]], np.float32)
            img = cv2.warpAffine(np.ascontiguousarray(img), m, (IN_W, IN_H), borderMode=cv2.BORDER_REFLECT)
            pts = [None if p is None else (m[0, 0] * p[0] + m[0, 2], m[1, 1] * p[1] + m[1, 2]) for p in pts]
            pts = [p if p is not None and 0 <= p[0] < IN_W and 0 <= p[1] < IN_H else None for p in pts]
            img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        truth = torch.tensor([[-1.0, -1.0] if p is None else [p[0], p[1]] for p in pts])
        return x, torch.from_numpy(court_targets(pts)), truth


def keypoint_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """pred, truth: (N, 14, 2) in 1280x720 pixels, NaN = none. Correct = both present and within 7 px."""
    has_t, has_p = np.isfinite(truth[..., 0]), np.isfinite(pred[..., 0])
    d = np.linalg.norm(pred - truth, axis=-1)
    correct = has_t & has_p & (d < TOL_PX)
    both = has_t & has_p
    return {"precision": round(float(correct.sum() / max(has_p.sum(), 1)), 4),
            "accuracy": round(float(correct.sum() / max(has_t.sum(), 1)), 4),
            "median_px": round(float(np.median(d[both])), 2) if both.any() else None,
            "keypoints": int(has_t.sum())}


def validate(net: torch.nn.Module, loader: DataLoader, dev: torch.device, min_peak: float = 0.5) -> dict:
    preds, truths = [], []
    net.eval()
    with torch.inference_mode():
        for x, _, t in loader:
            hms = torch.sigmoid(net(x.to(dev))).float().cpu().numpy()
            for b in range(len(hms)):
                kp = np.full((KEYPOINTS, 2), np.nan)
                for k in range(KEYPOINTS):
                    pk = heatmap_peak(hms[b, k], min_peak)
                    if pk is not None:
                        kp[k] = pk[0] * SCALE, pk[1] * SCALE
                preds.append(kp)
                tt = t[b].numpy().astype(float) * SCALE
                tt[t[b].numpy()[:, 0] < 0] = np.nan
                truths.append(tt)
    net.train()
    return keypoint_scores(np.array(preds), np.array(truths))


def test(weights: list[str]) -> None:
    """Score checkpoints on data_val.json through the production detector."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    recs = records("val")
    truth = np.array([[[x, y] if 0 <= x < 1280 and 0 <= y < 720 else [np.nan, np.nan] for x, y in r["kps"]]
                      for r in recs], float)
    # The official split is by frame: most validation videos are also training videos. Two cleaner subsets:
    # videos absent from data_train.json (unseen by both networks) and videos we held out from our training
    # (unseen by ours, seen by a network trained on all of data_train.json).
    train_videos = {r["id"].rsplit("_", 1)[0] for r in records("train")}
    video = [r["id"].rsplit("_", 1)[0] for r in recs]
    subsets = {"all": np.ones(len(recs), bool),
               "videos_not_in_train": np.array([v not in train_videos for v in video]),
               "videos_held_out_by_us": np.array([v in train_videos and held_out(v) for v in video])}
    for w in weights:
        det = CourtKeypointDetector(w, dev)
        preds, t0 = [], time.perf_counter()
        for k in range(0, len(recs), 16):
            imgs = [cv2.imread(str(DATA / "images" / f"{r['id']}.jpg")) for r in recs[k:k + 16]]
            kps, _ = det.detect_batch(imgs)
            preds.append(kps * SCALE)
        ms = 1000 * (time.perf_counter() - t0) / len(recs)
        pred = np.concatenate(preds)
        out = {name: {**keypoint_scores(pred[m], truth[m]), "frames": int(m.sum())} for name, m in subsets.items()}
        print(Path(w).name, det.arch, json.dumps({**out, "ms_per_frame": round(ms, 1)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--test", nargs="*")
    args = ap.parse_args()
    if args.test:
        test(args.test)
        return
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_recs = [r for r in records("train") if not held_out(r["id"].rsplit("_", 1)[0])]
    hold_recs = [r for r in records("train") if held_out(r["id"].rsplit("_", 1)[0])]
    print(f"train {len(train_recs)} frames, held-out {len(hold_recs)} frames (by video)", flush=True)
    train = DataLoader(Frames(train_recs, True), batch_size=args.batch, shuffle=True, num_workers=2,
                       persistent_workers=True, drop_last=True)
    hold = DataLoader(Frames(hold_recs, False), batch_size=16, num_workers=2, persistent_workers=True)
    net = CourtUNet().to(dev)
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
                      f"loss {np.mean(losses):.4f}  held-out {json.dumps(validate(net, hold, dev))}")
                return
        scores = validate(net, hold, dev)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **scores})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.4f}  held-out {json.dumps(scores)}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if scores["accuracy"] > best:
            best = scores["accuracy"]
            OUT.parent.mkdir(exist_ok=True)
            torch.save({"arch": ARCH, "model": net.state_dict(), "epoch": epoch + 1, "held_out": scores,
                        "history": history, "trained_on": "TennisCourtDetector data_train.json minus 10 % of videos",
                        "selected_on": "held-out training videos (accuracy at 7 px, 1280x720)",
                        "args": vars(args)}, OUT)
    print("best held-out accuracy", best, "->", OUT)


if __name__ == "__main__":
    main()
