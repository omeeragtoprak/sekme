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

from courtvision.eval.tracknet import LABEL_OFFSET_PX  # noqa: E402
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


LABEL_OFFSET = LABEL_OFFSET_PX  # TrackNet label minus the ball's image centre, 1280x720 px


def to_net(x: float, y: float, centred: bool) -> tuple[float, float]:
    """A label (1280x720) in network-input pixels. ``centred``: the ball's image centre, through the
    pixel-centre mapping cv2.resize uses (ball network v3 on); otherwise label * scale (v1, v2)."""
    if not centred:
        return x * IN_W / SRC_W, y * IN_H / SRC_H
    return ((x - LABEL_OFFSET[0] + 0.5) * IN_W / SRC_W - 0.5, (y - LABEL_OFFSET[1] + 0.5) * IN_H / SRC_H - 0.5)


def to_src(x: float, y: float, centred: bool) -> tuple[float, float]:
    """Inverse of ``to_net``; with ``centred`` it returns the label minus LABEL_OFFSET."""
    if not centred:
        return x * SRC_W / IN_W, y * SRC_H / IN_H
    return (x + 0.5) * SRC_W / IN_W - 0.5, (y + 0.5) * SRC_H / IN_H - 0.5


def labels(clip: Path, centred: bool = False) -> tuple[list[str], list[tuple[float, float] | None]]:
    """Frame names and ball positions in network-input pixels (None: not visible)."""
    names, pts = [], []
    for r in csv.DictReader((clip / "Label.csv").open()):
        names.append(r["file name"])
        v = (r.get("visibility") or "0").strip()
        ok = v not in ("", "0") and (r.get("x-coordinate") or "") != ""
        pts.append(to_net(float(r["x-coordinate"]), float(r["y-coordinate"]), centred) if ok else None)
    return names, pts


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A straight-line blur 3-9 px long at a random angle: a fast ball or a camera pan smears the frame.

    Odd lengths only: an even-length line is not centred on filter2D's anchor and would shift the image
    against its labels (by ~0.5 px at 512x288; ball network v2 was trained with that shift on 3 of 7
    lengths).
    """
    k = 2 * int(rng.integers(1, 5)) + 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


class Windows(Dataset):
    """Every run of three consecutive labelled frames inside one clip."""

    def __init__(self, clip_dirs: list[Path], augment: bool, stride: int = 1, blur: float = 0.0,
                 centred: bool = False) -> None:
        self.augment, self.blur = augment, blur
        self.items = []
        for c in clip_dirs:
            names, pts = labels(c, centred)
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
            if rng.random() < self.blur:
                seed = int(rng.integers(1 << 30))  # the same smear on all three frames
                imgs = [motion_blur(im, np.random.default_rng(seed)) for im in imgs]
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


def validate(net: torch.nn.Module, loader: DataLoader, dev: torch.device, thr: float = 0.5,
             centred: bool = False) -> dict:
    """Middle frame of each window: argmax of the heatmap if above ``thr``, in 1280x720 pixels."""
    pred, truth = [], []
    net.eval()
    with torch.inference_mode():
        for x, _, pts in loader:
            hm = torch.sigmoid(net(x.to(dev))[:, 1]).float().cpu().numpy()
            for b in range(len(hm)):
                t = pts[b, 1].numpy()
                truth.append(to_src(float(t[0]), float(t[1]), centred) if t[0] >= 0 else None)
                if hm[b].max() > thr:
                    y, x_ = np.unravel_index(int(hm[b].argmax()), hm[b].shape)
                    pred.append(((x_ + 0.5) * SRC_W / IN_W - 0.5, (y + 0.5) * SRC_H / IN_H - 0.5))
                else:
                    pred.append(None)
    net.train()
    return detection_scores(pred, truth)


RESUME_KEYS = ("epochs", "batch", "lr", "train_games", "blur", "centre_labels")


def state_path(out: Path) -> Path:
    return out.with_name(out.stem + ".state.pt")


def atomic_save(obj: dict, path: Path) -> None:
    """torch.save through a temporary file, so a kill mid-write never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def recipe(args: dict, steps_per_epoch: int) -> dict:
    """What a resumed run must share with the stopped one (the schedule spans epochs x steps_per_epoch)."""
    return {**{k: args[k] for k in RESUME_KEYS}, "steps_per_epoch": steps_per_epoch}


def save_state(path: Path, net: torch.nn.Module, opt: torch.optim.Optimizer, sched, *, epoch: int,
               best: float, step: int, history: list, recipe: dict) -> None:
    """Everything needed to continue after ``epoch``."""
    atomic_save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch, "best": best, "step": step, "history": history, "args": recipe,
                 "rng": torch.get_rng_state()}, path)


def load_state(path: Path, net: torch.nn.Module, opt: torch.optim.Optimizer, sched,
               recipe: dict) -> tuple[int, float, int, list]:
    """Restore a run saved by save_state; refuses a state written with a different recipe."""
    s = torch.load(path, map_location="cpu", weights_only=False)
    saved = s["args"]
    # A state written before steps_per_epoch was recorded still has to match the rest of the recipe.
    if set(RESUME_KEYS) - set(saved) or any(saved[k] != recipe.get(k) for k in saved):
        raise SystemExit(f"{path} was written with {saved}, this run is {recipe}")
    net.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    sched.load_state_dict(s["sched"])
    torch.set_rng_state(s["rng"])
    return s["epoch"], s["best"], s["step"], s["history"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (timing / smoke test)")
    ap.add_argument("--train-games", type=int, default=5, choices=range(1, 8),
                    help="train on games 1..N (6-7 select when N=5); games 8-10 are the test set")
    ap.add_argument("--blur", type=float, default=0.0, help="probability of motion-blur augmentation")
    ap.add_argument("--centre-labels", action="store_true",
                    help="targets at the ball's image centre (labels minus LABEL_OFFSET, pixel-centre mapping)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out stem>.state.pt (written after every epoch) if it exists")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    if args.train_games == 5:  # v1: games 1-5 train, games 6-7 choose the epoch
        train_clips, val_clips = clips(TRAIN), clips(VAL)
        trained_on, selected_on = "TrackNet tennis games 1-5", "TrackNet tennis games 6-7 (F1 at 4 px, 1280x720)"
    else:  # games 1..N, every fifth clip held out to choose the epoch (WASB trained on games 1-7)
        pool = clips(range(1, args.train_games + 1))
        val_clips = pool[::5]
        train_clips = [c for c in pool if c not in val_clips]
        trained_on = f"TrackNet tennis games 1-{args.train_games} minus every 5th clip"
        selected_on = f"every 5th clip of games 1-{args.train_games} (F1 at 4 px, 1280x720)"
    if args.centre_labels:
        trained_on += f", targets at the label minus {LABEL_OFFSET} px"
        selected_on = selected_on.replace("1280x720)", "1280x720, against the same targets)")
    build_cache(train_clips + val_clips)
    print(f"cache ready in {time.time() - t0:.0f} s; train {len(train_clips)} clips, select {len(val_clips)}",
          flush=True)
    c = args.centre_labels
    train = DataLoader(Windows(train_clips, augment=True, blur=args.blur, centred=c), batch_size=args.batch,
                       shuffle=True, num_workers=2, persistent_workers=True, drop_last=True)
    val = DataLoader(Windows(val_clips, augment=False, stride=4, centred=c), batch_size=16, num_workers=2,
                     persistent_workers=True)
    out = Path(args.out)
    net = BallUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train),
                                                pct_start=0.1)
    best, step, history, start = -1.0, 0, [], 0
    state, rec = state_path(out), recipe(vars(args), len(train))
    if args.resume and state.exists():
        start, best, step, history = load_state(state, net, opt, sched, rec)
        if out.exists():  # the best checkpoint is written just before the state: a kill in between
            ck = torch.load(out, map_location="cpu", weights_only=False)  # leaves it one epoch ahead
            if ck.get("epoch", 0) > start and all(ck["args"].get(k) == rec[k] for k in RESUME_KEYS):
                best = max(best, ck["val"]["4px"]["f1"])
        print(f"resumed after epoch {start} (best F1@4px {best})", flush=True)
    elif args.resume:
        print(f"--resume: no {state.name}, starting from scratch", flush=True)
    for epoch in range(start, args.epochs):
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
                      f"loss {np.mean(losses):.5f}  val {json.dumps(validate(net, val, dev, centred=c))}")
                return
        scores = validate(net, val, dev, centred=c)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **scores})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.5f}  val {json.dumps(scores)}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if scores["4px"]["f1"] > best:
            best = scores["4px"]["f1"]
            out.parent.mkdir(exist_ok=True)
            atomic_save({"arch": ARCH, "model": net.state_dict(), "epoch": epoch + 1, "val": scores,
                         "history": history, "trained_on": trained_on, "selected_on": selected_on,
                         "args": vars(args), "label_offset_px": LABEL_OFFSET if c else None}, out)
        save_state(state, net, opt, sched, epoch=epoch + 1, best=best, step=step, history=history, recipe=rec)
    print("best val F1@4px", best, "->", out)


if __name__ == "__main__":
    main()
