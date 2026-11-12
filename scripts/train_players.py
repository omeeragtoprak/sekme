"""Distil the SEKME person network (models/person_unet.py) from a teacher detector.

    .venv/bin/python scripts/train_players.py --labels      # teacher boxes (YOLO26-s, imgsz 1280, conf 0.3)
    .venv/bin/python scripts/train_players.py [--epochs 6]  # train
    .venv/bin/python scripts/train_players.py --test models/sekme_person.pt

There are no human box labels for broadcast tennis here, so the targets are the persons the teacher
finds; the network is trained from scratch to reproduce them. Images: TrackNet frames (games 1-5 every
second frame for training, 6-7 for epoch choice, 8-10 for the test) and the 8 841 TennisCourtDetector
frames (training; many tournaments and courts). The test reports agreement with the teacher on games
8-10 (box IoU >= 0.5), for all persons and for large ones (taller than 60 px at 720p: the players and
nothing far in the stands).
"""

from __future__ import annotations

import argparse
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

from courtvision.models.person_unet import ARCH, IN_H, IN_W, PersonUNet, decode, person_loss, person_targets  # noqa: E402
from courtvision.perception.players import _iou  # noqa: E402

DATA = ROOT / "data"
LABELS = DATA / "players" / "teacher_boxes.json"
OUT = ROOT / "models" / "sekme_person.pt"
TEACHER = ROOT / "models" / "yolo26s.pt"


def tracknet_frames(games: range, step: int = 1) -> list[Path]:
    out = []
    for g in games:
        for clip in sorted((DATA / "tracknet" / "Dataset" / f"game{g}").glob("Clip*"), key=lambda p: int(p.name[4:])):
            out += sorted(clip.glob("*.jpg"))[::step]
    return out


def court_frames() -> list[Path]:
    return sorted((DATA / "court" / "images").glob("*.jpg"))


def make_labels() -> None:
    from ultralytics import YOLO

    model = YOLO(str(TEACHER))
    paths = tracknet_frames(range(1, 11)) + court_frames()
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if str(p.relative_to(DATA)) not in labels]
    t0 = time.time()
    for k in range(0, len(todo), 16):
        batch = todo[k:k + 16]
        for p, r in zip(batch, model.predict([str(p) for p in batch], imgsz=1280, conf=0.3, classes=[0],
                                             device="mps", half=True, verbose=False), strict=True):
            h, w = r.orig_shape
            labels[str(p.relative_to(DATA))] = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h, c] for b, c in
                                                zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                                    r.boxes.conf.cpu().numpy().tolist(), strict=True)]
        if k % 1600 == 0:
            print(f"{k + len(batch)}/{len(todo)} images, {time.time() - t0:.0f} s", flush=True)
            LABELS.write_text(json.dumps(labels))
    LABELS.write_text(json.dumps(labels))
    print("labels:", len(labels), "images")


class Frames(Dataset):
    def __init__(self, paths: list[Path], labels: dict, augment: bool) -> None:
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        img = cv2.resize(cv2.imread(str(p)), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        boxes = np.array([b[:4] for b in self.labels[str(p.relative_to(DATA))]], np.float32).reshape(-1, 4)
        boxes *= [IN_W, IN_H, IN_W, IN_H]
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1]
                boxes = np.stack([IN_W - boxes[:, 2], boxes[:, 1], IN_W - boxes[:, 0], boxes[:, 3]], axis=1)
            img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        centre, size, mask = person_targets(boxes)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return x, torch.from_numpy(centre), torch.from_numpy(size), torch.from_numpy(mask)


def agreement(pred: list[list[np.ndarray]], truth: list[list[np.ndarray]], min_h: float = 0.0) -> dict:
    """Greedy IoU >= 0.5 matching per image; boxes in 1280x720 pixels; ``min_h`` keeps tall persons only."""
    tp = fp = fn = 0
    for ps, ts in zip(pred, truth, strict=True):
        # A tall teacher box may be matched by a slightly shorter prediction: keep predictions down to 0.8x.
        ps = [b for b in ps if b[3] - b[1] >= 0.8 * min_h]
        ts = [b for b in ts if b[3] - b[1] >= min_h]
        used: set[int] = set()
        for b in ps:
            best = max(((k, _iou(b, t)) for k, t in enumerate(ts) if k not in used), key=lambda kv: kv[1],
                       default=(None, 0.0))
            if best[0] is not None and best[1] >= 0.5:
                used.add(best[0])
                tp += 1
            elif b[3] - b[1] >= min_h:  # an unmatched shorter prediction is not counted against a tall-only test
                fp += 1
        fn += len(ts) - len(used)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "boxes": tp + fn}


def evaluate(net: torch.nn.Module, paths: list[Path], labels: dict, dev: torch.device) -> dict:
    preds, truths = [], []
    net.eval()
    scale = np.array([1280 / IN_W, 720 / IN_H] * 2)
    with torch.inference_mode():
        for k in range(0, len(paths), 16):
            batch = paths[k:k + 16]
            x = torch.stack([torch.from_numpy(cv2.resize(cv2.imread(str(p)), (IN_W, IN_H),
                                                         interpolation=cv2.INTER_AREA)).permute(2, 0, 1)
                             for p in batch]).float().div_(255.0)
            out = net(x.to(dev)).float().cpu().numpy()
            for o, p in zip(out, batch, strict=True):
                preds.append([b * scale for b, _ in decode(o)])
                truths.append([np.array(b[:4]) * [1280, 720, 1280, 720] for b in labels[str(p.relative_to(DATA))]])
    net.train()
    return {"all": agreement(preds, truths), "players": agreement(preds, truths, min_h=60.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--test", nargs="*")
    args = ap.parse_args()
    if args.labels:
        make_labels()
        return
    labels = json.loads(LABELS.read_text())
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.test:
        ckpt = torch.load(args.test[0], map_location="cpu", weights_only=True)
        net = PersonUNet()
        net.load_state_dict(ckpt["model"])
        print("games 8-10 vs teacher", json.dumps(evaluate(net.to(dev), tracknet_frames(range(8, 11)), labels, dev)))
        return
    torch.manual_seed(0)
    train_paths = tracknet_frames(range(1, 6), step=2) + court_frames()
    sel_paths = tracknet_frames(range(6, 8), step=8)
    print(f"train {len(train_paths)} images, selection {len(sel_paths)}", flush=True)
    train = DataLoader(Frames(train_paths, labels, True), batch_size=args.batch, shuffle=True, num_workers=2,
                       persistent_workers=True, drop_last=True)
    net = PersonUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train),
                                                pct_start=0.1)
    best, step, history = -1.0, 0, []
    for epoch in range(args.epochs):
        t1, losses = time.time(), []
        for x, c, s, m in train:
            loss = person_loss(net(x.to(dev)), c.to(dev), s.to(dev), m.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"{step} steps, {(time.time() - t1) / step:.3f} s/step, {len(train)} steps/epoch, "
                      f"loss {np.mean(losses):.4f}  selection {json.dumps(evaluate(net, sel_paths, labels, dev))}")
                return
        scores = evaluate(net, sel_paths, labels, dev)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **scores})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.4f}  selection {json.dumps(scores)}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if scores["players"]["f1"] > best:
            best = scores["players"]["f1"]
            OUT.parent.mkdir(exist_ok=True)
            torch.save({"arch": ARCH, "model": net.state_dict(), "epoch": epoch + 1, "selection": scores,
                        "history": history, "teacher": "Ultralytics YOLO26-s, imgsz 1280, conf 0.3",
                        "trained_on": "TrackNet games 1-5 (every 2nd frame) + TennisCourtDetector frames",
                        "selected_on": "TrackNet games 6-7 (player F1 vs teacher)", "args": vars(args)}, OUT)
    print("best selection player F1", best, "->", OUT)


if __name__ == "__main__":
    main()
