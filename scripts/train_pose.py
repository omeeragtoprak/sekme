"""Distil the SEKME pose network (models/pose_unet.py) from YOLO26-s-pose on player crops.

    .venv/bin/python scripts/train_pose.py --labels      # crops + teacher keypoints (needs teacher_boxes.json)
    .venv/bin/python scripts/train_pose.py [--epochs 8]  # train
    .venv/bin/python scripts/train_pose.py --test models/sekme_pose.pt

Crops are cut exactly as at run time (models/pose_unet.crop_box) around the tall persons (> 60 px at
720p) of the teacher person boxes (scripts/train_players.py --labels). The teacher, YOLO26-s-pose at
imgsz 320 as in perception/pose.py, labels the crop; keypoints it is less than 0.5 sure of are left out.
Split by match: TrackNet games 1-5 + TennisCourtDetector frames train, games 6-7 choose the epoch,
games 8-10 test. Metric: share of teacher keypoints found within 10 % of the person's height (PCK@0.1).
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

from courtvision.models.pose_unet import (  # noqa: E402
    ARCH,
    FLIP,  # noqa: E402
    IN_H,
    IN_W,
    KEYPOINTS,
    PoseUNet,
    crop_box,
    pose_targets,
)
from courtvision.models.unet import wbce_loss  # noqa: E402
from courtvision.perception.court import heatmap_peak  # noqa: E402

DATA = ROOT / "data"
BOXES = DATA / "players" / "teacher_boxes.json"
CROPS = DATA / "pose" / "crops"
LABELS = DATA / "pose" / "teacher_pose.json"
OUT = ROOT / "models" / "sekme_pose.pt"
TEACHER = ROOT / "models" / "yolo26s-pose.pt"
MIN_H_720 = 60.0
MIN_CONF = 0.5


def split_of(key: str) -> str:
    if key.startswith("tracknet/"):
        g = int(key.split("/")[2][4:])
        return "train" if g <= 5 else "select" if g <= 7 else "test"
    return "train"


def make_labels(tracknet_step: int = 4) -> None:
    from ultralytics import YOLO

    model = YOLO(str(TEACHER))
    boxes = json.loads(BOXES.read_text())
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    CROPS.mkdir(parents=True, exist_ok=True)
    keys = sorted(boxes)
    keys = [k for k in keys if not k.startswith("tracknet/") or int(Path(k).stem) % tracknet_step == 0]
    t0, n_new = time.time(), 0
    for i, key in enumerate(keys):
        img = None
        for j, b in enumerate(boxes[key]):
            cid = f"{key.replace('/', '__')}__{j}"
            if cid in labels:
                continue
            if img is None:
                img = cv2.imread(str(DATA / key))
            h, w = img.shape[:2]
            box = np.array(b[:4]) * [w, h, w, h]
            if (box[3] - box[1]) * 720 / h < MIN_H_720:
                continue
            crop = crop_box(box, w, h)
            if crop is None:
                continue
            X1, Y1, X2, Y2 = crop
            res = model.predict(img[Y1:Y2, X1:X2], imgsz=320, verbose=False, device="mps", half=True)[0]
            if res.keypoints is None or len(res.keypoints) == 0 or res.boxes is None:
                continue
            bb = res.boxes.xyxy.cpu().numpy()
            cx, cy = (box[0] + box[2]) / 2 - X1, (box[1] + box[3]) / 2 - Y1
            k = int(np.argmin(np.hypot((bb[:, 0] + bb[:, 2]) / 2 - cx, (bb[:, 1] + bb[:, 3]) / 2 - cy)))
            kp = res.keypoints.xy[k].cpu().numpy() / [X2 - X1, Y2 - Y1]
            conf = res.keypoints.conf[k].cpu().numpy()
            cv2.imwrite(str(CROPS / f"{cid}.jpg"),
                        cv2.resize(img[Y1:Y2, X1:X2], (IN_W, IN_H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            labels[cid] = {"src": key, "kps": kp.round(4).tolist(), "conf": conf.round(3).tolist(),
                           "h720": float((box[3] - box[1]) * 720 / h)}
            n_new += 1
        if i % 2000 == 0:
            print(f"{i}/{len(keys)} images, {n_new} new crops, {time.time() - t0:.0f} s", flush=True)
            LABELS.write_text(json.dumps(labels))
    LABELS.write_text(json.dumps(labels))
    print("crops:", len(labels))


def points(rec: dict) -> list[tuple[float, float] | None]:
    return [(x * IN_W, y * IN_H) if c >= MIN_CONF else None for (x, y), c in zip(rec["kps"], rec["conf"], strict=True)]


class Crops(Dataset):
    def __init__(self, items: list[tuple[str, dict]], augment: bool) -> None:
        self.items, self.augment = items, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        cid, rec = self.items[i]
        img = cv2.imread(str(CROPS / f"{cid}.jpg"))
        pts = points(rec)
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img = img[:, ::-1]
                mirrored = [None if p is None else (IN_W - 1 - p[0], p[1]) for p in pts]
                pts = [mirrored[FLIP[k]] for k in range(KEYPOINTS)]
            img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        truth = torch.tensor([[-1.0, -1.0] if p is None else [p[0], p[1]] for p in pts])
        return x, torch.from_numpy(pose_targets(pts)), truth


def pck(net: torch.nn.Module, loader: DataLoader, dev: torch.device, frac: float = 0.1) -> dict:
    """Share of teacher keypoints our network places within ``frac`` of the person height (in crop pixels)."""
    hit = total = predicted = 0
    dists = []
    person_h = IN_H / 1.45  # the crop is the box height plus 45 % (see crop_box)
    net.eval()
    with torch.inference_mode():
        for x, _, t in loader:
            hms = torch.sigmoid(net(x.to(dev))).float().cpu().numpy()
            for b in range(len(hms)):
                for k in range(KEYPOINTS):
                    if t[b, k, 0] < 0:
                        continue
                    total += 1
                    pk = heatmap_peak(hms[b, k], 0.3)
                    if pk is None:
                        continue
                    predicted += 1
                    d = float(np.hypot(pk[0] - t[b, k, 0], pk[1] - t[b, k, 1]))
                    dists.append(d / person_h)
                    hit += d <= frac * person_h
    net.train()
    return {"pck": round(hit / max(total, 1), 4), "found": round(predicted / max(total, 1), 4),
            "median_dist_of_height": round(float(np.median(dists)), 4) if dists else None, "keypoints": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--test", nargs="*")
    args = ap.parse_args()
    if args.labels:
        make_labels()
        return
    labels = json.loads(LABELS.read_text())
    items = {s: [(k, v) for k, v in labels.items() if split_of(v["src"]) == s] for s in ("train", "select", "test")}
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.test:
        net = PoseUNet()
        net.load_state_dict(torch.load(args.test[0], map_location="cpu", weights_only=True)["model"])
        loader = DataLoader(Crops(items["test"], False), batch_size=64, num_workers=2)
        print("games 8-10 vs teacher", json.dumps(pck(net.to(dev), loader, dev)))
        return
    torch.manual_seed(0)
    print({s: len(v) for s, v in items.items()}, flush=True)
    train = DataLoader(Crops(items["train"], True), batch_size=args.batch, shuffle=True, num_workers=2,
                       persistent_workers=True, drop_last=True)
    sel = DataLoader(Crops(items["select"], False), batch_size=64, num_workers=2, persistent_workers=True)
    net = PoseUNet().to(dev)
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
                      f"loss {np.mean(losses):.4f}  selection {json.dumps(pck(net, sel, dev))}")
                return
        scores = pck(net, sel, dev)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **scores})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.4f}  selection {json.dumps(scores)}  "
              f"{time.time() - t1:.0f} s", flush=True)
        if scores["pck"] > best:
            best = scores["pck"]
            OUT.parent.mkdir(exist_ok=True)
            torch.save({"arch": ARCH, "model": net.state_dict(), "epoch": epoch + 1, "selection": scores,
                        "history": history, "teacher": "Ultralytics YOLO26-s-pose on the crop, imgsz 320",
                        "trained_on": "player crops from TrackNet games 1-5 + TennisCourtDetector frames",
                        "selected_on": "TrackNet games 6-7 (PCK@0.1 vs teacher)", "args": vars(args)}, OUT)
    print("best selection PCK", best, "->", OUT)


if __name__ == "__main__":
    main()
