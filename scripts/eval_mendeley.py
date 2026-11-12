"""Human-labelled test of the person and pose networks on the Tennis Player Actions dataset.

    .venv/bin/python scripts/eval_mendeley.py --players models/yolo26s.pt models/sekme_person.pt \\
                                              --pose models/yolo26s-pose.pt models/sekme_pose.pt

Dataset: Wang, Lai, Huang & Lin (2024), Tennis Player Actions Dataset for Human Pose Estimation,
Mendeley Data, doi:10.17632/nv3rpsxhhk.1, CC BY 4.0. 2 000 frames (1280x720) from a camera high behind
the baseline, one player per frame labelled by hand (box + 17 COCO keypoints + neck), in four actions.
Never used for training here. Players: share of labelled players found (IoU >= 0.5); other people in
the frame are not labelled, so precision is not measurable. Pose: the crop is cut around the labelled
box as at run time; PCK@0.1 = share of labelled keypoints within 10 % of the box height.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.perception.players import _iou, person_tracker  # noqa: E402
from courtvision.perception.pose import pose_estimator  # noqa: E402

DATA = ROOT / "data" / "mendeley_pose" / "tpa"


def samples() -> list[tuple[Path, np.ndarray, np.ndarray, str]]:
    out = []
    for ann in sorted((DATA / "annotations").glob("*.json")):
        d = json.loads(ann.read_text())
        images = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            kp = np.array(a["keypoints"], float).reshape(-1, 3)[:17]
            out.append((DATA / "images" / ann.stem / im["file_name"], np.array([x, y, x + w, y + h]), kp, ann.stem))
    return out


def eval_players(weights: str, data: list) -> dict:
    tracker = person_tracker(weights, "mps")
    found = 0
    for path, box, _, _ in data:
        img = cv2.imread(str(path))
        if hasattr(tracker, "reset"):  # ours: unrelated still frames, so no track history
            tracker.reset()
            boxes = [d.box for d in tracker(img)]
        else:  # YOLO: plain detection at the run-time settings (tracking needs consecutive frames)
            r = tracker.model.predict(img, classes=[0], imgsz=tracker.imgsz, conf=tracker.conf, device="mps",
                                      half=True, verbose=False)[0]
            boxes = list(r.boxes.xyxy.cpu().numpy()) if r.boxes is not None else []
        found += any(_iou(b, box) >= 0.5 for b in boxes)
    return {"players_found": round(found / len(data), 4), "frames": len(data)}


def eval_pose(weights: str, data: list) -> dict:
    est = pose_estimator(weights, "mps")
    hit = total = 0
    per_action: dict[str, list[int]] = {}
    for path, box, kp, action in data:
        pose = est.estimate(cv2.imread(str(path)), box)
        h = box[3] - box[1]
        for k in range(17):
            if kp[k, 2] <= 0:
                continue
            total += 1
            ok = pose is not None and pose.conf[k] >= 0.3 and np.hypot(*(pose.kps[k] - kp[k, :2])) <= 0.1 * h
            hit += ok
            per_action.setdefault(action, [0, 0])
            per_action[action][0] += ok
            per_action[action][1] += 1
    return {"pck@0.1": round(hit / max(total, 1), 4), "keypoints": total,
            "by_action": {a: round(v[0] / v[1], 4) for a, v in sorted(per_action.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="*", default=[])
    ap.add_argument("--pose", nargs="*", default=[])
    args = ap.parse_args()
    data = samples()
    for w in args.players:
        print("players", Path(w).name, json.dumps(eval_players(w, data)), flush=True)
    for w in args.pose:
        print("pose", Path(w).name, json.dumps(eval_pose(w, data)), flush=True)


if __name__ == "__main__":
    main()
