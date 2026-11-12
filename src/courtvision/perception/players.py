"""Player detection/tracking and selection of the two players on court.

The detector backend is Ultralytics YOLO (AGPL-3.0); it sits behind this
module so an Apache-licensed detector can replace it without touching the
rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from courtvision.calib.camera import apply_h
from courtvision.perception.court import FrameCalibration


@dataclass
class PersonDet:
    track_id: int
    box: np.ndarray  # x1, y1, x2, y2
    conf: float

    @property
    def foot(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])


class YoloPersonTracker:
    def __init__(self, weights: str, device: str, imgsz: int = 1280, conf: float = 0.2) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frame: np.ndarray) -> list[PersonDet]:
        # fp16 on GPU/MPS: 43 -> 34 ms/frame at 1080p, the same boxes (median shift 0.13 px over 1221 boxes).
        res = self.model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", imgsz=self.imgsz,
                               conf=self.conf, device=self.device, verbose=False, half=self.device != "cpu")[0]
        out: list[PersonDet] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else -np.arange(1, len(boxes) + 1)
        for b, c, t in zip(boxes, confs, ids, strict=True):
            out.append(PersonDet(int(t), b, float(c)))
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Greedy frame-to-frame association by box overlap; a track survives ``max_age`` frames unseen."""

    def __init__(self, min_iou: float = 0.3, max_age: int = 15) -> None:
        self.min_iou, self.max_age = min_iou, max_age
        self.tracks: dict[int, tuple[np.ndarray, int]] = {}  # id -> (last box, frames since seen)
        self.next_id = 1

    def update(self, boxes: list[np.ndarray]) -> list[int]:
        pairs = sorted(((_iou(tb, b), tid, k) for tid, (tb, _) in self.tracks.items() for k, b in enumerate(boxes)),
                       reverse=True)
        ids: list[int | None] = [None] * len(boxes)
        used: set[int] = set()
        for iou, tid, k in pairs:
            if iou < self.min_iou:
                break
            if tid in used or ids[k] is not None:
                continue
            ids[k] = tid
            used.add(tid)
        for k in range(len(boxes)):
            if ids[k] is None:
                ids[k] = self.next_id
                self.next_id += 1
        seen = dict(zip(ids, boxes, strict=True))
        kept = {tid: (box, age + 1) for tid, (box, age) in self.tracks.items()
                if tid not in seen and age + 1 <= self.max_age}
        self.tracks = {**kept, **{tid: (box, 0) for tid, box in seen.items()}}
        return [int(i) for i in ids]


class SekmePersonTracker:
    """Our person network (models/person_unet.py) plus the IoU tracker; same output as YoloPersonTracker."""

    def __init__(self, weights: str, device: str, threshold: float = 0.3) -> None:
        import torch

        from courtvision.models.person_unet import PersonUNet

        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.net = PersonUNet()
        self.net.load_state_dict(ckpt["model"], strict=True)
        self.device = torch.device(device)
        self.net.eval().to(self.device)
        self.threshold = threshold
        self.tracker = IouTracker()

    def reset(self) -> None:
        """Forget all tracks (a camera cut: nobody on screen is the person who was there before)."""
        self.tracker = IouTracker()

    def __call__(self, frame: np.ndarray) -> list[PersonDet]:
        import cv2
        import torch

        from courtvision.models.person_unet import IN_H, IN_W, decode

        h0, w0 = frame.shape[:2]
        x = cv2.resize(frame, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            t = torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float().div_(255.0)
            out = self.net(t)[0].float().cpu().numpy()
        dets = decode(out, self.threshold)
        scale = np.array([w0 / IN_W, h0 / IN_H, w0 / IN_W, h0 / IN_H])
        boxes = [b * scale for b, _ in dets]
        ids = self.tracker.update(boxes)
        return [PersonDet(i, b, s) for i, b, (_, s) in zip(ids, boxes, dets, strict=True)]


def person_tracker(weights: str, device: str, imgsz: int = 1280):
    """Our person network when the checkpoint holds it, Ultralytics YOLO otherwise."""
    import torch

    from courtvision.models.person_unet import ARCH

    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  an Ultralytics .pt needs its own classes to unpickle
        ckpt = None
    if isinstance(ckpt, dict) and ckpt.get("arch") == ARCH:
        return SekmePersonTracker(weights, device)
    return YoloPersonTracker(weights, device, imgsz=imgsz)


# Area where a player can plausibly stand during a point (metres, world frame).
PLAY_X = 7.5
PLAY_Y_MIN, PLAY_Y_MAX = 0.8, 17.5
MAX_PLAYER_SPEED = 11.0  # m/s, above elite sprint speed


@dataclass
class PlayerTrack:
    side: str  # "near" | "far"
    boxes: np.ndarray  # (N, 4) NaN when missing
    world: np.ndarray  # (N, 2) foot position on court, NaN when missing/uncalibrated
    track_ids: list[int] = field(default_factory=list)


def _plausible(w: np.ndarray, d: PersonDet, last: np.ndarray | None, frames: int, fps: float,
               ref_h: float | None) -> bool:
    """Reachable from the last position at human speed (plus box jitter) and of the player's size.

    The size gate rejects ball kids and line judges who stand close to a player.
    """
    if last is not None and np.linalg.norm(w - last) > MAX_PLAYER_SPEED * frames / fps + 0.35:
        return False
    return ref_h is None or 0.6 * ref_h <= d.box[3] - d.box[1] <= 1.6 * ref_h


def select_players(dets: list[list[PersonDet]], calib: list[FrameCalibration],
                   fps: float = 30.0) -> dict[str, PlayerTrack]:
    """Choose the near and far player per frame: dominant in-court track, then continuity fallback."""
    n = len(dets)
    feet_world: list[list[np.ndarray | None]] = []
    for i in range(n):
        c = calib[i]
        row: list[np.ndarray | None] = []
        for d in dets[i]:
            row.append(apply_h(c.H_inv, d.foot) if c.valid and c.H_inv is not None else None)
        feet_world.append(row)

    def in_half(w: np.ndarray | None, side: str) -> bool:
        if w is None or abs(w[0]) > PLAY_X:
            return False
        y = -w[1] if side == "near" else w[1]
        return PLAY_Y_MIN <= y <= PLAY_Y_MAX

    result: dict[str, PlayerTrack] = {}
    for side in ("near", "far"):
        presence: dict[int, float] = {}
        for i in range(n):
            for d, w in zip(dets[i], feet_world[i], strict=True):
                if in_half(w, side) and d.track_id >= 0:
                    presence[d.track_id] = presence.get(d.track_id, 0.0) + d.conf * (d.box[3] - d.box[1])
        main_ids = sorted(presence, key=lambda k: -presence[k])[:1]
        heights = [d.box[3] - d.box[1] for i in range(n) for d in dets[i] if d.track_id in main_ids]
        ref_h = float(np.median(heights)) if heights else None
        boxes = np.full((n, 4), np.nan)
        world = np.full((n, 2), np.nan)
        used: list[int] = []
        last: np.ndarray | None = None
        last_i = -1
        for i in range(n):
            if last is not None and i - last_i > fps:
                last = None  # lost for over a second (cut, close-up): re-seed instead of chaining
            pick = None
            for d, w in zip(dets[i], feet_world[i], strict=True):
                if d.track_id in main_ids and in_half(w, side) and _plausible(w, d, last, i - last_i, fps, ref_h):
                    pick = (d, w)
            if pick is None:  # identity switch / missed association: nearest plausible in-half detection
                best = None
                for d, w in zip(dets[i], feet_world[i], strict=True):
                    if w is None or not in_half(w, side) or last is None:
                        continue
                    if not _plausible(w, d, last, i - last_i, fps, ref_h):
                        continue
                    dist = float(np.linalg.norm(w - last))
                    if best is None or dist < best[0]:
                        best = (dist, d, w)
                if best is not None:
                    pick = (best[1], best[2])
            if pick is None and last is None:  # re-seed: the most player-like person in this half
                cands = [(d.conf * (d.box[3] - d.box[1]) / (1.0 + abs(w[0]) / 4.0), d, w)
                         for d, w in zip(dets[i], feet_world[i], strict=True)
                         if w is not None and in_half(w, side) and _plausible(w, d, None, 0, fps, ref_h)]
                if cands:
                    _, d, w = max(cands, key=lambda c: c[0])
                    pick = (d, w)
            if pick is not None:
                boxes[i] = pick[0].box
                world[i] = pick[1]
                last, last_i = pick[1], i
                if pick[0].track_id not in used:
                    used.append(pick[0].track_id)
        result[side] = PlayerTrack(side, boxes, world, used)
    return result
