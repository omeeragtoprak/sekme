"""Body pose of the hitting player at each racket contact (YOLO26-pose, COCO 17 keypoints).

Only the hitter's box at the contact frame is processed (one crop per shot), so the cost is a few
seconds per clip. Keypoints are anatomical: index 10 is the person's right wrist whichever way they
face the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12


@dataclass(frozen=True)
class Pose:
    kps: np.ndarray  # (17, 2) image pixels
    conf: np.ndarray  # (17,)

    def point(self, i: int, min_conf: float = 0.3) -> np.ndarray | None:
        return self.kps[i] if self.conf[i] >= min_conf else None

    def mid(self, a: int, b: int, min_conf: float = 0.3) -> np.ndarray | None:
        pa, pb = self.point(a, min_conf), self.point(b, min_conf)
        if pa is None or pb is None:
            return pa if pb is None else pb
        return (pa + pb) / 2


class PoseEstimator:
    def __init__(self, weights: str | Path, device: str = "cpu", imgsz: int = 320) -> None:
        from ultralytics import YOLO  # AGPL-3.0, see docs/LICENSES.md

        self.model = YOLO(str(weights))
        self.device, self.imgsz = device, imgsz

    def estimate(self, frame: np.ndarray, box: np.ndarray) -> Pose | None:
        """Pose of the person in ``box`` (x1, y1, x2, y2); the crop adds room for an extended arm."""
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        X1, Y1 = int(max(0, x1 - 0.6 * w)), int(max(0, y1 - 0.35 * h))
        X2, Y2 = int(min(W, x2 + 0.6 * w)), int(min(H, y2 + 0.1 * h))
        if X2 - X1 < 8 or Y2 - Y1 < 8:
            return None
        res = self.model.predict(frame[Y1:Y2, X1:X2], imgsz=self.imgsz, verbose=False, device=self.device)[0]
        if res.keypoints is None or len(res.keypoints) == 0 or res.boxes is None:
            return None
        # The detection closest to the crop's centre is the tracked player (not a ball kid or umpire).
        b = res.boxes.xyxy.cpu().numpy()
        cx, cy = (x1 + x2) / 2 - X1, (y1 + y2) / 2 - Y1
        j = int(np.argmin(np.hypot((b[:, 0] + b[:, 2]) / 2 - cx, (b[:, 1] + b[:, 3]) / 2 - cy)))
        kps = res.keypoints.xy[j].cpu().numpy() + np.array([X1, Y1])
        conf = res.keypoints.conf[j].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)
        return Pose(kps.astype(np.float32), conf.astype(np.float32))


def poses_at(video: str | Path, requests: list[tuple[int, np.ndarray]], estimator: PoseEstimator) -> list[Pose | None]:
    """Poses for (frame, box) requests, reading only the frames that are needed."""
    out: list[Pose | None] = [None] * len(requests)
    order = sorted(range(len(requests)), key=lambda k: requests[k][0])
    cap = cv2.VideoCapture(str(video))
    try:
        for k in order:
            frame, box = requests[k]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if ok and np.all(np.isfinite(box)):
                out[k] = estimator.estimate(img, box)
    finally:
        cap.release()
    return out
