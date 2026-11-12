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

from courtvision.models.pose_unet import crop_box

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
        crop = crop_box(box, frame.shape[1], frame.shape[0])
        if crop is None:
            return None
        X1, Y1, X2, Y2 = crop
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


class SekmePoseEstimator:
    """Our pose network (models/pose_unet.py) on the same crop; same output as PoseEstimator."""

    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        import torch

        from courtvision.models.pose_unet import PoseUNet

        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.net = PoseUNet()
        self.net.load_state_dict(ckpt["model"], strict=True)
        self.device = torch.device(device)
        self.net.eval().to(self.device)

    def estimate(self, frame: np.ndarray, box: np.ndarray) -> Pose | None:
        import torch

        from courtvision.models.pose_unet import IN_H, IN_W, KEYPOINTS
        from courtvision.perception.court import heatmap_peak

        crop = crop_box(box, frame.shape[1], frame.shape[0])
        if crop is None:
            return None
        X1, Y1, X2, Y2 = crop
        img = cv2.resize(frame[Y1:Y2, X1:X2], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float().div_(255.0)
            hms = torch.sigmoid(self.net(x)[0]).float().cpu().numpy()
        kps, conf = np.zeros((KEYPOINTS, 2), np.float32), np.zeros(KEYPOINTS, np.float32)
        sx, sy = (X2 - X1) / IN_W, (Y2 - Y1) / IN_H
        for k in range(KEYPOINTS):
            pk = heatmap_peak(hms[k], 0.05)
            if pk is not None:
                kps[k] = X1 + (pk[0] + 0.5) * sx - 0.5, Y1 + (pk[1] + 0.5) * sy - 0.5
                conf[k] = pk[2]
        return Pose(kps, conf)


def pose_estimator(weights: str | Path, device: str = "cpu"):
    """Our pose network when the checkpoint holds it, YOLO26-pose otherwise."""
    import torch

    from courtvision.models.pose_unet import ARCH

    try:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  an Ultralytics .pt needs its own classes to unpickle
        ckpt = None
    if isinstance(ckpt, dict) and ckpt.get("arch") == ARCH:
        return SekmePoseEstimator(weights, device)
    return PoseEstimator(weights, device)


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
