"""Court keypoint detection, shot-cut detection and per-frame calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from courtvision.calib.camera import Camera, apply_h, camera_from_homography, fit_homography
from courtvision.calib.line_refine import refine_homography
from courtvision.geometry.court import CourtModel
from courtvision.models.court_net import CourtKeypointNet
from courtvision.models.court_unet import ARCH as COURT_UNET_ARCH
from courtvision.models.court_unet import CourtUNet


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def heatmap_peak(hm: np.ndarray, min_peak: float) -> tuple[float, float, float] | None:
    """Centre (x, y, score) of the blob holding the heatmap maximum.

    The court model's blobs saturate at 1.0 over a disk ~20 px wide, so the
    argmax alone lands on the plateau's first (top-left) pixel; the centroid
    of the thresholded blob recovers the disk centre.
    """
    iy, ix = np.unravel_index(int(np.argmax(hm)), hm.shape)
    peak = float(hm[iy, ix])
    if peak < min_peak:
        return None
    mask = (hm >= 0.5 * peak).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    ys, xs = np.nonzero(labels == labels[iy, ix])
    w = hm[ys, xs]
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), peak


def load_court_net(weights: str) -> tuple[torch.nn.Module, str]:
    """Our court U-Net (checkpoint with ``arch``) or the TennisCourtDetector network (bare state dict)."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    if ckpt.get("arch") == COURT_UNET_ARCH:
        net: torch.nn.Module = CourtUNet()
        net.load_state_dict(ckpt["model"], strict=True)
        return net, COURT_UNET_ARCH
    net = CourtKeypointNet(out_channels=15)
    net.load_state_dict(ckpt)
    return net, "tennis-court-detector"


class CourtKeypointDetector:
    def __init__(self, weights: str, device: torch.device, min_peak: float = 0.5) -> None:
        self.device = device
        self.net, self.arch = load_court_net(weights)
        self.net.eval().to(device)
        self.min_peak = min_peak

    @torch.inference_mode()
    def detect_batch(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (B, 14, 2) in original pixels (NaN = not found) and peak scores (B, 14)."""
        W, H = CourtKeypointNet.INPUT_W, CourtKeypointNet.INPUT_H
        h0, w0 = frames[0].shape[:2]
        batch = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        hms = torch.sigmoid(self.net(x)).float().cpu().numpy()
        kps = np.full((len(frames), 14, 2), np.nan)
        scores = np.zeros((len(frames), 14))
        sx, sy = w0 / W, h0 / H
        off = 0.0 if self.arch == COURT_UNET_ARCH else 0.5  # our network was trained on labels / scale
        for b in range(len(frames)):
            for k in range(14):
                pk = heatmap_peak(hms[b, k], self.min_peak)
                if pk is not None:
                    kps[b, k] = (pk[0] + off) * sx - off, (pk[1] + off) * sy - off
                    scores[b, k] = pk[2]
        return kps, scores


def frame_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype(np.float32)


def detect_shot_cuts(hists: list[np.ndarray], threshold: float = 0.55) -> list[int]:
    """Frame indices that start a new shot, from HSV histogram correlation drops."""
    cuts = [0]
    for i in range(1, len(hists)):
        if cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL) < threshold:
            cuts.append(i)
    return cuts


@dataclass
class CourtMeasurement:
    """Per-frame calibration measurement before temporal smoothing."""

    H: np.ndarray | None  # world -> image, line-refined
    support: float  # fraction of court-line samples on a white ridge
    line_rmse_px: float
    from_keyframe: bool  # True when the keypoint network ran on this frame
    reason: str = ""


class CourtCalibrator:
    """Keypoint network on keyframes / cuts, line-ridge tracking in between.

    The heavy network only runs every ``keyframe_interval`` frames, on shot
    cuts, or when line support drops (camera move). Every accepted frame is
    refined against the painted lines at full resolution.
    """

    def __init__(self, detector: CourtKeypointDetector, court: CourtModel, keyframe_interval: int = 15,
                 min_support: float = 0.55) -> None:
        self.detector = detector
        self.court = court
        self.interval = keyframe_interval
        self.min_support = min_support
        self._last_H: np.ndarray | None = None
        self._last_key = -10**9

    def __call__(self, idx: int, frame: np.ndarray, is_cut: bool) -> CourtMeasurement:
        if is_cut:
            self._last_H = None
        if self._last_H is not None and idx - self._last_key < self.interval:
            ref = refine_homography(frame, self._last_H, self.court)
            if ref.support >= self.min_support:
                self._last_H = ref.H
                return CourtMeasurement(ref.H, ref.support, ref.rmse_px, False)
        kps, _ = self.detector.detect_batch([frame])
        self._last_key = idx
        fit = fit_homography(kps[0], self.court.keypoints(), ransac_px=0.006 * frame.shape[1])
        if fit is None or fit.n_inliers < 6:
            self._last_H = None
            return CourtMeasurement(None, 0.0, float("nan"), True, "anahtar nokta sayısı homografi için yetersiz")
        ref = refine_homography(frame, fit.H, self.court)
        if ref.support < self.min_support:
            self._last_H = None
            return CourtMeasurement(ref.H, ref.support, ref.rmse_px, True,
                                    f"çizgi desteği {ref.support:.2f} < {self.min_support}")
        self._last_H = ref.H
        return CourtMeasurement(ref.H, ref.support, ref.rmse_px, True)


@dataclass
class FrameCalibration:
    valid: bool
    H: np.ndarray | None = None  # world ground plane -> image
    H_inv: np.ndarray | None = None  # image -> world ground plane
    camera: Camera | None = None
    support: float = 0.0
    line_rmse_px: float = float("nan")
    reason: str = ""


def calibrate_sequence(
    meas: list[CourtMeasurement],
    cuts: list[int],
    image_size: tuple[int, int],
    court: CourtModel,
    smooth_window: int = 7,
    min_support: float = 0.55,
) -> list[FrameCalibration]:
    """Temporal median of projected keypoints inside each shot, refit, and gate.

    The gate is the honesty mechanism: frames whose court fit is not backed
    by the painted lines are invalid and downstream modules refuse to
    measure on them.
    """
    n = len(meas)
    world = court.keypoints()
    proj = np.full((n, 14, 2), np.nan)
    for i, m in enumerate(meas):
        if m.H is not None and m.support >= min_support:
            proj[i] = apply_h(m.H, world)
    bounds = [*cuts, n]
    half = smooth_window // 2
    out: list[FrameCalibration] = []
    for s, e in zip(bounds[:-1], bounds[1:], strict=True):
        for i in range(s, e):
            m = meas[i]
            if not np.isfinite(proj[i, 0, 0]):
                out.append(FrameCalibration(False, m.H, None, None, m.support, m.line_rmse_px,
                                            m.reason or "kalibre kort yok"))
                continue
            win = proj[max(i - half, s):min(i + half + 1, e)]
            win = win[np.isfinite(win[:, 0, 0])]
            fit = fit_homography(np.median(win, axis=0), world, ransac_px=2.0)
            if fit is None:
                out.append(FrameCalibration(False, m.H, None, None, m.support, m.line_rmse_px, "refit failed"))
                continue
            cam = camera_from_homography(fit.H, image_size)
            out.append(FrameCalibration(True, fit.H, np.linalg.inv(fit.H), cam, m.support, m.line_rmse_px))
    return out
