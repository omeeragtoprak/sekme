from __future__ import annotations

import cv2
import numpy as np

from courtvision.calib.camera import apply_h
from courtvision.calib.line_refine import line_support, refine_homography
from courtvision.geometry.court import CourtModel
from courtvision.perception.court import heatmap_peak


def test_heatmap_peak_finds_centre_of_saturated_disk() -> None:
    # Regression: the court model's blobs saturate at 1.0; argmax alone returned the
    # plateau's top-left pixel (~one radius off), which shifted every keypoint up-left.
    hm = np.zeros((360, 640), dtype=np.float32)
    cv2.circle(hm, (300, 200), 20, 1.0, -1)
    hm = cv2.GaussianBlur(hm, (0, 0), 3)
    hm = np.clip(hm * 1.6, 0, 1)
    x, y, _ = heatmap_peak(hm, 0.5)
    assert abs(x - 300) < 0.5 and abs(y - 200) < 0.5


def _render_court(cam_H: np.ndarray, court: CourtModel, size=(1920, 1080)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), (120, 80, 40), dtype=np.uint8)  # dark blue court
    for _, a, b in court.line_segments():
        ts = np.linspace(0, 1, 400)
        pts = apply_h(cam_H, a[None] + ts[:, None] * (b - a)[None])
        cv2.polylines(img, [np.round(pts * 8).astype(np.int32)], False, (235, 235, 235), 5, cv2.LINE_AA, shift=3)
    rng = np.random.default_rng(1)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def test_line_refinement_recovers_true_homography(broadcast_camera) -> None:
    court = CourtModel()
    H_true = broadcast_camera.ground_homography()
    img = _render_court(H_true, court)
    # Perturb the homography by a few pixels, as a coarse keypoint fit would.
    shift = np.array([[1, 0, 4.0], [0, 1, -5.0], [0, 0, 1]])
    H0 = shift @ H_true
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    assert line_support(gray, H0, court) < 0.5

    ref = refine_homography(img, H0, court)
    kp = court.keypoints()
    err = np.linalg.norm(apply_h(ref.H, kp) - apply_h(H_true, kp), axis=1)
    assert err.max() < 1.0, err
    assert ref.support > 0.9
