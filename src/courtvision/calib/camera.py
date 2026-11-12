"""Court calibration: homography fitting and a full pinhole camera from it.

A homography alone maps the ground plane to the image. Anything above the
ground (the ball centre sits 3.35 cm up even at contact; a struck ball is
metres up) needs a real camera, so we recover focal length, rotation and
translation from the plane homography (Zhang's constraints, square pixels,
principal point at the image centre).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Project world points (..., 3) to pixels (..., 2)."""
        p = np.asarray(pts, dtype=np.float64)
        cam = p @ self.R.T + self.t
        uvw = cam @ self.K.T
        return uvw[..., :2] / uvw[..., 2:3]

    def depth(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        return (p @ self.R.T + self.t)[..., 2]

    def ray(self, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray (origin, unit direction) through pixel(s) (..., 2)."""
        px = np.asarray(px, dtype=np.float64)
        h = np.concatenate([px, np.ones(px.shape[:-1] + (1,))], axis=-1)
        d = (h @ np.linalg.inv(self.K).T) @ self.R  # rows: R^T K^-1 [u v 1]
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(self.center, d.shape), d

    def backproject_to_height(self, px: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height ``z``; returns (..., 3)."""
        o, d = self.ray(px)
        s = (np.asarray(z) - o[..., 2]) / d[..., 2]
        return o + d * s[..., None]

    def height_above(self, px: np.ndarray, ground_xy: np.ndarray) -> float:
        """Height of the pixel ray where it passes closest to the vertical line through ``ground_xy``."""
        o, d = self.ray(np.asarray(px, dtype=np.float64))
        g = np.asarray(ground_xy, dtype=np.float64)
        s = -np.dot(o[:2] - g, d[:2]) / np.dot(d[:2], d[:2])
        return float(o[2] + s * d[2])

    def ground_homography(self) -> np.ndarray:
        """Homography world (x, y, 1) on z=0 -> image implied by this camera."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]


def _shift(cx: float, cy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])


def estimate_focal(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Focal length (px) from a plane->image homography, or None if ill-posed.

    Uses r1.r2 = 0 and |r1| = |r2| with K = diag(f, f, 1) after moving the
    principal point to the origin; both constraints are linear in w = 1/f^2.
    """
    Hs = _shift(cx, cy) @ H
    Hs = Hs / np.linalg.norm(Hs)
    h1, h2 = Hs[:, 0], Hs[:, 1]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denom = float(a @ a)
    if denom < 1e-18:
        return None
    w = -float(a @ b) / denom
    if not np.isfinite(w) or w <= 0:
        return None
    return float(1.0 / np.sqrt(w))


def camera_from_homography(
    H: np.ndarray, image_size: tuple[int, int], focal: float | None = None
) -> Camera | None:
    """Decompose a ground-plane homography into a pinhole camera.

    ``image_size`` is (width, height). Returns None when the focal length
    cannot be recovered (e.g. a near fronto-parallel view) or the pose is
    not physical.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    f = focal if focal is not None else estimate_focal(H, cx, cy)
    if f is None or not (0.2 * w < f < 20.0 * w):
        return None
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    if t[2] < 0:  # the court origin must lie in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    cam = Camera(K=K, R=R, t=t)
    if cam.center[2] <= 0:  # camera below the court plane is not physical
        return None
    return cam


@dataclass(frozen=True)
class HomographyFit:
    H: np.ndarray  # world plane -> image
    inliers: np.ndarray  # bool (N,)
    rmse_px: float
    n_inliers: int


def fit_homography(img_pts: np.ndarray, world_xy: np.ndarray, ransac_px: float = 8.0) -> HomographyFit | None:
    """Robust ground homography from keypoints; rows of ``img_pts`` may be NaN (missing)."""
    ok = np.all(np.isfinite(img_pts), axis=1)
    if ok.sum() < 4:
        return None
    src = world_xy[ok].astype(np.float64)
    dst = img_pts[ok].astype(np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px, maxIters=2000, confidence=0.999)
    if H is None:
        return None
    inl_sub = mask.ravel().astype(bool)
    if inl_sub.sum() < 4:
        return None
    H, _ = cv2.findHomography(src[inl_sub], dst[inl_sub], 0)  # least-squares refit on inliers
    if H is None:
        return None
    H = H / H[2, 2]
    proj = cv2.perspectiveTransform(src[inl_sub][None], H)[0]
    rmse = float(np.sqrt(np.mean(np.sum((proj - dst[inl_sub]) ** 2, axis=1))))
    inliers = np.zeros(len(img_pts), dtype=bool)
    inliers[np.flatnonzero(ok)[inl_sub]] = True
    return HomographyFit(H=H, inliers=inliers, rmse_px=rmse, n_inliers=int(inl_sub.sum()))


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to points (..., 2)."""
    p = np.asarray(pts, dtype=np.float64)
    h = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1) @ H.T
    return h[..., :2] / h[..., 2:3]
