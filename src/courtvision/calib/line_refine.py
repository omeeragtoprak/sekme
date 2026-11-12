"""Sub-pixel homography refinement against the painted court lines.

Keypoint heatmaps are coarse (640x360) and say nothing about whether the
fitted court actually sits on the white lines. Here we project every court
line with the current homography, search along the image normal for the
bright line ridge at full resolution, and re-fit the homography by
minimising the perpendicular distance of those ridge points to the
projected lines (robust loss, so players and the net occluding a line do
not drag the fit). The fraction of samples that find a ridge is the *line
support* score used as a calibration gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from courtvision.calib.camera import apply_h
from courtvision.geometry.court import CourtModel


@dataclass(frozen=True)
class LineRefinement:
    H: np.ndarray
    support: float  # fraction of line samples with a clear white ridge (after refinement)
    n_points: int
    rmse_px: float  # perpendicular residual RMS of inlier ridge points


def _line_profile_peak(gray: np.ndarray, p: np.ndarray, n: np.ndarray, reach: int,
                       min_contrast: float) -> float | None:
    """Signed offset along the unit normal ``n`` of the brightest ridge near ``p``, or None."""
    offs = np.arange(-reach, reach + 1, dtype=np.float32)
    xs = (p[0] + offs * n[0]).astype(np.float32)[None]
    ys = (p[1] + offs * n[1]).astype(np.float32)[None]
    prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    i = int(np.argmax(prof))
    if i == 0 or i == len(prof) - 1:
        return None
    # Each side has its own background (boundary lines separate dark court from lighter run-off),
    # and the paint must stand out from both.
    peak = float(prof[i])
    bg_l, bg_r = float(prof[:i].min()), float(prof[i + 1:].min())
    if peak - max(bg_l, bg_r) < min_contrast:
        return None
    # Centre of the ridge at half height (FWHM midpoint): on wide, flat-topped lines the
    # brightest pixel can sit anywhere across the paint, the half-height edges cannot.
    half_l, half_r = 0.5 * (peak + bg_l), 0.5 * (peak + bg_r)
    lo = i
    while lo > 0 and prof[lo - 1] >= half_l:
        lo -= 1
    hi = i
    while hi < len(prof) - 1 and prof[hi + 1] >= half_r:
        hi += 1
    if lo == 0 or hi == len(prof) - 1:
        return None  # ridge not fully inside the search window
    left = offs[lo] - (prof[lo] - half_l) / max(float(prof[lo] - prof[lo - 1]), 1e-6)
    right = offs[hi] + (prof[hi] - half_r) / max(float(prof[hi] - prof[hi + 1]), 1e-6)
    return float(0.5 * (left + right))


def _sample_ridges(gray: np.ndarray, H: np.ndarray, court: CourtModel, per_line: int, reach: int,
                   min_contrast: float, return_offsets: bool = False):
    h, w = gray.shape
    pts: list[np.ndarray] = []
    line_ids: list[int] = []
    offsets: list[float] = []
    tried = 0
    for li, (_, a, b) in enumerate(court.line_segments()):
        ts = np.linspace(0.04, 0.96, per_line)
        world = a[None] + ts[:, None] * (b - a)[None]
        img = apply_h(H, world)
        d = img[-1] - img[0]
        if np.linalg.norm(d) < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        for p in img:
            if not (reach <= p[0] < w - reach and reach <= p[1] < h - reach):
                continue
            tried += 1
            off = _line_profile_peak(gray, p, nrm, reach, min_contrast)
            if off is not None:
                pts.append(p + off * nrm)
                line_ids.append(li)
                offsets.append(off)
    if return_offsets:
        return pts, line_ids, tried, np.asarray(offsets)
    return pts, line_ids, tried


def _perp_residuals(hvec: np.ndarray, pts: np.ndarray, line_ids: np.ndarray, ends: np.ndarray) -> np.ndarray:
    H = np.append(hvec, 1.0).reshape(3, 3)
    a = apply_h(H, ends[:, 0])
    b = apply_h(H, ends[:, 1])
    d = b - a
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.linalg.norm(d, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", pts - a[line_ids], nrm[line_ids])


def line_support(gray: np.ndarray, H: np.ndarray, court: CourtModel, per_line: int = 24, reach: int = 10,
                 tol_px: float = 3.0, min_contrast: float = 18.0) -> float:
    """Fraction of projected line samples with a white ridge centred within ``tol_px`` of the projection."""
    pts, _, tried, offs = _sample_ridges(gray, H, court, per_line, reach, min_contrast, return_offsets=True)
    return float(np.sum(np.abs(offs) <= tol_px)) / tried if tried else 0.0


def refine_homography(frame_bgr: np.ndarray, H: np.ndarray, court: CourtModel, per_line: int = 40,
                      reach: int | None = None, min_contrast: float = 18.0, iterations: int = 2) -> LineRefinement:
    """Refine ``H`` (world -> image) so the projected court lines sit on the image line ridges."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    h, w = gray.shape
    r = reach if reach is not None else max(8, int(round(0.008 * w)))
    ends = np.array([[a, b] for _, a, b in court.line_segments()])
    Hc = H / H[2, 2]
    resid = np.zeros(0)
    pts_arr = np.zeros((0, 2))
    for it in range(iterations):
        pts, ids, _ = _sample_ridges(gray, Hc, court, per_line, r if it == 0 else max(8, r // 2), min_contrast)
        if len(pts) < 12:
            break
        pts_arr = np.asarray(pts)
        ids_arr = np.asarray(ids)
        sol = least_squares(_perp_residuals, (Hc.ravel()[:8]), args=(pts_arr, ids_arr, ends),
                            loss="soft_l1", f_scale=1.5, x_scale="jac")
        Hn = np.append(sol.x, 1.0).reshape(3, 3)
        if not np.all(np.isfinite(Hn)):
            break
        Hc = Hn
        resid = _perp_residuals(sol.x, pts_arr, ids_arr, ends)
    inl = np.abs(resid) < 2.5
    rmse = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else float("nan")
    support = line_support(gray, Hc, court, min_contrast=min_contrast)
    return LineRefinement(Hc, support, int(inl.sum()), rmse)
