"""Player kinematics from ground-plane foot positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class Kinematics:
    position: np.ndarray  # (N, 2) smoothed metres, NaN when unknown
    speed_kmh: np.ndarray  # (N,)
    distance_m: np.ndarray  # (N,) cumulative
    peak_kmh: float  # robust peak (98th percentile of the smoothed speed)
    mean_kmh: float
    outliers_removed: int = 0


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(valid)))
    return segs


def _fill_short_gaps(world: np.ndarray, max_gap: int) -> np.ndarray:
    out = world.copy()
    ok = np.flatnonzero(np.all(np.isfinite(world), axis=1))
    for a, b in zip(ok[:-1], ok[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            for j in range(a + 1, b):
                w = (j - a) / (b - a)
                out[j] = (1 - w) * world[a] + w * world[b]
    return out


def hampel(world: np.ndarray, half: int = 4, k: float = 3.0, floor_m: float = 0.35) -> tuple[np.ndarray, int]:
    """Replace foot-point outliers (box glitches, brief identity swaps) with NaN."""
    out = world.copy()
    n_out = 0
    for i in range(len(world)):
        if not np.all(np.isfinite(world[i])):
            continue
        win = world[max(0, i - half):i + half + 1]
        win = win[np.all(np.isfinite(win), axis=1)]
        if len(win) < 3:
            continue
        med = np.median(win, axis=0)
        mad = 1.4826 * np.median(np.linalg.norm(win - med, axis=1))
        if np.linalg.norm(world[i] - med) > max(k * mad, floor_m):
            out[i] = np.nan
            n_out += 1
    return out, n_out


def player_kinematics(world: np.ndarray, fps: float, window_s: float = 0.6) -> Kinematics:
    """Hampel-cleaned, Savitzky-Golay-smoothed positions; speeds from the smoothed derivative.

    Foot points jitter with the bounding box (decimetres on a far player at
    720p); the quadratic window keeps sprint accelerations while removing that
    noise. The peak is reported as the 98th percentile so one glitch cannot
    set it.
    """
    world, n_out = hampel(world)
    world = _fill_short_gaps(world, max_gap=max(2, int(0.2 * fps)))
    n = len(world)
    pos = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    win = max(5, int(round(window_s * fps)) | 1)
    for s, e in _segments(np.all(np.isfinite(world), axis=1)):
        seg = world[s:e]
        if e - s >= win:
            pos[s:e] = savgol_filter(seg, win, 2, axis=0)
            vel[s:e] = savgol_filter(seg, win, 2, deriv=1, delta=1.0 / fps, axis=0)
        else:  # too short to smooth: keep the position, do not claim a speed
            pos[s:e] = seg
    speed = np.linalg.norm(vel, axis=1) * 3.6
    step = np.zeros(n)
    ok = np.all(np.isfinite(pos), axis=1)
    both = ok[1:] & ok[:-1]
    step[1:][both] = np.linalg.norm(np.diff(pos, axis=0)[both], axis=1)
    dist = np.cumsum(step)
    finite = speed[np.isfinite(speed)]
    return Kinematics(pos, speed, dist, float(np.percentile(finite, 98)) if len(finite) else 0.0,
                      float(finite.mean()) if len(finite) else 0.0, n_out)
