"""Geometry of the ball's image track: two-piece changepoints and line-meeting kinks (bounces, contacts)."""

from __future__ import annotations

import numpy as np


def changepoint(t: np.ndarray, xy: np.ndarray, allowed: np.ndarray) -> tuple[int | None, float]:
    """Index splitting the track into two quadratic pieces best; returns (index, SSE gain ratio)."""
    def sse(a: int, b: int) -> float:
        if b - a < 3:
            return np.inf
        tt = t[a:b]
        return float(sum(np.sum((xy[a:b, d] - np.polyval(np.polyfit(tt, xy[a:b, d], 2), tt)) ** 2)
                         for d in (0, 1)))

    n = len(t)
    whole = sse(0, n)
    best, best_cost = None, np.inf
    for b in range(3, n - 2):
        if not allowed[b]:
            continue
        c = sse(0, b + 1) + sse(b, n)
        if c < best_cost:
            best, best_cost = b, c
    gain = whole / max(best_cost, 1e-6) if best is not None else 1.0
    return best, gain


def refine_contact(frames: np.ndarray, xy: np.ndarray, contact: int, max_back: int = 3, jump: float = 2.0) -> int:
    """Serve contact as the meeting point of the toss and the struck flight (image lines through two detections each).

    The changepoint can only land on detected frames, so a missed contact frame pushes it later; a late
    contact shortens the flight and inflates the serve speed (US Open 2023 serve 28: +23 % vs the radar).
    """
    frames = np.asarray(frames)
    if len(frames) < 4:
        return contact
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(frames)
    for i in range(1, len(frames) - 2):
        if not contact - max_back <= frames[i + 1] <= contact or speed[i] < jump * max(speed[i - 1], 1e-6):
            continue
        a1, b1 = xy[i], (xy[i] - xy[i - 1]) / (frames[i] - frames[i - 1])  # toss line through i-1, i
        a2, b2 = xy[i + 1], (xy[i + 2] - xy[i + 1]) / (frames[i + 2] - frames[i + 1])  # struck line through i+1, i+2
        # Closest approach of P1(f) = a1 + b1 (f - f_i) and P2(f) = a2 + b2 (f - f_{i+1}).
        da = (a1 - b1 * frames[i]) - (a2 - b2 * frames[i + 1])
        db = b1 - b2
        f = -float(da @ db) / max(float(db @ db), 1e-9)
        return int(round(min(max(f, frames[i]), frames[i + 1])))
    return contact


def contact_in_gap(frames: np.ndarray, xy: np.ndarray, contact: int, min_gap: int = 2,
                   jump: float = 2.0) -> tuple[float, np.ndarray] | None:
    """Serve contact hidden in a detection gap right after the last toss detection ``contact``.

    The changepoint can only split on a detected frame; when the racket and the backdrop hide the ball
    from the end of the toss until it is well on its way (2025 QF serve, frames 8881-8889 missed), the
    split lands on the last toss frame and the contact comes out early. If the next detection is at least
    ``min_gap`` frames later and the ball covers the gap ``jump`` times faster than the toss moved, the
    contact is where the toss line meets the struck line, inside the gap. Returns (sub-frame contact,
    image point) or None.
    """
    frames = np.asarray(frames)
    c = int(np.searchsorted(frames, contact))
    if c < 1 or c + 2 >= len(frames) or frames[c] != contact or frames[c + 1] - frames[c] < min_gap:
        return None
    toss = np.linalg.norm(xy[c] - xy[c - 1]) / (frames[c] - frames[c - 1])
    across = np.linalg.norm(xy[c + 1] - xy[c]) / (frames[c + 1] - frames[c])
    if across < jump * max(toss, 1e-6):
        return None
    f, p = kink_point(frames[c - 1:c + 1], xy[c - 1:c + 1], frames[c + 1:c + 3], xy[c + 1:c + 3])
    return f, p


def kink_point(f_pre: np.ndarray, p_pre: np.ndarray, f_post: np.ndarray,
               p_post: np.ndarray) -> tuple[float, np.ndarray]:
    """Where the image line through two samples before an event meets the line through two after it.

    Returns the (sub-frame) time of closest approach, clipped between the samples, and the image point.
    """
    b1 = (p_pre[1] - p_pre[0]) / (f_pre[1] - f_pre[0])
    b2 = (p_post[1] - p_post[0]) / (f_post[1] - f_post[0])
    da = (p_pre[1] - b1 * f_pre[1]) - (p_post[0] - b2 * f_post[0])
    db = b1 - b2
    f = -float(da @ db) / max(float(db @ db), 1e-9)
    f = min(max(f, float(f_pre[1])), float(f_post[0]))
    return f, 0.5 * ((p_pre[1] + b1 * (f - f_pre[1])) + (p_post[0] + b2 * (f - f_post[0])))
