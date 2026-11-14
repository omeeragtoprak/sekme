"""Strokes (hits): persistent reversals of the ball's direction towards / away from the camera.

The reversal must persist, which rejects the short image-space wobble a bounce produces on the near side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.perception.ball import BallTrack
from courtvision.perception.players import PlayerTrack


@dataclass
class Hit:
    frame: int
    side: str  # "near" | "far": who hit it
    px: np.ndarray
    near_player: bool  # ball was next to that player's box
    serve: bool = False  # a toss above this player preceded the contact
    toss_start: int | None = None  # first toss frame (serves only); the point starts here
    contact_sub: float | None = None  # serves: sub-frame contact estimated inside a detection gap


def _smooth(y: np.ndarray, win: int = 3) -> np.ndarray:
    out = np.full_like(y, np.nan)
    half = win // 2
    for i in range(len(y)):
        seg = y[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def _near_box(px: np.ndarray, box: np.ndarray, grow: float = 0.7) -> bool:
    if not np.all(np.isfinite(box)) or not np.all(np.isfinite(px)):
        return False
    h = box[3] - box[1]
    return bool(box[0] - grow * h <= px[0] <= box[2] + grow * h and box[1] - grow * h <= px[1] <= box[3] + 0.3 * h)


def detect_hits(ball: BallTrack, players: dict[str, PlayerTrack], fps: float, frame_h: int) -> list[Hit]:
    """Persistent reversals of the ball's image-y direction (towards/away from the camera)."""
    y = _smooth(ball.xy[:, 1])
    n = len(y)
    k = max(2, int(round(0.12 * fps)))  # ~4 frames at 30 fps
    thr = 0.012 * frame_h
    dy = np.full(n, np.nan)
    dy[k:n - k] = y[2 * k:] - y[:n - 2 * k]
    state = 0
    state_since = 0
    hits: list[Hit] = []
    min_gap = int(0.4 * fps)
    for i in range(n):
        if not np.isfinite(dy[i]) or abs(dy[i]) < thr:
            continue
        s = 1 if dy[i] > 0 else -1  # +1: travelling towards the camera (near side)
        if state == 0:
            state, state_since = s, i
            continue
        if s == state:
            continue
        ahead = dy[i:i + int(0.2 * fps)]
        ahead = ahead[np.isfinite(ahead)]
        if len(ahead) < 3 or np.mean(np.sign(ahead) == s) < 0.8:
            continue  # not persistent: a bounce wobble, not a hit
        lo, hi = max(state_since, i - 2 * k), min(n, i + k + 1)
        seg = y[lo:hi]
        if not np.isfinite(seg).any():
            continue
        j = lo + int(np.nanargmax(seg) if state == 1 else np.nanargmin(seg))
        side = "near" if state == 1 else "far"
        if hits and j - hits[-1].frame < min_gap:
            state, state_since = s, i
            continue
        box = players[side].boxes[j] if side in players else np.full(4, np.nan)
        hits.append(Hit(j, side, ball.xy[j].copy(), _near_box(ball.xy[j], box)))
        state, state_since = s, i
    return hits


def _masked(ball: BallTrack, s: int, e: int) -> BallTrack:
    xy = np.full_like(ball.xy, np.nan)
    xy[s:e] = ball.xy[s:e]
    src = np.zeros_like(ball.source)
    src[s:e] = ball.source[s:e]
    return BallTrack(xy, np.where(src > 0, ball.score, 0.0), src)


def detect_hits_in_segments(ball: BallTrack, players: dict[str, PlayerTrack], fps: float, frame_h: int,
                            segments: list[tuple[int, int]]) -> list[Hit]:
    hits: list[Hit] = []
    for s, e in segments:
        hits += detect_hits(_masked(ball, s, e), players, fps, frame_h)
    return hits
