"""Strokes (hits): persistent reversals of the ball's direction towards / away from the camera.

The reversal must persist, which rejects the short image-space wobble a bounce produces on the near side.
"""

from __future__ import annotations

import math
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
    if n <= 2 * k:  # too short for the direction window
        return []
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


# Adopted 2026-11-17 (docs/EVAL_SPOT.md, test slot 1; on by default): a far player's stroke towards
# the near side keeps rising in the image after contact, so the image-y reversal above comes at the apex, not at the
# contact, and many far strokes are missed. A break in the ball's 2D image velocity next to the far player marks them.
BREAK_ANGLE_DEG = 60.0
BREAK_SPEED_RATIO = 2.5
# A bounce in front of the far player also turns the ball's image velocity; it happens at the feet, a stroke at racket
# height. A break counts only with the ball at least this many box heights above the box's bottom (dev matches of
# docs/EVAL_SPOT.md: labelled far strokes 10 % quantile 0.56, labelled far bounces 90 % quantile 0.06).
BREAK_MIN_HEIGHT = 0.3


def _velocity(xy: np.ndarray, idx: list[int]) -> np.ndarray | None:
    if len(idx) < 2:
        return None
    t = np.array(idx, float)
    return np.array([np.polyfit(t, xy[idx, d], 1)[0] for d in (0, 1)])


def far_breaks(ball: BallTrack, players: dict[str, PlayerTrack], fps: float,
               segments: list[tuple[int, int]]) -> list[Hit]:
    """Frames where the detected ball's image velocity turns by more than BREAK_ANGLE_DEG or changes speed by more
    than BREAK_SPEED_RATIO times, over k detected frames on each side, with the ball next to the far player's box;
    breaks within 0.2 s are merged (the strongest kept). A merged break with the ball lower than BREAK_MIN_HEIGHT box
    heights above the box's bottom is a bounce at the player's feet, not a stroke, and is dropped. Each other is a far
    hit."""
    if "far" not in players:
        return []
    k = max(3, math.ceil(0.1 * fps - 1e-9))  # 0.1 s rounded up: 3 at 25 and 30 fps, 6 at 60
    merge = max(2, int(round(0.2 * fps)))
    xy, src, boxes = ball.xy, ball.source, players["far"].boxes
    out: list[tuple[int, float]] = []
    for s, e in segments:
        for i in range(s + k, e - k):
            if src[i] != 1 or not np.all(np.isfinite(boxes[i])) or not _near_box(xy[i], boxes[i]):
                continue
            a = _velocity(xy, [j for j in range(i - k, i + 1) if src[j] == 1])
            b = _velocity(xy, [j for j in range(i, i + k + 1) if src[j] == 1])
            if a is None or b is None:
                continue
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na < 1.0 or nb < 1.0:
                continue  # about still: no direction to turn
            angle = float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))
            ratio = max(na, nb) / min(na, nb)
            if angle <= BREAK_ANGLE_DEG and ratio <= BREAK_SPEED_RATIO:
                continue
            strength = angle + 30.0 * (ratio - 1.0)
            if out and i - out[-1][0] <= merge:
                if strength > out[-1][1]:
                    out[-1] = (i, strength)
            else:
                out.append((i, strength))
    return [Hit(i, "far", xy[i].copy(), True) for i, _ in out
            if xy[i][1] <= boxes[i][3] - BREAK_MIN_HEIGHT * (boxes[i][3] - boxes[i][1])]


def add_far_breaks(hits: list[Hit], breaks: list[Hit], fps: float) -> list[Hit]:
    """The breaks with no hit already within 8 frames at 30 fps (0.27 s) join the hits, in frame order."""
    near = max(2, int(round(8 * fps / 30)))
    new = [b for b in breaks if not any(abs(b.frame - h.frame) <= near for h in hits)]
    return sorted([*hits, *new], key=lambda h: h.frame)
