"""Strokes (hits): persistent reversals of the ball's direction towards / away from the camera.

The reversal must persist, which rejects the short image-space wobble a bounce produces on the near side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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


# Adopted 2026-11-17 (docs/EVAL_SPOT.md, test slot 2): the image-y reversal of a far rally stroke comes at the outgoing
# ball's image apex, after the contact (dev matches: most far strokes the detector finds 0.13-0.3 s off are late),
# and the ball is often not detected next to the player at the contact itself. A line through the incoming ball and
# a curve through the outgoing one (it passes its apex at the reversal) meet at the contact.
CONTACT_WINDOW_S = 0.4  # before the reversal hit
CONTACT_AFTER_S = 0.1  # past it
CONTACT_MIN_ANGLE_DEG = 35.0
CONTACT_MIN_GAIN = 4.0  # one curve's squared error over the line and curve's (with the contact, the ball is two)
CONTACT_MAX_FPS = 30  # nominal rate: the evidence is 25-30 fps video only; faster video is left as it is


def _line(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.stack([t, np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(a, p, rcond=None)
    return coef[0], coef[1], float(((a @ coef - p) ** 2).sum())


def _curve(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """x linear and y quadratic in t."""
    a = np.stack([t, np.ones_like(t)], 1)
    cx, *_ = np.linalg.lstsq(a, p[:, 0], rcond=None)
    q = np.stack([t * t, t, np.ones_like(t)], 1)
    cy, *_ = np.linalg.lstsq(q, p[:, 1], rcond=None)
    return cx, cy, float(((a @ cx - p[:, 0]) ** 2).sum() + ((q @ cy - p[:, 1]) ** 2).sum())


def far_contact(ball: BallTrack, boxes: np.ndarray, frame: int, fps: float, start: int = 0,
                end: int | None = None) -> tuple[int, np.ndarray] | None:
    """The contact of a far stroke found at ``frame``, from the detected ball CONTACT_WINDOW_S before it to
    CONTACT_AFTER_S past it (inside the camera segment [start, end)). The detections are split, at a split before
    the hit, into a line (x and y against time; at least 2 detections at 25-30 fps, 4 at 60) and a curve (x
    linear, y quadratic; at least 0.1 s of detections, 3 at 25-30 fps) where their total squared error is least.
    It is a contact when at their closest approach (between the last detection on the line and the first on the
    curve) the line and the curve turn by at least CONTACT_MIN_ANGLE_DEG, both move at least 30 px/s, they explain
    the ball CONTACT_MIN_GAIN times better than one curve, and the point there is next to the far player's box at
    least BREAK_MIN_HEIGHT box heights above its bottom (lower is a bounce at the feet). Returns that frame and
    point."""
    n = len(ball.xy) if end is None else min(end, len(ball.xy))
    w = max(4, int(round(CONTACT_WINDOW_S * fps)))
    w2 = int(round(CONTACT_AFTER_S * fps))
    m_in = max(2, int(round(fps / 15)))
    m_out = max(3, int(round(0.1 * fps)))
    still = 1.0 / max(1, int(round(fps / 30)))  # px/frame: 30 px/s at 30 fps, 0.5 at 60
    idx = [i for i in range(max(0, start, frame - w), min(n, frame + w2 + 1))
           if ball.source[i] == 1 and np.all(np.isfinite(ball.xy[i]))]
    if len(idx) < m_in + m_out:
        return None
    t = np.array(idx, float)
    p = ball.xy[idx]
    one = _curve(t, p)[2]
    best = None
    for k in range(m_in, len(idx) - m_out + 1):
        if t[k] > frame + 1:  # the contact comes before the reversal hit
            break
        va, pa, sa = _line(t[:k], p[:k])
        cx, cy, sb = _curve(t[k:], p[k:])
        if best is None or sa + sb < best[0]:
            best = (sa + sb, k, va, pa, cx, cy)
    if best is None:
        return None
    two, k, va, pa, cx, cy = best
    ts = np.linspace(t[k - 1], t[k], max(2, int(t[k] - t[k - 1]) * 4 + 1))  # quarter frames
    on_line = pa[None, :] + va[None, :] * ts[:, None]
    on_curve = np.stack([cx[0] * ts + cx[1], cy[0] * ts ** 2 + cy[1] * ts + cy[2]], 1)
    j = int(np.argmin(((on_line - on_curve) ** 2).sum(1)))
    tc = float(ts[j])
    vb = np.array([cx[0], 2 * cy[0] * tc + cy[1]])
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < still or nb < still:
        return None  # about still: no direction to turn
    angle = float(np.degrees(np.arccos(np.clip(va @ vb / (na * nb), -1.0, 1.0))))
    if angle < CONTACT_MIN_ANGLE_DEG or one < CONTACT_MIN_GAIN * max(two, 1e-6):
        return None
    q = (on_line[j] + on_curve[j]) / 2
    g = int(round(tc))
    box = boxes[g]
    if not np.all(np.isfinite(box)) or not _near_box(q, box):
        return None
    if q[1] > box[3] - BREAK_MIN_HEIGHT * (box[3] - box[1]):
        return None  # at the feet: a bounce, not the contact
    return g, q


def move_far_contacts(hits: list[Hit], ball: BallTrack, players: dict[str, PlayerTrack], fps: float,
                      segments: list[tuple[int, int]]) -> list[Hit]:
    """Each far rally hit (not a serve: its contact comes from the toss) moved back to its far_contact inside its
    camera segment, when that is before it and after the previous hit; the moved hit is next to the far player by
    construction. Video faster than CONTACT_MAX_FPS (nominal) is left as it is."""
    if "far" not in players or int(round(fps)) > CONTACT_MAX_FPS:
        return sorted(hits, key=lambda x: x.frame)
    boxes = players["far"].boxes
    out: list[Hit] = []
    for h in sorted(hits, key=lambda x: x.frame):
        seg = next(((s, e) for s, e in segments if s <= h.frame < e), None)
        if h.side == "far" and not h.serve and seg is not None:
            got = far_contact(ball, boxes, h.frame, fps, start=seg[0], end=seg[1])
            if got is not None and got[0] < h.frame and (not out or got[0] > out[-1].frame):
                h = replace(h, frame=got[0], px=got[1], near_player=True)
        out.append(h)
    return out
