"""Serves: a slow, near-vertical toss above a player standing on the baseline, with a dead ball before it,
followed by a contact that sends the ball towards the opponent.

They are found separately from rally strokes: a far server's contact does not reverse the ball's image
direction (toss descent and serve both travel down the image), and the toss apex looks like a reversal.
"""

from __future__ import annotations

import numpy as np

from courtvision.events.hits import Hit
from courtvision.events.image_track import changepoint, contact_in_gap, refine_contact
from courtvision.perception.ball import BallTrack
from courtvision.perception.players import PlayerTrack

TOSS_REACH = 0.35  # ball within this many box heights beside the server's box


TOSS_ABOVE = 0.25  # ball above the shoulders: higher than this many box heights below the box top


TOSS_APEX = 0.3  # the toss (or the part we see) reaches this many box heights above the head


TOSS_MAX_SPEED = 0.15  # box heights per frame; a struck ball is faster


SERVER_ABS_Y = (10.9, 13.5)  # server's feet on, or up to ~1.6 m behind, the baseline (m)


# Candidate, off by default (docs/EVAL_SPOT.md, candidate 1): the ball net loses the nearly still ball at the
# toss apex for 0.18-0.25 s (US Open 2025 final, near serves), which splits the toss into two runs.
TOSS_BRIDGE_S = 0.3


def _box_near(track: PlayerTrack, i: int, win: int = 5) -> np.ndarray | None:
    b = track.boxes[max(0, i - win):i + win + 1]
    b = b[np.isfinite(b).all(axis=1)]
    return np.median(b, axis=0) if len(b) else None


def _toss_runs(ball: BallTrack, track: PlayerTrack, s: int, e: int, bridge: int = 0,
               frame_scale: int = 1) -> list[tuple[int, int]]:
    """Runs of >= 3 detections that are slow and above the player's shoulders (gaps up to 2 frames).

    ``frame_scale`` (candidate, 2 at 60 fps): the gap and the per-frame speed limit were set at 30 fps.

    ``bridge`` > 0 (candidate): two runs separated only by missed frames, at most ``bridge`` of them, are
    joined when the ball rises into the gap and falls out of it, i.e. the gap holds the toss apex.
    """
    runs: list[tuple[int, int]] = []
    run: list[int] = []
    for i in range(s, e):
        if ball.source[i] != 1:
            if run and i - run[-1] > 2 * frame_scale:
                runs.append((run[0], run[-1]))
                run = []
            continue
        box = _box_near(track, i)
        ok = False
        if box is not None:
            h = box[3] - box[1]
            x, y = ball.xy[i]
            prev = ball.xy[i - 1] if i > 0 else np.full(2, np.nan)
            speed = float(np.linalg.norm(ball.xy[i] - prev)) if np.all(np.isfinite(prev)) else 0.0
            ok = (box[0] - TOSS_REACH * h <= x <= box[2] + TOSS_REACH * h and y < box[1] + TOSS_ABOVE * h
                  and speed < TOSS_MAX_SPEED / frame_scale * h)
        if ok:
            run.append(i)
        elif run:
            runs.append((run[0], run[-1]))
            run = []
    if run:
        runs.append((run[0], run[-1]))
    if bridge > 0:
        runs = _bridge_apex(ball, runs, bridge)
    return [(a, b) for a, b in runs if np.sum(ball.source[a:b + 1] == 1) >= 3]


def _y_slope(ball: BallTrack, frames: np.ndarray) -> float:
    """Image-y velocity (px per frame) by least squares; nan with fewer than two detections."""
    if len(frames) < 2:
        return float("nan")
    return float(np.polyfit(frames, ball.xy[frames, 1], 1)[0])


def _bridge_apex(ball: BallTrack, runs: list[tuple[int, int]], bridge: int) -> list[tuple[int, int]]:
    """Join consecutive toss runs across a gap of missed frames that holds the apex (up before, down after)."""
    out: list[tuple[int, int]] = []
    for a, b in runs:
        if out:
            pa, pb = out[-1]
            before = np.flatnonzero(ball.source[pa:pb + 1] == 1)[-3:] + pa
            after = np.flatnonzero(ball.source[a:b + 1] == 1)[:3] + a
            if (a - pb - 1 <= bridge and not np.any(ball.source[pb + 1:a] == 1)
                    and _y_slope(ball, before) < 0 < _y_slope(ball, after)):
                out[-1] = (pa, b)
                continue
        out.append((a, b))
    return out


def detect_serves(ball: BallTrack, players: dict[str, PlayerTrack], fps: float,
                  segments: list[tuple[int, int]], dead_ball_s: float = 1.0, bridge_s: float = 0.0,
                  frame_scale: int = 1) -> list[Hit]:
    """Serve contacts: a toss above a baseline player after a dead ball, then a strike towards the opponent.

    ``bridge_s`` > 0 is a candidate (off by default): a toss may span a missed apex up to that long, and
    the contact is searched over the whole run, because a near server's struck ball flies away from the
    camera and can be slow enough in the image to pass the toss gates for a few frames. The toss checks
    then use the detections up to the contact, the direction check those after it.
    """
    bridge = int(round(bridge_s * fps))
    serves: list[Hit] = []
    for S, E in segments:
        for side, track in players.items():
            for a, b in _toss_runs(ball, track, S, E, bridge, frame_scale):
                box = _box_near(track, a)
                if box is None:
                    continue
                h = box[3] - box[1]
                win = np.arange(a, min(E, b + int(0.3 * fps) + 1))
                win = win[ball.source[win] == 1]
                split = None
                if bridge_s > 0:
                    k, _ = changepoint(win / fps, ball.xy[win], win <= b + 3)
                    if k is None:
                        continue
                    split = int(win[k])
                    det = win[win <= split]
                else:
                    det = np.flatnonzero(ball.source[a:b + 1] == 1) + a
                tx, ty = ball.xy[det, 0], ball.xy[det, 1]
                # Vertical travel: a whole toss (bridged) goes up and down, so count the rise and the fall.
                vert = (ty[0] - ty.min()) + (ty[-1] - ty.min()) if split is not None else np.ptp(ty)
                if ty.min() > box[1] - TOSS_APEX * h or np.ptp(tx) > max(0.5 * vert, 0.1 * h):
                    continue  # not high enough above the head, or not a near-vertical toss
                feet = track.world[a:b + 1]
                feet = feet[np.isfinite(feet).all(axis=1)]
                if not len(feet):
                    continue
                fy = float(np.median(feet[:, 1]))
                if not SERVER_ABS_Y[0] <= abs(fy) <= SERVER_ABS_Y[1] or (fy > 0) != (side == "far"):
                    continue
                # Dead ball before: nothing seen, or only the ball held / bounced by the server.
                pre = np.arange(max(S, a - int(dead_ball_s * fps)), a)
                pre = pre[ball.source[pre] == 1]
                reach = np.array([box[0] - TOSS_REACH * h, box[1] - h, box[2] + TOSS_REACH * h, box[3] + 0.3 * h])
                if len(pre) and not np.all((ball.xy[pre, 0] >= reach[0]) & (ball.xy[pre, 0] <= reach[2])
                                           & (ball.xy[pre, 1] >= reach[1]) & (ball.xy[pre, 1] <= reach[3])):
                    continue
                # Contact: changepoint between the toss parabola and the struck flight.
                if split is None:
                    after = win[win > b]
                    if len(after) < 3:
                        continue
                    allowed = (win >= b - 2) & (win <= b + 3)
                    k, _ = changepoint(win / fps, ball.xy[win], allowed)
                    contact = int(win[k]) if k is not None else b
                else:
                    after = win[win > split]
                    if len(after) < 3:
                        continue
                    contact = split
                gap = contact_in_gap(win, ball.xy[win], contact)
                if gap is not None:
                    contact, px, sub = int(round(gap[0])), gap[1], gap[0]
                else:
                    contact = refine_contact(win, ball.xy[win], contact)
                    px, sub = ball.xy[contact].copy(), None
                    if not np.all(np.isfinite(px)):  # refined into a missed frame: between its neighbours
                        lo, hi = win[win < contact], win[win > contact]
                        if len(lo) and len(hi):
                            w = (contact - lo[-1]) / (hi[0] - lo[-1])
                            px = (1 - w) * ball.xy[lo[-1]] + w * ball.xy[hi[0]]
                # Struck towards the opponent: far server sends it down the image, near server up.
                dy = ball.xy[after[-1], 1] - ball.xy[after[0], 1]
                if (dy > 0) != (side == "far"):
                    continue
                serves.append(Hit(contact, side, px, True, serve=True, toss_start=a, contact_sub=sub))
    return sorted(serves, key=lambda h: h.frame)


def merge_serves(hits: list[Hit], serves: list[Hit], fps: float) -> list[Hit]:
    """Replace reversal hits inside a toss (the apex) or at its contact by the serve hit."""
    min_gap = int(0.4 * fps)
    kept = [h for h in hits
            if not any(sv.toss_start is not None and sv.toss_start <= h.frame < sv.frame + min_gap for sv in serves)]
    return sorted(kept + serves, key=lambda h: h.frame)
