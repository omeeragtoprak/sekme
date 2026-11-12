"""Hit and bounce events, flight segmentation and per-flight 3D fits.

Hits are where the ball's travel direction (towards / away from the camera)
reverses; the reversal must persist, which rejects the short image-space
wobble a bounce produces on the near side. Each flight between hits is
searched for one bounce by a two-piece changepoint fit, restricted to the
receiving half, and then fitted in 3D (physics/flight.py).

Serves are found separately: a far server's contact does not reverse the
ball's image direction (toss descent and serve both travel down the image),
and the toss apex looks like a reversal. A serve is a slow, near-vertical
toss above a player standing on the baseline, with a dead ball before it,
followed by a contact that sends the ball towards the opponent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from courtvision.calib.camera import Camera, apply_h
from courtvision.geometry.court import (
    BALL_RADIUS,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    NET_POST_X,
    Half,
    Zone,
    rally_zone,
    service_box,
)
from courtvision.i18n import VERDICT, num
from courtvision.officiating.line_call import LineCall, call_bounce, with_play_evidence
from courtvision.perception.ball import BallTrack
from courtvision.perception.court import FrameCalibration
from courtvision.perception.players import PlayerTrack
from courtvision.physics.flight import Anchor, FlightFit, fit_flight, flight_states, flight_states_batch


@dataclass
class Hit:
    frame: int
    side: str  # "near" | "far": who hit it
    px: np.ndarray
    near_player: bool  # ball was next to that player's box
    serve: bool = False  # a toss above this player preceded the contact
    toss_start: int | None = None  # first toss frame (serves only); the point starts here


@dataclass
class Flight:
    start_frame: int  # hit frame (or first tracked frame)
    end_frame: int  # next hit frame (or last tracked frame)
    hitter: str
    receiver: Half
    kind: str = "rally"  # "rally" | "serve"
    bounce_frame: int | None = None  # changepoint estimate
    fit: FlightFit | None = None
    call: LineCall | None = None
    zone: Zone | None = None
    hit_speed_kmh: float | None = None
    bounce_speed_kmh: float | None = None
    net_clearance_m: float | None = None
    net_clearance_sd_m: float | None = None  # 1-sigma, propagated from the fit covariance
    hit_height_m: float | None = None
    contact_time: float | None = None  # refined by the 3D fit (s)
    play_continued: bool | None = None  # receiver struck the ball after the bounce (evidence for IN)
    rejected: str | None = None  # why a fitted flight was not trusted (no 3D call is made)
    bounce_2d: tuple[float, np.ndarray, np.ndarray] | None = None  # (frame, ground xy, cov) from the image kink
    notes: list[str] = field(default_factory=list)


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


def _changepoint(t: np.ndarray, xy: np.ndarray, allowed: np.ndarray) -> tuple[int | None, float]:
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


TOSS_REACH = 0.35  # ball within this many box heights beside the server's box
TOSS_ABOVE = 0.25  # ball above the shoulders: higher than this many box heights below the box top
TOSS_APEX = 0.3  # the toss (or the part we see) reaches this many box heights above the head
TOSS_MAX_SPEED = 0.15  # box heights per frame; a struck ball is faster
SERVER_ABS_Y = (10.9, 13.5)  # server's feet on, or up to ~1.6 m behind, the baseline (m)


def _box_near(track: PlayerTrack, i: int, win: int = 5) -> np.ndarray | None:
    b = track.boxes[max(0, i - win):i + win + 1]
    b = b[np.isfinite(b).all(axis=1)]
    return np.median(b, axis=0) if len(b) else None


def _toss_runs(ball: BallTrack, track: PlayerTrack, s: int, e: int) -> list[tuple[int, int]]:
    """Runs of >= 3 detections that are slow and above the player's shoulders (gaps up to 2 frames)."""
    runs: list[tuple[int, int]] = []
    run: list[int] = []
    for i in range(s, e):
        if ball.source[i] != 1:
            if run and i - run[-1] > 2:
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
                  and speed < TOSS_MAX_SPEED * h)
        if ok:
            run.append(i)
        elif run:
            runs.append((run[0], run[-1]))
            run = []
    if run:
        runs.append((run[0], run[-1]))
    return [(a, b) for a, b in runs if np.sum(ball.source[a:b + 1] == 1) >= 3]


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


def detect_serves(ball: BallTrack, players: dict[str, PlayerTrack], fps: float,
                  segments: list[tuple[int, int]], dead_ball_s: float = 1.0) -> list[Hit]:
    """Serve contacts: a toss above a baseline player after a dead ball, then a strike towards the opponent."""
    serves: list[Hit] = []
    for S, E in segments:
        for side, track in players.items():
            for a, b in _toss_runs(ball, track, S, E):
                box = _box_near(track, a)
                if box is None:
                    continue
                h = box[3] - box[1]
                det = np.flatnonzero(ball.source[a:b + 1] == 1) + a
                tx, ty = ball.xy[det, 0], ball.xy[det, 1]
                if ty.min() > box[1] - TOSS_APEX * h or np.ptp(tx) > max(0.5 * np.ptp(ty), 0.1 * h):
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
                win = np.arange(a, min(E, b + int(0.3 * fps) + 1))
                win = win[ball.source[win] == 1]
                after = win[win > b]
                if len(after) < 3:
                    continue
                allowed = (win >= b - 2) & (win <= b + 3)
                k, _ = _changepoint(win / fps, ball.xy[win], allowed)
                contact = int(win[k]) if k is not None else b
                contact = refine_contact(win, ball.xy[win], contact)
                # Struck towards the opponent: far server sends it down the image, near server up.
                dy = ball.xy[after[-1], 1] - ball.xy[after[0], 1]
                if (dy > 0) != (side == "far"):
                    continue
                serves.append(Hit(contact, side, ball.xy[contact].copy(), True, serve=True, toss_start=a))
    return sorted(serves, key=lambda h: h.frame)


def merge_serves(hits: list[Hit], serves: list[Hit], fps: float) -> list[Hit]:
    """Replace reversal hits inside a toss (the apex) or at its contact by the serve hit."""
    min_gap = int(0.4 * fps)
    kept = [h for h in hits
            if not any(sv.toss_start is not None and sv.toss_start <= h.frame < sv.frame + min_gap for sv in serves)]
    return sorted(kept + serves, key=lambda h: h.frame)


def net_height(x: float) -> float:
    """Net top height (m) at lateral position x, linear between centre strap and posts."""
    return NET_HEIGHT_CENTER + (NET_HEIGHT_POST - NET_HEIGHT_CENTER) * min(abs(x) / NET_POST_X, 1.0)


def _clearance_on_path(pos: np.ndarray) -> float | None:
    cross = np.flatnonzero(np.diff(np.sign(pos[:, 1])) != 0)
    if not len(cross):
        return None
    a, b = pos[cross[0]], pos[cross[0] + 1]
    c = a + (b - a) * (a[1] / (a[1] - b[1]))
    return float(c[2] - BALL_RADIUS - net_height(c[0]))


def net_clearance(params: np.ndarray, t0: float, t1: float) -> float | None:
    """Gap between the ball's underside and the net tape where the path between t0 and t1 crosses y=0."""
    return _clearance_on_path(flight_states(params, np.linspace(t0, t1, 200))[0])


def net_clearance_sd(params: np.ndarray, cov: np.ndarray, t0: float, t1: float) -> float | None:
    """1-sigma of ``net_clearance`` by linear propagation of the parameter covariance."""
    n = len(params)
    steps = 1e-4 * np.maximum(1.0, np.abs(params))
    P = np.vstack([params, params + np.diag(steps)])  # nominal + one forward step per parameter
    pos, _ = flight_states_batch(P, np.linspace(t0, t1, 200))
    c = [_clearance_on_path(pp) for pp in pos]
    if any(v is None for v in c):
        return None
    J = (np.array(c[1:], dtype=float) - c[0]) / steps
    C = cov[:n, :n]  # the contact times (extra rows) do not move the crossing
    return float(np.sqrt(max(J @ C @ J, 0.0)))


CONTACT_IN_FRONT_M = 0.4  # groundstroke contact is typically ahead of the feet, towards the net


def _anchor(players: dict[str, PlayerTrack], side: str, frame: int, fps: float) -> Anchor | None:
    track = players.get(side)
    if track is None or not np.all(np.isfinite(track.world[frame])):
        return None
    xy = track.world[frame].copy()
    xy[1] += CONTACT_IN_FRONT_M if side == "near" else -CONTACT_IN_FRONT_M
    return Anchor(frame / fps, xy)


def measurable_segments(calib: list[FrameCalibration], cuts: list[int], min_len: int) -> list[tuple[int, int]]:
    """Maximal runs [s, e) of calibrated frames that do not cross a shot cut."""
    cut_set = set(cuts)
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for i, c in enumerate(calib):
        if start is not None and (not c.valid or i in cut_set):
            segs.append((start, i))
            start = None
        if c.valid and start is None:
            start = i
    if start is not None:
        segs.append((start, len(calib)))
    return [(a, b) for a, b in segs if b - a >= min_len]


def _masked(ball: BallTrack, s: int, e: int) -> BallTrack:
    xy = np.full_like(ball.xy, np.nan)
    xy[s:e] = ball.xy[s:e]
    src = np.zeros_like(ball.source)
    src[s:e] = ball.source[s:e]
    return BallTrack(xy, np.where(src > 0, ball.score, 0.0), src)


# A serve contact ~0.5 m in front of the feet moves along the camera ray by ~0.1-0.2 m in height.
SERVE_CONTACT_TOL_M = 0.3
# The serve contact is seen (end of the toss): it lies within +-1 frame of the detected frame, a uniform
# spread with sd 2/sqrt(12) ~ 0.58 frame; and it is within about half a metre of the server's body.
# With the looser rally priors (2 frames, 1.0/1.3 m) the fit traded contact depth for time and the
# serve speeds came out 12 % above the stadium radar board on the US Open clip (+7 % with these).
SERVE_T_SIGMA_FRAMES = 0.5
SERVE_XY_SIGMA_M = 0.5
RALLY_MAX_CONTACT_M = 3.25
# ATP serve impact height 3.03 +- 0.12 m (n=630, Baiget et al., Biol Sport 2023, PMC9806764); this
# leaves ~4.7 sigma for tall players and fit error.
SERVE_MAX_CONTACT_M = 3.6


def _plausibility(fl: Flight, fit: FlightFit, frame_w: int) -> list[str]:
    """Physical sanity gates: a fit that fails any of them never produces a call."""
    why = []
    if fit.n_obs < 10:
        why.append(f"yalnızca {fit.n_obs} tutarlı gözlem")
    if fit.rms_px > 0.004 * frame_w:
        why.append(f"yeniden izdüşüm hatası {num(fit.rms_px)} px")
    if fl.hit_speed_kmh is not None and fl.hit_speed_kmh > 260:
        why.append(f"temas hızı {fl.hit_speed_kmh:.0f} km/sa, kaydedilmiş en hızlı vuruştan yüksek")
    if fl.bounce_speed_kmh is not None and fl.bounce_speed_kmh > 220:
        why.append(f"sekme hızı {fl.bounce_speed_kmh:.0f} km/sa")
    max_h = SERVE_MAX_CONTACT_M if fl.kind == "serve" else RALLY_MAX_CONTACT_M
    if fl.hit_height_m is not None and not 0.05 < fl.hit_height_m < max_h:
        why.append(f"temas yüksekliği {num(fl.hit_height_m, 2)} m")
    sd = float(np.sqrt(max(np.linalg.eigvalsh(fit.bounce_cov).max(), 0.0)))
    if sd > 1.0:
        why.append(f"sekme konumu belirlenemedi (σ {num(sd)} m)")
    bx, by = fit.bounce_xy
    if abs(bx) > 12 or abs(by) > 19:
        why.append("sekme oyun alanının dışında")
    return why


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


IMAGE_BOUNCE_PX = 2.5  # detection noise used to propagate the 2D bounce's uncertainty
# Gates for the 2D fallback, chosen on TrackNet games 1-7 (games 8-10 held out): +18 true / +3 false bounces.
IMAGE_BOUNCE_MAX_SD_M = 0.5  # depth 1-sigma; wider says nothing about a line (false kinks: median 1.1 m)
IMAGE_BOUNCE_MIN_NET_M = 3.0  # kinks near the net were mostly the ball crossing it, not bouncing
IMAGE_BOUNCE_MAX_X_M, IMAGE_BOUNCE_MAX_Y_M = 5.5, 12.9  # court plus ~1 m


def image_bounce(frames: np.ndarray, xy: np.ndarray, b: int, calib: list[FrameCalibration],
                 n_samples: int = 200) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Bounce from the image alone, for flights whose 3D fit failed or was not trusted.

    The kink of the image track around detection ``b`` (two detections each side, ``b`` itself left out),
    put on the ground, lifted by the ball radius towards the camera; covariance from detection noise
    (Monte Carlo) plus the calibration term. Returns (frame, ground xy, 2x2 cov) or None.
    """
    if b < 2 or b + 2 >= len(frames):
        return None
    pre, post = slice(b - 2, b), slice(b + 1, b + 3)
    f, px = kink_point(frames[pre], xy[pre], frames[post], xy[post])
    cal = calib[int(round(f))]
    if not cal.valid or cal.H_inv is None:
        return None
    lift = cal.camera.center if cal.camera is not None else None

    def ground(p: np.ndarray) -> np.ndarray:
        g = apply_h(cal.H_inv, p)
        return g if lift is None else g + (lift[:2] - g) * BALL_RADIUS / lift[2]

    rng = np.random.default_rng(int(frames[b]))
    sims = []
    for _ in range(n_samples):
        q_pre = xy[pre] + rng.normal(0, IMAGE_BOUNCE_PX, (2, 2))
        q_post = xy[post] + rng.normal(0, IMAGE_BOUNCE_PX, (2, 2))
        sims.append(ground(kink_point(frames[pre], q_pre, frames[post], q_post)[1]))
    g = ground(px)
    cov = np.cov(np.array(sims).T) + calibration_cov(cal, g)
    return f, g, cov


# Systematic calibration error at the bounce, in image pixels (1 sigma). It is shared by every frame of a
# flight, so the fit cannot average it away. Floor: unmodelled lens distortion leaves a +-0.7 px pattern
# at the image edges (README), about 0.5 px RMS; worse frames use their own court-line residual.
CAL_PX_FLOOR = 0.5


def calibration_cov(cal: FrameCalibration, xy: np.ndarray) -> np.ndarray:
    """2x2 ground covariance (m^2) of the homography error at ground point ``xy``."""
    if cal.H is None or cal.H_inv is None:
        return np.zeros((2, 2))
    px = apply_h(cal.H, np.asarray(xy, float))
    d = 0.5
    J = np.column_stack([(apply_h(cal.H_inv, px + [d, 0]) - apply_h(cal.H_inv, px - [d, 0])) / (2 * d),
                         (apply_h(cal.H_inv, px + [0, d]) - apply_h(cal.H_inv, px - [0, d])) / (2 * d)])
    rmse = cal.line_rmse_px if np.isfinite(cal.line_rmse_px) else 0.0
    s = max(rmse, CAL_PX_FLOOR)
    return s * s * J @ J.T


def build_flights(ball: BallTrack, hits: list[Hit], calib: list[FrameCalibration], fps: float,
                  players: dict[str, PlayerTrack], segments: list[tuple[int, int]] | None = None,
                  frame_w: int = 1920, min_points: int = 8, pixel_sigma: float = 2.5,
                  dead_ball_s: float = 1.0) -> list[Flight]:
    """Flights between hits inside measurable segments (no flight crosses a cut or an uncalibrated span).

    A serve's toss starts a new point: the flight before it ends with the point (no receiver
    evidence), and the toss itself is not a flight.
    """
    segs = segments if segments is not None else [(0, len(ball.xy))]
    flights: list[Flight] = []
    for S, E in segs:
        starts = sorted({h.toss_start for h in hits if h.serve and h.toss_start is not None and S < h.toss_start < E})
        # The previous point ends before the dead-ball window (only the ball held by the server is seen there).
        ends = [st - int(dead_ball_s * fps) for st in starts] + [E]
        for P, Q in zip([S, *starts], ends, strict=True):
            tracked = np.flatnonzero(np.isfinite(ball.xy[P:Q, 0])) + P if Q > P else np.array([], dtype=int)
            if len(tracked) == 0:
                continue
            pt_hits = [h for h in hits if P <= h.frame < Q and (h.serve or tracked[0] < h.frame < tracked[-1])]
            if pt_hits and pt_hits[0].serve:  # no leading flight: before the contact the ball was being tossed
                bounds = [h.frame for h in pt_hits] + [int(tracked[-1])]
                openers: list[Hit | None] = list(pt_hits)
            else:
                pt_hits = [h for h in pt_hits if not h.serve]
                bounds = [int(tracked[0])] + [h.frame for h in pt_hits] + [int(tracked[-1])]
                openers = [None, *pt_hits]
            enders: list[str | None] = [h.side for h in openers[1:] if h is not None] + [None]
            for (s, e), opener, ender in zip(zip(bounds[:-1], bounds[1:], strict=True), openers, enders,
                                             strict=True):
                if e - s < min_points:
                    continue
                fl = _fit_one(ball, calib, fps, players, s, e, opener.side if opener else None, ender, S, frame_w,
                              min_points, pixel_sigma, dead_ball_s, serve=bool(opener and opener.serve))
                flights.append(fl)
    return flights


def _fit_one(ball: BallTrack, calib: list[FrameCalibration], fps: float, players: dict[str, PlayerTrack],
             s: int, e: int, hitter: str | None, ender: str | None, seg_start: int, frame_w: int,
             min_points: int, pixel_sigma: float, dead_ball_s: float, serve: bool = False) -> Flight:
    leading = hitter is None
    if hitter is None:  # leading flight: infer the hitter from the travel direction
        ys = ball.xy[s:e, 1]
        ys = ys[np.isfinite(ys)]
        hitter = "far" if len(ys) > 1 and ys[-1] > ys[0] else "near"
    receiver = Half.FAR if hitter == "near" else Half.NEAR
    fl = Flight(s, e, hitter, receiver, kind="serve" if serve else "rally")  # final kind is decided after the fit
    if leading:
        fl.notes.append("kamera bölümü başladığında top zaten havadaydı (klip başı ya da kamera geçişi)")
    if ender is None:
        fl.notes.append("sonraki vuruş görülmedi: sayı bitti, top kadrajdan çıktı ya da bölüm sona erdi")
    else:
        fl.play_continued = ender == ("near" if receiver == Half.NEAR else "far")
    idx = np.arange(s + 2, e - 1)  # skip the racket-contact frames at both ends
    ok = (ball.source[idx] == 1) & np.array([calib[i].valid and calib[i].camera is not None for i in idx])
    idx = idx[ok]
    if len(idx) < min_points:
        fl.notes.append(f"yalnızca {len(idx)} kalibre tespit var; 3B uydurma yapılmadı")
        return fl
    t = idx / fps
    xy = ball.xy[idx]
    allowed = np.array([(apply_h(calib[i].H_inv, ball.xy[i])[1] > 0) == (receiver == Half.FAR) for i in idx])
    b, gain = _changepoint(t, xy, allowed)
    if b is None or gain < 1.5:
        fl.notes.append(f"belirgin sekme yok (kırılma kazancı {num(gain, 2)}): vole ya da top kadrajdan çıktı")
        return fl
    fl.bounce_frame = int(idx[b])
    cams: list[Camera] = [calib[i].camera for i in idx]  # type: ignore[misc]
    anchors = []
    start_anchor = None if leading else _anchor(players, hitter, s, fps)
    feet = players[hitter].world[s] if hitter in players else np.full(2, np.nan)
    cam = calib[s].camera
    h_contact = None
    if (serve and start_anchor is not None and cam is not None and np.all(np.isfinite(feet))
            and np.all(np.isfinite(ball.xy[s]))):
        # The serve contact is seen directly (end of the toss): measure its height on the camera ray above
        # the server and pin the fit to it, instead of extrapolating back from the first flight frames.
        h_contact = cam.height_above(ball.xy[s], feet)
        start_anchor = replace(start_anchor, t_sigma=SERVE_T_SIGMA_FRAMES / fps, sigma_x=SERVE_XY_SIGMA_M,
                               sigma_y=SERVE_XY_SIGMA_M, z_min=h_contact - SERVE_CONTACT_TOL_M,
                               z_max=h_contact + SERVE_CONTACT_TOL_M)
    if start_anchor is not None:
        anchors.append(start_anchor)
    if ender is not None and (a := _anchor(players, ender, e, fps)) is not None:
        anchors.append(a)
    fit = fit_flight(t, xy, cams, float(t[b]), pixel_sigma=pixel_sigma, anchors=tuple(anchors))
    fl.fit = fit
    if not fit.success or not np.all(np.isfinite(fit.bounce_cov)):
        fl.rejected = f"3B uydurma başarısız: {fit.message}"
        fl.notes.append(fl.rejected)
        return _image_call(fl, idx, xy, b, calib, serve, feet)
    t_hit = float(fit.anchor_times[0]) if start_anchor is not None else s / fps
    fl.contact_time = t_hit if start_anchor is not None else None
    p_hit, v_hit = fit.state_at(t_hit)
    if not leading:
        fl.hit_speed_kmh = float(np.linalg.norm(v_hit) * 3.6)
        fl.hit_height_m = float(p_hit[2])
    if h_contact is not None:
        fl.notes.append(f"servis temas yüksekliği {num(h_contact, 2)} m, servisçinin üstünden geçen kamera ışınında "
                        f"ölçüldü (uydurmanın temas yüksekliği {num(fl.hit_height_m, 2)} m)")
        fl.hit_height_m = h_contact
    fl.bounce_speed_kmh = float(np.linalg.norm(fit.params[3:6]) * 3.6)
    why = _plausibility(fl, fit, frame_w)
    if why:
        fl.rejected = "; ".join(why)
        fl.notes.append(f"3B uydurma reddedildi: {fl.rejected}")
        return _image_call(fl, idx, xy, b, calib, serve, feet)
    if not leading:
        fl.net_clearance_m = net_clearance(fit.params, t_hit, fit.bounce_time)
        fl.net_clearance_sd_m = net_clearance_sd(fit.params, fit.cov, t_hit, fit.bounce_time)
    # A serve starts a point: the ball was dead just before, or the camera segment just began after a cut.
    before = ball.xy[max(seg_start, s - int(dead_ball_s * fps)):max(seg_start, s - int(0.3 * fps)), 0]
    # A ball seen in that window was in play (e.g. this is the return of a serve just after a cut).
    after_dead_ball = not np.isfinite(before).any() and (
        len(before) > 0 or (seg_start > 0 and s - seg_start < dead_ball_s * fps))
    # Two independent serve cues: the toss seen in the image, and the contact geometry (high, at the baseline).
    geom = not leading and (fl.hit_height_m or 0.0) > 2.1 and abs(p_hit[1]) > 11.0
    server_x = float(feet[0]) if np.isfinite(feet[0]) else float(p_hit[0])  # contact is above the server
    if serve or (geom and after_dead_ball):
        fl.kind = "serve"
        fl.zone = service_box(receiver, server_x)
    else:
        fl.zone = rally_zone(receiver)
    bounce_cov = fit.bounce_cov + calibration_cov(calib[fl.bounce_frame], fit.bounce_xy)
    call = call_bounce(fit.bounce_xy, bounce_cov, fl.zone, seed=fl.bounce_frame)
    if fl.kind == "serve" and not (serve and geom):
        # One cue only: decisive only where reading it as a rally shot gives the same verdict.
        cue = ("top atışı görüldü ama temas geometrisi servise uymuyor" if serve
               else "temas servise benziyor ama top atışı görülmedi")
        fl.notes.append(f"servis doğrulanamadı ({cue})")
        alt = call_bounce(fit.bounce_xy, bounce_cov, rally_zone(receiver), seed=fl.bounce_frame)
        if alt.verdict != call.verdict:
            call = replace(call, verdict="REVIEW",
                           conflict=f"servis olarak {VERDICT[call.verdict]}, "
                                    f"ralli vuruşu olarak {VERDICT[alt.verdict]} ({cue})")
    fl.call = with_play_evidence(call, fl.play_continued)
    if fl.play_continued and fl.call.verdict != "IN":
        fl.notes.append("karşı oyuncu bu sekmeden sonra topa vurdu: içeride lehine kanıt, inceleme önerilir")
    return fl


def _image_call(fl: Flight, idx: np.ndarray, xy: np.ndarray, b: int, calib: list[FrameCalibration], serve: bool,
                feet: np.ndarray) -> Flight:
    """Fallback when the 3D fit is missing or not trusted: call the bounce from the image kink (wider sigma)."""
    out = image_bounce(idx, xy, b, calib)
    if out is None:
        fl.notes.append("görüntüden sekme noktası da çıkarılamadı: karar yok")
        return fl
    _, g, cov = out
    if (np.sqrt(cov[1, 1]) > IMAGE_BOUNCE_MAX_SD_M or abs(g[1]) < IMAGE_BOUNCE_MIN_NET_M
            or abs(g[0]) > IMAGE_BOUNCE_MAX_X_M or abs(g[1]) > IMAGE_BOUNCE_MAX_Y_M):
        fl.notes.append(f"görüntüden sekme noktası güvenilmez (x {num(g[0], 1)} m, y {num(g[1], 1)} m, "
                        f"σ {num(np.sqrt(cov[1, 1]), 2)} m): karar yok")
        return fl
    fl.bounce_2d = out
    fl.zone = service_box(fl.receiver, float(feet[0])) if serve and np.isfinite(feet[0]) else rally_zone(fl.receiver)
    call = call_bounce(g, cov, fl.zone, seed=fl.bounce_frame)
    if serve:  # only the toss says "serve" here (no contact geometry): decisive only if both readings agree
        alt = call_bounce(g, cov, rally_zone(fl.receiver), seed=fl.bounce_frame)
        if alt.verdict != call.verdict:
            call = replace(call, verdict="REVIEW", conflict=f"servis olarak {VERDICT[call.verdict]}, "
                                                            f"ralli vuruşu olarak {VERDICT[alt.verdict]}")
    fl.kind = "serve" if serve else "rally"
    fl.call = with_play_evidence(call, fl.play_continued)
    fl.notes.append("sekme 3B uçuştan değil, görüntüdeki iz kırılmasından ölçüldü (daha geniş belirsizlik)")
    return fl


def detect_hits_in_segments(ball: BallTrack, players: dict[str, PlayerTrack], fps: float, frame_h: int,
                            segments: list[tuple[int, int]]) -> list[Hit]:
    hits: list[Hit] = []
    for s, e in segments:
        hits += detect_hits(_masked(ball, s, e), players, fps, frame_h)
    return hits
