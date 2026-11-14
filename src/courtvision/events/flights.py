"""Flights between strokes and their line calls.

Each flight is searched for one bounce by a two-piece changepoint fit restricted to the receiving half,
fitted in 3D (physics/flight.py) and checked by physical plausibility gates. A rejected fit falls back to
the gated image-kink bounce. The call adds the calibration uncertainty to the fit covariance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from courtvision.calib.camera import Camera, apply_h
from courtvision.events.hits import Hit
from courtvision.events.image_track import changepoint, kink_point
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


# A serve contact ~0.5 m in front of the feet moves along the camera ray by ~0.1-0.2 m in height.
SERVE_CONTACT_TOL_M = 0.3
# Candidate, off by default (docs/EVAL_SERVE_SPEED.md): pin the fit to the measured serve contact height.
# With +-0.3 m and a 0.1 m hinge the fit drifts up to 0.9 m away from it, and the serve speed follows the
# drift (US Open 2025 final, development: +26 % per metre on 15 near-side serves).
SERVE_PIN_TOL_M = 0.05
SERVE_PIN_SCALE_M = 0.01


# The serve contact is seen (end of the toss): it lies within +-1 frame of the detected frame, a uniform
# spread with sd 2/sqrt(12) ~ 0.58 frame; and it is within about half a metre of the server's body.
# With the looser rally priors (2 frames, 1.0/1.3 m) the fit traded contact depth for time and the
# serve speeds came out 12 % above the stadium radar board on the US Open clip (+7 % with these).
SERVE_T_SIGMA_FRAMES = 0.5


SERVE_XY_SIGMA_M = 0.5


# A contact hidden in a detection gap (``contact_in_gap``): on development serves bracketed by eye the
# toss/struck line meeting came out 0-1 frame after the contact; with a prior as wide as the gap
# (sd gap/sqrt(12)) the fit drifted about a frame later still (docs/EVAL_SERVE_CONTACT.md).
GAP_T_SIGMA_FRAMES = 1.0


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
                  dead_ball_s: float = 1.0, pin_serve_height: bool = False) -> list[Flight]:
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
                              min_points, pixel_sigma, dead_ball_s, serve=bool(opener and opener.serve),
                              contact_px=opener.px if opener is not None else None,
                              contact_sub=opener.contact_sub if opener is not None else None,
                              pin_serve_height=pin_serve_height)
                flights.append(fl)
    return flights


def _fit_one(ball: BallTrack, calib: list[FrameCalibration], fps: float, players: dict[str, PlayerTrack],
             s: int, e: int, hitter: str | None, ender: str | None, seg_start: int, frame_w: int,
             min_points: int, pixel_sigma: float, dead_ball_s: float, serve: bool = False,
             contact_px: np.ndarray | None = None, contact_sub: float | None = None,
             pin_serve_height: bool = False) -> Flight:
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
    b, gain = changepoint(t, xy, allowed)
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
    # A contact hidden in a detection gap has no detection: use where the toss and struck lines meet.
    px_contact = contact_px if contact_sub is not None and contact_px is not None else ball.xy[s]
    if (serve and start_anchor is not None and cam is not None and np.all(np.isfinite(feet))
            and np.all(np.isfinite(px_contact))):
        # The serve contact is seen directly (end of the toss): measure its height on the camera ray above
        # the server and pin the fit to it, instead of extrapolating back from the first flight frames.
        h_contact = cam.height_above(px_contact, feet)
        t_c, t_sd = (start_anchor.t, SERVE_T_SIGMA_FRAMES) if contact_sub is None else (contact_sub / fps,
                                                                                       GAP_T_SIGMA_FRAMES)
        tol, z_scale = (SERVE_PIN_TOL_M, SERVE_PIN_SCALE_M) if pin_serve_height else (SERVE_CONTACT_TOL_M, 0.1)
        start_anchor = replace(start_anchor, t=t_c, t_sigma=t_sd / fps, sigma_x=SERVE_XY_SIGMA_M,
                               sigma_y=SERVE_XY_SIGMA_M, z_min=h_contact - tol, z_max=h_contact + tol,
                               z_scale=z_scale)
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
