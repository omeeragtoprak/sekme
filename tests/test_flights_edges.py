"""Flights and their calls at the edges (events/flights.py): play evidence, a rejected fit falling back to the image
kink and its gates, the serve or rally zone, leading flights, the plausibility gates at their limits, flights with
no call and the segmenting around them. Values checked by running the code; each gate was also broken on purpose
in a copy and some test failed (gaps found by a test review)."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.events.flights import (
    IMAGE_BOUNCE_MAX_SPAN_S,
    Flight,
    _image_call,
    _plausibility,
    build_flights,
    measurable_segments,
)
from courtvision.events.hits import Hit
from courtvision.geometry.court import Half, rally_zone, service_box
from courtvision.perception.ball import BallTrack
from courtvision.perception.court import FrameCalibration
from courtvision.perception.players import PlayerTrack
from courtvision.sim.simulate import simulate_shot
from test_events_gates import FPS, _calib, _fit

N = 150


def _scene(camera):
    H = camera.ground_homography()
    calib = [FrameCalibration(True, H, np.linalg.inv(H), camera, 1.0, 0.3) for _ in range(N)]
    players = {side: PlayerTrack(side, np.tile(box, (N, 1)), np.tile(feet, (N, 1))) for side, box, feet in
               (("near", [900.0, 700.0, 960.0, 900.0], [0.5, -11.8]),
                ("far", [900.0, 100.0, 930.0, 170.0], [-1.0, 11.8]))}
    return calib, players


def _ball(xy: np.ndarray, source: int = 1) -> BallTrack:
    src = np.where(np.isfinite(xy[:, 0]), source, 0)
    return BallTrack(xy, src.astype(float), src)


def _shot(camera, start, target, speed, s0, rng_seed=0, lead=0, noise=0.5):
    """One simulated shot placed at frame s0 (first frame = contact); ``lead`` frames before it repeat the first
    detection (a ball seen before the contact, so a hit at s0 is not the first tracked frame)."""
    shot = simulate_shot(camera, np.array(start), np.array(target), speed, fps=FPS, pixel_noise=noise,
                         rng=np.random.default_rng(rng_seed), phase=0.0)
    xy = np.full((N, 2), np.nan)
    m = min(len(shot.pixels), N - s0)
    xy[s0:s0 + m] = shot.pixels[:m]
    xy[s0 - lead:s0] = shot.pixels[0]
    return xy


def _rally(camera, a_target):
    """Near shot from frame 10, far return from the frame where the first ends (k)."""
    rng = np.random.default_rng(1)
    a = simulate_shot(camera, np.array([0.5, -11.0, 1.0]), np.array(a_target), 110, fps=FPS, pixel_noise=0.5,
                      rng=rng, phase=0.0)
    xy = np.full((N, 2), np.nan)
    k = 10 + len(a.pixels)
    xy[10:k] = a.pixels
    b = simulate_shot(camera, np.array([-1.0, 11.0, 1.0]), np.array([1.5, -6.0]), 100, fps=FPS, pixel_noise=0.5,
                      rng=rng, phase=0.0)
    m = min(len(b.pixels), N - k)
    xy[k:k + m] = b.pixels[:m]
    return xy, k


# ---------------------------------------------------------------- 1. leading flight (flights.py 331-339, 360, 390)

def test_a_leading_flight_infers_its_hitter_from_the_direction_and_reports_no_hit_speed(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    # A ball already in the air at the start of the segment (no hit before it): near -> far, then far -> near.
    cases = (([0.5, -11.0, 1.0], [-1.0, 7.0], "near", Half.FAR, (-1.0, 7.0)),
             ([-1.0, 11.0, 1.0], [1.5, -6.0], "far", Half.NEAR, (1.5, -6.0)))
    for start, target, hitter, receiver, bounce_xy in cases:
        ball = _ball(_shot(broadcast_camera, start, target, 100, 10))
        (fl,) = build_flights(ball, [], calib, FPS, players, [(0, N)])
        assert (fl.hitter, fl.receiver) == (hitter, receiver)
        assert any("zaten havadaydı" in n for n in fl.notes)
        # Nothing is measured at a contact that was not seen ...
        assert fl.hit_speed_kmh is None and fl.hit_height_m is None and fl.contact_time is None
        assert fl.net_clearance_m is None and fl.net_clearance_sd_m is None
        # ... but the bounce is still fitted and called on the receiving half.
        assert fl.rejected is None and fl.call is not None and fl.call.verdict == "IN"
        assert fl.fit is not None and np.allclose(fl.fit.bounce_xy, bounce_xy, atol=0.15)


def test_a_flight_after_a_hit_reports_hit_speed_height_and_net_clearance(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    xy, k = _rally(broadcast_camera, [-1.0, 7.0])
    hits = [Hit(10, "near", xy[10].copy(), True), Hit(k, "far", xy[k].copy(), True)]
    first, second = build_flights(_ball(xy), hits, calib, FPS, players, [(0, N)])
    assert first.hit_speed_kmh is None  # the hit at 10 is the first tracked frame: a leading flight
    assert second.hitter == "far" and second.rejected is None
    assert second.hit_speed_kmh == pytest.approx(100.0, abs=8.0)  # simulated at 100 km/h
    assert second.hit_height_m == pytest.approx(1.0, abs=0.15)
    assert second.net_clearance_m is not None and second.net_clearance_m > 0
    assert second.net_clearance_sd_m is not None and 0 < second.net_clearance_sd_m < 0.3
    assert second.contact_time == pytest.approx(k / FPS, abs=0.1)


# ------------------------------------------------- 2. play_continued downgrades OUT, never IN (flights.py 343, 432)

def test_play_continued_downgrades_an_out_to_review_but_leaves_in_and_unfollowed_calls_alone(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    xy, k = _rally(broadcast_camera, [-1.0, 14.0])  # the near shot lands 2.1 m beyond the far baseline
    ball = _ball(xy)
    far_hit = Hit(k, "far", xy[k].copy(), True)
    first = build_flights(ball, [Hit(10, "near", xy[10].copy(), True), far_hit], calib, FPS, players, [(0, N)])[0]
    assert first.play_continued is True
    assert first.call is not None and first.call.verdict == "REVIEW" and first.call.p_in < 0.01
    assert "ralli sürdü" in (first.call.conflict or "")
    assert any("içeride lehine kanıt" in n for n in first.notes)
    # The same bounce followed by the hitter's own side (not the receiver): no evidence, the OUT stands.
    own = build_flights(ball, [Hit(10, "near", xy[10].copy(), True), Hit(k, "near", xy[k].copy(), True)], calib,
                        FPS, players, [(0, N)])[0]
    assert own.play_continued is False and own.call is not None
    assert own.call.verdict == "OUT" and own.call.conflict is None
    assert not any("lehine kanıt" in n for n in own.notes)
    # An IN followed by the receiver's stroke stays IN with no note.
    xy_in, k_in = _rally(broadcast_camera, [-1.0, 7.0])
    ok = build_flights(_ball(xy_in), [Hit(10, "near", xy_in[10].copy(), True),
                                      Hit(k_in, "far", xy_in[k_in].copy(), True)], calib, FPS, players, [(0, N)])[0]
    assert ok.play_continued is True and ok.call is not None and ok.call.verdict == "IN"
    assert not any("lehine kanıt" in n for n in ok.notes)


# ------------------------- 3. a rejected 3D fit falls back to the image kink (flights.py 399-403, 438-467)

def test_a_fit_rejected_for_contact_height_falls_back_to_the_image_kink_and_a_serve_is_not_rejected(
        broadcast_camera):
    calib, players = _scene(broadcast_camera)
    ball = _ball(_shot(broadcast_camera, [0.5, -11.8, 3.4], [-1.0, 5.5], 150, 12, lead=4))
    # 3.4 m is above a rally contact (3.25 m): the fit is not trusted, the call comes from the image.
    rally = Hit(12, "near", ball.xy[12].copy(), True)
    (fl,) = build_flights(ball, [rally], calib, FPS, players, [(0, N)])
    assert fl.fit is not None and fl.rejected is not None and "temas yüksekliği" in fl.rejected
    assert fl.bounce_2d is not None and fl.call is not None and fl.call.verdict == "IN"
    assert fl.kind == "rally" and fl.net_clearance_m is None
    assert fl.bounce_2d[1] == pytest.approx([-1.0, 5.5], abs=0.15)
    assert any("görüntüdeki iz kırılmasından" in n for n in fl.notes)
    # The same ball struck as a serve (toss seen) has the 3.6 m limit and the fit stays.
    serve = Hit(12, "near", ball.xy[12].copy(), True, serve=True, toss_start=2)
    (sv,) = build_flights(ball, [serve], calib, FPS, players, [(0, N)])
    assert sv.rejected is None and sv.bounce_2d is None and sv.kind == "serve"
    assert sv.hit_height_m == pytest.approx(3.4, abs=0.02)  # the height measured on the camera ray replaces the fit's
    assert sv.call is not None and sv.call.verdict == "IN"


def _kink_track(gx, gy, scale=100.0):
    """Five detections around a ground point (gx, gy) metres under H = diag(scale, scale, 1): two arriving, the
    changepoint detection, two leaving (the image kink is exactly at the ground point)."""
    px = np.array([gx, gy]) * scale
    return np.array([px + [-16.0, -40.0], px + [-8.0, -20.0], px + [0.2, 0.3], px + [8.0, -20.0], px + [16.0, -40.0]])


def _image_cal(scale=100.0):
    H = np.diag([scale, scale, 1.0])
    return FrameCalibration(True, H, np.linalg.inv(H), None, 1.0, 0.3)


def _image(gx, gy, *, idx=(0, 1, 2, 3, 4), fps=20.0, serve=False, pc=None, scale=100.0):
    fl = Flight(0, 60, "near", Half.FAR, bounce_frame=2, play_continued=pc)
    return _image_call(fl, np.array(idx), _kink_track(gx, gy, scale), 2, [_image_cal(scale)] * 80, serve,
                       np.array([0.5, -11.0]), fps)


@pytest.mark.parametrize("g, accepted", [((1.0, 3.1), True), ((1.0, 2.9), False),  # IMAGE_BOUNCE_MIN_NET_M = 3
                                         ((5.4, 5.0), True), ((5.7, 5.0), False),  # IMAGE_BOUNCE_MAX_X_M = 5.5
                                         ((1.0, 12.8), True), ((1.0, 13.1), False)])  # IMAGE_BOUNCE_MAX_Y_M = 12.9
def test_the_image_kink_is_refused_near_the_net_and_outside_the_court_plus_one_metre(g, accepted):
    out = _image(*g)
    assert (out.call is not None) is accepted and (out.bounce_2d is not None) is accepted
    if accepted:
        assert out.bounce_2d[1] == pytest.approx(g, abs=0.02)
    else:
        assert any("güvenilmez" in n and "karar yok" in n for n in out.notes)


def test_the_image_kink_depth_sigma_gate_and_the_span_gate_at_its_edge():
    # Same geometry, coarser pixels per metre: 2.5 px noise is 0.42 m (kept) or 0.83 m (refused) deep.
    kept, refused = _image(1.0, 5.0, scale=10.0), _image(1.0, 5.0, scale=5.0)
    assert kept.call is not None and kept.bounce_2d is not None and np.sqrt(kept.bounce_2d[2][1, 1]) < 0.5
    assert refused.call is None and any("σ 0,83" in n for n in refused.notes)
    # 5 frames at 20 fps is exactly 0.25 s: not above IMAGE_BOUNCE_MAX_SPAN_S; 6 frames (0.3 s) is.
    assert IMAGE_BOUNCE_MAX_SPAN_S == 0.25
    assert _image(1.0, 5.0, idx=(0, 2, 3, 4, 5)).call is not None
    wide = _image(1.0, 5.0, idx=(0, 2, 3, 4, 6))
    assert wide.call is None and wide.bounce_2d is None and any("0,30 sn" in n for n in wide.notes)


def test_the_image_kink_serve_reading_is_review_when_service_box_and_rally_zone_disagree():
    # 6.9 m deep, 0.5 m past the service line (6.4 m) but well inside the baseline.
    rally, serve = _image(-1.5, 6.9, serve=False), _image(-1.5, 6.9, serve=True)
    assert rally.call is not None and rally.call.verdict == "IN" and rally.kind == "rally"
    assert serve.kind == "serve" and serve.call is not None and serve.call.verdict == "REVIEW"
    assert serve.call.conflict == "servis olarak dışarıda, ralli vuruşu olarak içeride"
    assert serve.zone == service_box(Half.FAR, 0.5) and rally.zone == rally_zone(Half.FAR)
    # The play evidence applies to the fallback call as well: OUT + receiver played on -> REVIEW.
    assert _image(1.0, 12.5).call.verdict == "OUT"
    assert _image(1.0, 12.5, pc=True).call.verdict == "REVIEW"


# ------------------------------------------- 4. _plausibility gates at their edges (flights.py 156-176)

def _why(fl_kw=None, fit_kw=None, frame_w=1000, kind="rally"):
    fl = Flight(0, 40, "near", Half.FAR, kind=kind, **(fl_kw or {}))
    return _plausibility(fl, _fit(**(fit_kw or {})), frame_w)


@pytest.mark.parametrize("kw, ok, bad", [
    ({"hit_speed_kmh": 260.0}, True, "temas hızı"), ({"hit_speed_kmh": 260.5}, False, "temas hızı"),
    ({"bounce_speed_kmh": 220.0}, True, "sekme hızı"), ({"bounce_speed_kmh": 220.5}, False, "sekme hızı"),
    ({"hit_height_m": 0.06}, True, "temas yüksekliği"), ({"hit_height_m": 0.05}, False, "temas yüksekliği"),
    ({"hit_height_m": -0.2}, False, "temas yüksekliği"),  # a contact below the court is impossible
    ({"hit_height_m": 3.2}, True, "temas yüksekliği"), ({"hit_height_m": 3.25}, False, "temas yüksekliği"),
])
def test_plausibility_flight_gates_at_their_boundaries(kw, ok, bad):
    why = _why(kw)
    assert (why == []) is ok and (ok or any(bad in w for w in why))


def test_plausibility_serve_height_limit_is_looser_than_the_rally_one():
    assert any("temas yüksekliği" in w for w in _why({"hit_height_m": 3.4}, kind="rally"))
    assert _why({"hit_height_m": 3.4}, kind="serve") == []
    assert _why({"hit_height_m": 3.59}, kind="serve") == []
    assert any("3,60 m" in w for w in _why({"hit_height_m": 3.6}, kind="serve"))


@pytest.mark.parametrize("fit_kw, ok, bad", [
    ({"n_obs": 10}, True, "tutarlı gözlem"), ({"n_obs": 9}, False, "tutarlı gözlem"),
    ({"rms": 4.0}, True, "izdüşüm"), ({"rms": 4.01}, False, "izdüşüm"),  # 0.004 * frame_w (1000 px)
    ({"sd": 0.99}, True, "belirlenemedi"), ({"sd": 1.01}, False, "belirlenemedi"),
    ({"xy": (12.0, 0.0)}, True, "dışında"), ({"xy": (12.1, 0.0)}, False, "dışında"),
    ({"xy": (0.0, -19.0)}, True, "dışında"), ({"xy": (0.0, -19.1)}, False, "dışında"),
])
def test_plausibility_fit_gates_at_their_boundaries(fit_kw, ok, bad):
    why = _why(fit_kw=fit_kw)
    assert (why == []) is ok and (ok or any(bad in w for w in why))


def test_plausibility_ignores_hit_quantities_that_a_leading_flight_does_not_have():
    assert _why({}) == []  # hit speed and height None: only the fit is judged


# ------------------------- 5. serve versus rally zone: one cue is decisive only if both readings agree

def test_serve_cue_combinations_choose_the_zone_and_downgrade_a_disagreement(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    target = [-1.5, 6.9]  # past the service line (6.4 m): OUT as a serve, IN as a rally ball

    def run(height, serve, target=target):
        ball = _ball(_shot(broadcast_camera, [0.5, -11.8, height], target, 150, 12, lead=4))
        hit = Hit(12, "near", ball.xy[12].copy(), True, serve=serve, toss_start=2 if serve else None)
        (fl,) = build_flights(ball, [hit], calib, FPS, players, [(0, N)])
        return fl

    both = run(2.95, True)  # toss seen and the contact geometry fits (> 2.1 m, behind the baseline)
    assert both.kind == "serve" and both.call.verdict == "OUT" and both.call.conflict is None
    assert not any("doğrulanamadı" in n for n in both.notes)
    toss_only = run(1.0, True)  # toss seen, contact at 1 m: geometry says rally
    assert toss_only.kind == "serve" and toss_only.call.verdict == "REVIEW"
    assert "top atışı görüldü ama temas geometrisi servise uymuyor" in toss_only.call.conflict
    geom_only = run(2.95, False)  # no toss, but a high contact at the baseline right after a dead ball
    assert geom_only.kind == "serve" and geom_only.call.verdict == "REVIEW"
    assert "temas servise benziyor ama top atışı görülmedi" in geom_only.call.conflict
    rally = run(1.0, False)
    assert rally.kind == "rally" and rally.call.verdict == "IN" and rally.zone == rally_zone(Half.FAR)
    # One cue only, but both readings agree (inside the service box): the call stays, the note remains.
    agree = run(2.95, False, target=[-1.5, 6.0])
    assert agree.kind == "serve" and agree.call.verdict == "IN" and agree.call.conflict is None
    assert any("servis doğrulanamadı" in n for n in agree.notes)


# ---------------- 6. flights that end without a call (flights.py 345-356) and the segmenting around them

def test_a_flight_without_a_bounce_on_the_receiving_half_makes_no_call(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    # A volley-like straight track: no changepoint at all.
    xy = np.full((N, 2), np.nan)
    xy[10:50] = np.column_stack([np.linspace(900, 1000, 40), np.linspace(700, 300, 40)])
    (fl,) = build_flights(_ball(xy), [], calib, FPS, players, [(0, N)])
    assert fl.bounce_frame is None and fl.fit is None and fl.call is None and fl.zone is None
    assert any("belirgin sekme yok" in n for n in fl.notes)
    # A real bounce, but on the hitter's own half (a short ball): the search is restricted to the receiving
    # half, so there is no bounce and no call (it must not be called on the wrong side of the net).
    short = _ball(_shot(broadcast_camera, [0.5, -11.0, 1.0], [0.0, -4.0], 60, 12, lead=4))
    (own,) = build_flights(short, [Hit(12, "near", short.xy[12].copy(), True)], calib, FPS, players, [(0, N)])
    assert own.hitter == "near" and own.bounce_frame is None and own.call is None and own.fit is None
    assert any("belirgin sekme yok" in n for n in own.notes)


def test_interpolated_or_uncalibrated_detections_do_not_count_towards_the_fit(broadcast_camera):
    calib, players = _scene(broadcast_camera)
    xy = _shot(broadcast_camera, [0.5, -11.8, 1.0], [-1.0, 5.5], 150, 12, lead=4)
    hit = Hit(12, "near", xy[12].copy(), True)
    # All interpolated (source 2): nothing to fit.
    (a,) = build_flights(_ball(xy, source=2), [hit], calib, FPS, players, [(0, N)])
    assert a.fit is None and a.call is None and any("yalnızca 0 kalibre tespit" in n for n in a.notes)
    # Detected, but the camera calibration is invalid from frame 20 on: 6 usable detections < min_points 8.
    cut_calib = [FrameCalibration(False) if i >= 20 else c for i, c in enumerate(calib)]
    (b,) = build_flights(_ball(xy), [hit], cut_calib, FPS, players, [(0, N)])
    assert b.fit is None and b.call is None and any("yalnızca 6 kalibre tespit" in n for n in b.notes)


def test_measurable_segments_at_the_min_len_edge_and_with_cuts_inside_invalid_spans():
    cal = _calib([True] * 10 + [False] * 5 + [True] * 10)
    assert measurable_segments(cal, [], 10) == [(0, 10), (15, 25)]  # a run of exactly min_len is kept
    assert measurable_segments(cal, [], 11) == []
    assert measurable_segments([], [], 1) == [] and measurable_segments(_calib([False] * 5), [], 1) == []
    # A cut at frame 5 starts a new shot there; cuts outside the list or inside an invalid span add nothing.
    assert measurable_segments(_calib([True] * 20), [5, 20, 99], 1) == [(0, 5), (5, 20)]
    assert measurable_segments(_calib([True] * 6 + [False] * 3 + [True] * 6), [7, 8], 1) == [(0, 6), (9, 15)]


def test_the_dead_ball_window_before_a_serve_keeps_the_held_ball_out_of_the_previous_flight():
    n = 300
    xy = np.full((n, 2), np.nan)
    xy[0:121] = np.column_stack([np.linspace(600, 900, 121), np.linspace(100, 600, 121)])
    xy[160:185] = (603, 132)  # the server holds the ball: seen, but not in play
    xy[190:261] = np.column_stack([np.linspace(900, 600, 71), np.linspace(600, 100, 71)])
    px = np.zeros(2)
    hits = [Hit(60, "far", px, True), Hit(200, "near", px, True, serve=True, toss_start=185),
            Hit(240, "far", px, True)]
    ends = {}
    for dead in (1.0, 0.0):
        flights = build_flights(_ball(xy), hits, _calib([False] * n), FPS, {}, [(0, n)], dead_ball_s=dead)
        ends[dead] = [(f.start_frame, f.end_frame) for f in flights]
    assert ends[1.0] == [(0, 60), (60, 120), (200, 240), (240, 260)]
    assert ends[0.0][1] == (60, 184)  # without the window the held ball would extend the rally flight
    # A flight shorter than min_points is not built; no flight crosses a segment boundary.
    short = build_flights(_ball(xy), [Hit(60, "far", px, True), Hit(65, "near", px, True)], _calib([False] * n),
                          FPS, {}, [(0, 150)], min_points=8)
    assert [(f.start_frame, f.end_frame) for f in short] == [(0, 60), (65, 120)]
    split = build_flights(_ball(xy), [Hit(60, "far", px, True)], _calib([False] * n), FPS, {}, [(0, 50), (50, 121)])
    assert [(f.start_frame, f.end_frame) for f in split] == [(0, 49), (50, 60), (60, 120)]
