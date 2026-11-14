"""Regression tests for failures seen on the US Open 2023 highlights clip."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.events.flights import (
    Flight,
    _plausibility,
    build_flights,
    measurable_segments,
    net_clearance,
    net_clearance_sd,
)
from courtvision.events.hits import Hit, detect_hits_in_segments
from courtvision.events.serves import _toss_runs, detect_serves, merge_serves
from courtvision.geometry.court import Half
from courtvision.perception.ball import BallCandidate, BallTrack, drop_court_fixed, track_ball
from courtvision.perception.court import FrameCalibration
from courtvision.perception.players import PersonDet, PlayerTrack, _plausible
from courtvision.physics.flight import FlightFit


def _calib(valid: list[bool]) -> list[FrameCalibration]:
    return [FrameCalibration(v) for v in valid]


def test_segments_split_at_cuts_and_uncalibrated_frames() -> None:
    # 0-39 rally, close-up 40-59 (invalid), 60-99 rally with a cut at 80 (two shots, both calibrated).
    valid = [True] * 40 + [False] * 20 + [True] * 40
    segs = measurable_segments(_calib(valid), cuts=[0, 40, 60, 80], min_len=10)
    assert segs == [(0, 40), (60, 80), (80, 100)]
    # Too-short pieces are dropped rather than analysed.
    assert measurable_segments(_calib([True] * 5 + [False] + [True] * 30), [0], min_len=10) == [(6, 36)]


def _fit(n_obs: int = 30, rms: float = 1.0, sd: float = 0.1, xy=(0.0, 8.0)) -> FlightFit:
    params = np.zeros(11)
    params[:2] = xy
    cov = np.eye(11) * 1e-4
    cov[:2, :2] = np.diag([sd**2, sd**2])
    return FlightFit(params, cov, rms, n_obs, np.zeros(n_obs), np.zeros((n_obs, 3)), True)


def test_plausibility_rejects_physically_impossible_fits() -> None:
    fl = Flight(0, 20, "near", Half.FAR, hit_speed_kmh=341.0, bounce_speed_kmh=150.0, hit_height_m=3.31)
    why = _plausibility(fl, _fit(n_obs=9), frame_w=1280)
    assert any("temas hızı" in w for w in why)
    assert any("temas yüksekliği" in w for w in why)
    assert any("tutarlı gözlem" in w for w in why)
    # A wide covariance means the bounce is undetermined: no call, whatever the mean says.
    ok = Flight(0, 40, "near", Half.FAR, hit_speed_kmh=120.0, bounce_speed_kmh=80.0, hit_height_m=1.0)
    assert _plausibility(ok, _fit(), 1280) == []
    assert any("belirlenemedi" in w for w in _plausibility(ok, _fit(sd=2.9), 1280))
    assert any("izdüşüm hatası" in w for w in _plausibility(ok, _fit(rms=36.0), 1280))


def test_player_gate_rejects_teleports_and_wrong_sized_people() -> None:
    player = PersonDet(1, np.array([100.0, 100.0, 140.0, 220.0]), 0.9)  # 120 px tall
    ball_kid = PersonDet(7, np.array([300.0, 180.0, 330.0, 240.0]), 0.9)  # 60 px tall, crouching
    last = np.array([0.0, -12.0])
    assert _plausible(np.array([0.2, -12.1]), player, last, 1, 30.0, 120.0)
    assert not _plausible(np.array([4.0, -12.0]), player, last, 1, 30.0, 120.0)  # 4 m in 33 ms
    assert _plausible(np.array([4.0, -12.0]), player, last, 15, 30.0, 120.0)  # 4 m in 0.5 s is fine
    assert not _plausible(np.array([0.1, -12.0]), ball_kid, last, 1, 30.0, 120.0)


FPS = 30.0


def _players(n: int, far_y: float = 12.0) -> dict[str, PlayerTrack]:
    def track(side: str, box: list[float], w: list[float]) -> PlayerTrack:
        return PlayerTrack(side, np.tile(np.array(box, float), (n, 1)), np.tile(np.array(w, float), (n, 1)))
    return {"far": track("far", [600, 100, 630, 172], [-0.8, far_y]),
            "near": track("near", [1000, 600, 1070, 700], [1.0, -12.0])}


def _far_serve_rally(n: int = 140, incoming: bool = False) -> BallTrack:
    """Far server: ball held, toss (apex at frame 30), contact at 40, serve, bounce, return."""
    xy = np.full((n, 2), np.nan)
    for t in range(n):
        if t < 10:
            xy[t] = (300 + 25 * t, 420 - 30 * t) if incoming else (603, 132)  # rally ball or held ball
        elif t <= 40:
            xy[t] = (603 + 0.1 * (t - 10), 46 + 0.21 * (t - 30) ** 2)
        elif t <= 58:
            xy[t] = (606 + 16 * (t - 40), 67 + 18 * (t - 40))
        elif t <= 80:
            xy[t] = (894 + 12 * (t - 58), 391 + 9 * (t - 58))
        elif t <= 120:
            xy[t] = (1158 - 12 * (t - 80), 589 - 12 * (t - 80))
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    return BallTrack(xy, src.astype(float), src)


def test_far_serve_toss_apex_is_not_a_hit_and_contact_is_found() -> None:
    # US Open clip, first point: the toss apex was taken for a hit and the serve contact was missed.
    ball, players = _far_serve_rally(), _players(140)
    hits = detect_hits_in_segments(ball, players, FPS, 720, [(0, 140)])
    assert any(25 <= h.frame <= 35 for h in hits)  # the reversal detector alone sees the apex
    serves = detect_serves(ball, players, FPS, [(0, 140)])
    assert len(serves) == 1
    sv = serves[0]
    assert sv.side == "far" and 39 <= sv.frame <= 41 and sv.toss_start is not None and sv.toss_start <= 15
    merged = merge_serves(hits, serves, FPS)
    assert not any(h.toss_start is None and sv.toss_start <= h.frame < sv.frame + 12 for h in merged)
    assert any(h.side == "near" and 75 <= h.frame <= 85 for h in merged)


def test_toss_like_motion_is_not_a_serve_without_a_dead_ball_or_away_from_the_baseline() -> None:
    # A ball arriving in play just before (an overhead), or a player at the net, is not a serve.
    assert detect_serves(_far_serve_rally(incoming=True), _players(140), FPS, [(0, 140)]) == []
    assert detect_serves(_far_serve_rally(), _players(140, far_y=4.0), FPS, [(0, 140)]) == []


def test_a_serve_starts_a_new_point_and_the_toss_is_not_a_flight() -> None:
    n = 300
    xy = np.full((n, 2), np.nan)
    xy[0:121] = np.column_stack([np.linspace(600, 900, 121), np.linspace(100, 600, 121)])
    xy[180:261] = np.column_stack([np.linspace(900, 600, 81), np.linspace(600, 100, 81)])
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    ball = BallTrack(xy, src.astype(float), src)
    px = np.zeros(2)
    hits = [Hit(40, "far", px, True, serve=True, toss_start=12), Hit(80, "near", px, True),
            Hit(200, "near", px, True, serve=True, toss_start=185), Hit(240, "far", px, True)]
    flights = build_flights(ball, hits, _calib([False] * n), FPS, {}, [(0, n)])
    assert [(f.start_frame, f.end_frame, f.hitter, f.kind) for f in flights] == [
        (40, 80, "far", "serve"), (80, 120, "near", "rally"), (200, 240, "near", "serve"), (240, 260, "far", "rally")]
    assert flights[0].play_continued is True
    # The next point's server is not the receiver of the last shot: no false "play continued" evidence.
    assert flights[1].play_continued is None


def test_court_marks_are_dropped_but_the_ball_is_kept() -> None:
    # US Open clip: WASB fires on the baseline centre marks; a still mark took over the ball track.
    n = 40
    H = np.diag([0.02, 0.02, 1.0])
    cands = [[BallCandidate(500.0, 500.0, 8.0), BallCandidate(100.0 + 20 * i, 300.0, 10.0)] for i in range(n)]
    for i in range(10, 14):  # the ball hangs at a toss apex for a few frames
        cands[i][1] = BallCandidate(300.0, 300.0, 10.0)
    ground = [None if i in (5, 6) else H for i in range(n)]
    out = drop_court_fixed(cands, ground)
    assert all(any(c.y == 300.0 for c in cs) for cs in out)  # the ball survives everywhere
    assert all(not any((c.x, c.y) == (500.0, 500.0) for c in out[i]) for i in range(n) if ground[i] is not None)
    assert any((c.x, c.y) == (500.0, 500.0) for c in out[5])  # uncalibrated frames are left alone


def test_tracker_bridges_a_missed_detection_instead_of_jumping_to_a_decoy() -> None:
    # US Open clip, frames 62-65: one missed frame let a higher-scoring court mark capture the track.
    n = 21
    ball = [BallCandidate(100.0 + 10 * i, 400.0, 5.0) for i in range(n)]
    decoy = BallCandidate(700.0, 100.0, 12.0)
    cands = [[ball[i]] if i < 10 else [decoy] if i == 10 else [ball[i], decoy] for i in range(n)]
    tr = track_ball(cands, 1280)
    assert tr.source[10] == 2  # the miss is interpolated
    assert np.allclose(tr.xy[11:, 1], 400.0) and np.allclose(tr.xy[11:, 0], 100.0 + 10 * np.arange(11, n))


def test_net_clearance_sigma_matches_monte_carlo() -> None:
    # A far-to-near shot: bounce at y=-6 m after crossing the net.
    params = np.array([0.0, -6.0, 0.5, 0.0, -30.0, -4.0, 0.0, -20.0, 3.0, 0.0, 0.0])
    c0 = net_clearance(params, 0.0, 0.5)
    assert c0 is not None
    assert net_clearance_sd(params, np.zeros((13, 13)), 0.0, 0.5) == 0.0
    cov = np.diag([0.02, 0.05, 1e-4, 0.3, 1.0, 0.3, 0.3, 1.0, 0.3, 0.01, 0.01]) ** 2
    sd = net_clearance_sd(params, cov, 0.0, 0.5)
    rng = np.random.default_rng(1)
    mc = [net_clearance(p, 0.0, 0.5) for p in rng.multivariate_normal(params, cov, 400)]
    assert sd is not None and sd > 0
    assert np.std([v for v in mc if v is not None]) == pytest.approx(sd, rel=0.2)


def test_calibration_cov_scales_pixel_error_to_ground():
    import numpy as np

    from courtvision.events.flights import CAL_PX_FLOOR, calibration_cov
    from courtvision.perception.court import FrameCalibration

    H = np.diag([100.0, 100.0, 1.0])  # 1 m on the ground = 100 px
    cal = FrameCalibration(True, H, np.linalg.inv(H), None, 1.0, 0.2)
    cov = calibration_cov(cal, np.array([3.0, 4.0]))
    assert np.allclose(cov, np.eye(2) * (CAL_PX_FLOOR / 100.0) ** 2, rtol=1e-4)
    worse = calibration_cov(FrameCalibration(True, H, np.linalg.inv(H), None, 1.0, 2.0), np.array([3.0, 4.0]))
    assert np.allclose(worse, np.eye(2) * (2.0 / 100.0) ** 2, rtol=1e-4)


def test_refine_contact_finds_the_kink_across_a_missing_frame():
    import numpy as np

    from courtvision.events.image_track import refine_contact

    # US Open 2023 serve 28 (HD run): slow toss descent, contact frame 1309 missed, fast struck flight.
    frames = np.array([1305, 1306, 1308, 1310, 1311, 1314, 1315])
    xy = np.array([[616.3, 65.8], [616.5, 69.5], [620.8, 77.1], [621.7, 100.1], [623.0, 111.6],
                   [630.1, 160.8], [634.8, 182.5]])
    assert refine_contact(frames, xy, 1311) == 1308
    # Same serve with 1309 detected (360p-era run): the changepoint said 1309, the kink is at 1308.3.
    frames2 = np.array([1305, 1306, 1308, 1309, 1310, 1311])
    xy2 = np.array([[616.3, 65.8], [616.5, 69.5], [619.4, 76.3], [621.1, 86.5], [621.1, 99.4], [622.1, 112.1]])
    assert refine_contact(frames2, xy2, 1309) == 1308


def test_contact_in_gap_puts_a_hidden_serve_contact_inside_the_gap():
    from courtvision.events.image_track import contact_in_gap

    # US Open 2025 QF serve, ball net v4: toss seen to 8880, ball hidden by the racket and the backdrop
    # lettering 8881-8889 (contact ~8887-8888 by eye), struck flight from 8890.
    frames = np.array([8873, 8874, 8875, 8876, 8877, 8878, 8880, 8890, 8891, 8892])
    xy = np.array([[905.2, 78.2], [904.6, 78.5], [904.9, 79.7], [903.6, 80.5], [905.0, 83.8], [905.0, 83.6],
                   [906.1, 86.2], [923.9, 130.8], [931.8, 145.1], [940.7, 160.7]])
    f, p = contact_in_gap(frames, xy, 8880)
    assert 8886 < f < 8889
    assert 905 < p[0] < 925 and 86 < p[1] < 131
    # No gap after the split frame, or no speed-up across it: nothing to do.
    assert contact_in_gap(frames, xy, 8877) is None
    slow = xy.copy()
    slow[7:] = xy[6] + np.array([[1.0, 2.0], [1.1, 2.2], [1.2, 2.4]])
    assert contact_in_gap(frames, slow, 8880) is None


def test_kink_point_is_where_two_image_lines_meet():
    from courtvision.events.image_track import kink_point

    f, p = kink_point(np.array([1.0, 2.0]), np.array([[10.0, 100.0], [20.0, 110.0]]),
                      np.array([4.0, 5.0]), np.array([[40.0, 114.0], [50.0, 108.0]]))
    # Incoming y = 100 + 10 (f - 1), outgoing y = 114 - 6 (f - 4): they meet at f = 3.0 (x = 30).
    assert abs(f - 3.0) < 1e-9 and np.allclose(p, [30.0, 120.0])


def test_image_bounce_gives_a_ground_point_with_uncertainty():
    from courtvision.events.flights import image_bounce
    from courtvision.perception.court import FrameCalibration

    H = np.diag([100.0, 100.0, 1.0])  # 1 m = 100 px, camera straight above (no radius lift)
    cal = FrameCalibration(True, H, np.linalg.inv(H), None, 1.0, 0.3)
    frames = np.array([8, 9, 10, 11, 12])  # sample 2 is the changepoint (bounce) detection, not used
    xy = np.array([[280.0, 390.0], [290.0, 400.0], [301.0, 405.0], [310.0, 400.0], [320.0, 390.0]])
    out = image_bounce(frames, xy, 2, [cal] * 20)
    assert out is not None
    f, g, cov = out
    assert abs(f - 10.0) < 1e-6 and np.allclose(g, [3.0, 4.1], atol=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def _near_serve_60fps(n: int = 140) -> BallTrack:
    """Near server at 60 fps as on the US Open 2025 final (frames 1878-1941): ball held, toss with the apex
    missed for 13 frames (the ball net loses the nearly still ball), contact at 75, struck ball slow in the
    image (it flies away from the camera), so it still passes the toss gates for a while."""
    xy = np.full((n, 2), np.nan)
    for t in range(n):
        if t < 20:
            xy[t] = (1060, 640)  # held below the shoulders
        elif t <= 75:
            if not 44 <= t <= 56:
                xy[t] = (1060 - 0.4 * (t - 20), 470 + 0.189 * (t - 50) ** 2)
        elif t <= 120:
            xy[t] = (1038 - 5 * (t - 75), 588 - 6 * (t - 75))
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    return BallTrack(xy, src.astype(float), src)


def test_a_near_serve_with_the_toss_apex_missed_and_a_slow_struck_ball_is_found_by_the_bridge_candidate():
    ball, players = _near_serve_60fps(), _players(140)
    # Current detector: the apex gap splits the toss, the struck ball extends the second run past the
    # contact, and neither run passes (wrong direction after the first, not vertical for the second).
    assert detect_serves(ball, players, 60.0, [(0, 140)]) == []
    serves = detect_serves(ball, players, 60.0, [(0, 140)], bridge_s=0.3)
    assert len(serves) == 1 and serves[0].side == "near" and 74 <= serves[0].frame <= 76
    # The far serve of the other tests is found the same way with the candidate.
    far = detect_serves(_far_serve_rally(), _players(140), FPS, [(0, 140)], bridge_s=0.3)
    assert len(far) == 1 and 39 <= far[0].frame <= 41


def test_the_bridge_does_not_join_a_toss_to_a_struck_ball_seen_again_after_a_contact_gap():
    # US Open 2025 final serve 25425 (far, 60 fps): the ball is missed for 6 frames after the contact and
    # the struck ball is still slow and above the shoulders when it reappears. Falling on both sides of the
    # gap, so it is not an apex: the toss stays one run and the contact goes inside the gap, as without it.
    n = 140
    xy = np.full((n, 2), np.nan)
    for t in range(n):
        if t < 10:
            xy[t] = (603, 132)
        elif t <= 58:
            xy[t] = (603 + 0.1 * (t - 10), 46 + 0.06 * (t - 40) ** 2)
        elif 65 <= t <= 100:
            xy[t] = (608 + 2 * (t - 58), 65 + 6 * (t - 58))
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    ball = BallTrack(xy, src.astype(float), src)
    players = _players(n)
    assert _toss_runs(ball, players["far"], 0, n, bridge=18) == _toss_runs(ball, players["far"], 0, n) == [(11, 58)]
    for bridge in (0.0, 0.3):
        serves = detect_serves(ball, players, 60.0, [(0, n)], bridge_s=bridge)
        assert len(serves) == 1 and serves[0].side == "far" and 58 < serves[0].contact_sub < 65
    # The near serve's apex gap (rising into it, falling out of it) is joined.
    near = _near_serve_60fps()
    assert len(_toss_runs(near, players["near"], 0, 140)) == 2
    joined = _toss_runs(near, players["near"], 0, 140, bridge=18)
    assert len(joined) == 1 and joined[0][0] < 44 and joined[0][1] > 75


def test_a_bridged_toss_that_drifts_sideways_is_still_a_toss_and_its_apex_is_not_a_contact():
    # US Open 2024 final serve 1409 (far, 60 fps): apex missed for 13 frames, toss drifting ~40 px sideways.
    # Without the bridge the first half ends at the gap and, since the ball comes down "towards the
    # receiver" after it, is taken for a serve too (a false serve at the apex). Bridged, the whole toss
    # rises and falls ~90 px in all: counting both, it is vertical enough.
    n = 140
    xy = np.full((n, 2), np.nan)
    for t in range(n):
        if t < 10:
            xy[t] = (603, 132)
        elif t <= 66 and not 30 <= t <= 42:
            xy[t] = (603 + 0.7 * (t - 10), 52 + 0.0607 * (t - 36) ** 2)
        elif 67 <= t <= 100:
            xy[t] = (642 + 14 * (t - 66), 107 + 20 * (t - 66))
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    ball = BallTrack(xy, src.astype(float), src)
    old = detect_serves(ball, _players(n), 60.0, [(0, n)])
    assert len(old) == 2 and old[0].frame < 31  # the apex split, a false serve
    new = detect_serves(ball, _players(n), 60.0, [(0, n)], bridge_s=0.3)
    assert len(new) == 1 and new[0].side == "far" and 65 <= new[0].frame <= 68


def test_the_pin_candidate_narrows_only_the_serve_anchor_to_the_measured_contact_height(broadcast_camera,
                                                                                       monkeypatch):
    import courtvision.events.flights as flights_mod
    from courtvision.sim.simulate import simulate_shot

    fps, n, s0 = 60.0, 120, 10
    shot = simulate_shot(broadcast_camera, np.array([0.5, -11.8, 2.95]), np.array([-1.0, 5.5]), 190, fps=fps,
                         pixel_noise=0.0, rng=np.random.default_rng(0), phase=0.0)  # first frame = contact
    xy = np.full((n, 2), np.nan)
    m = min(len(shot.pixels), n - s0)
    xy[s0:s0 + m] = shot.pixels[:m]
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    ball = BallTrack(xy, src.astype(float), src)
    H = broadcast_camera.ground_homography()
    calib = [FrameCalibration(True, H, np.linalg.inv(H), broadcast_camera, 1.0, 0.3) for _ in range(n)]
    players = {side: PlayerTrack(side, np.tile(box, (n, 1)), np.tile(feet, (n, 1))) for side, box, feet in
               (("near", [900.0, 700.0, 960.0, 900.0], [0.5, -11.8]),
                ("far", [900.0, 100.0, 930.0, 170.0], [0.0, 12.0]))}
    serve = Hit(s0, "near", xy[s0].copy(), True, serve=True, toss_start=s0 - 5)

    class SeenError(Exception):
        pass

    def capture(*args, anchors=(), **kwargs):
        raise SeenError(anchors)

    monkeypatch.setattr(flights_mod, "fit_flight", capture)
    got = {}
    for pin in (False, True):
        with pytest.raises(SeenError) as exc:
            build_flights(ball, [serve], calib, fps, players, [(0, n)], pin_serve_height=pin)
        got[pin] = exc.value.args[0][0]
    h = 0.5 * (got[False].z_min + got[False].z_max)
    assert abs(h - 2.95) < 0.05  # measured on the camera ray above the server
    assert (got[False].z_max - got[False].z_min, got[False].z_scale) == pytest.approx((0.6, 0.1))
    assert (got[True].z_max - got[True].z_min, got[True].z_scale) == pytest.approx((0.1, 0.01))
    assert 0.5 * (got[True].z_min + got[True].z_max) == pytest.approx(h)


def test_at_60_fps_the_court_fixed_filter_needs_its_frame_counts_scaled_to_keep_the_toss_apex() -> None:
    # US Open 2025 final, serve at frame 1933 (60 fps): the net found the ball at the apex (frames 1913-1923,
    # scores 12-15), but a ball hanging for 0.2 s is >= 10 of +-15 frames on one spot and was removed.
    n = 80
    H = np.diag([0.02, 0.02, 1.0])
    cands = [[BallCandidate(500.0, 500.0, 8.0), BallCandidate(100.0 + 10 * i, 300.0, 10.0)] for i in range(n)]
    for i in range(30, 42):  # 12 frames at the apex
        cands[i][1] = BallCandidate(400.0, 300.0, 10.0)
    ground = [H] * n
    at30 = drop_court_fixed(cands, ground)
    assert not any(c.y == 300.0 for c in at30[36])  # frame counts set at 30 fps: the apex ball is dropped
    at60 = drop_court_fixed(cands, ground, win=30, min_frames=20)  # the same 0.5 s / 0.33 s at 60 fps
    assert all(any(c.y == 300.0 for c in cs) for cs in at60)  # the ball survives everywhere
    assert all(not any((c.x, c.y) == (500.0, 500.0) for c in cs) for cs in at60)  # a painted mark still goes


def test_at_60_fps_scaled_serve_thresholds_find_a_near_serve_the_30_fps_frame_counts_miss():
    n = 140
    xy = np.full((n, 2), np.nan)
    for t in range(n):
        if t < 20:
            xy[t] = (1060, 640)
        elif t <= 75:
            if not 49 <= t <= 51:  # a 3-frame gap at the apex (kept by the scaled court-fixed filter)
                xy[t] = (1060 - 0.4 * (t - 20), 470 + 0.189 * (t - 50) ** 2)
        elif t <= 120:
            xy[t] = (1038 - 5 * (t - 75), 588 - 6 * (t - 75))  # 7.8 px per frame: 0.078 player heights
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    ball = BallTrack(xy, src.astype(float), src)
    assert detect_serves(ball, _players(n), 60.0, [(0, n)]) == []
    sv = detect_serves(ball, _players(n), 60.0, [(0, n)], frame_scale=2)
    assert len(sv) == 1 and sv[0].side == "near" and 72 <= sv[0].frame <= 78  # contact at 75
