"""Hit and serve detection at their edges (events/hits.py, events/serves.py): segment edges, close hits at 30 and 60
fps, the hit's side, the dead-ball window, tosses without a serve, toss runs and merging serves with hits. Values
checked by running the code (gaps found by a test review)."""

from __future__ import annotations

import numpy as np

from courtvision.events.hits import Hit, detect_hits, detect_hits_in_segments
from courtvision.events.serves import _toss_runs, detect_serves, merge_serves
from courtvision.perception.ball import BallTrack
from test_events_gates import FPS, _far_serve_rally, _players


def _bt(xy: np.ndarray) -> BallTrack:
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    return BallTrack(xy, src.astype(float), src)


def _zig(n: int, turns: list[int], v: float = 15.0, y0: float = 300.0, x: float = 600.0) -> BallTrack:
    """Ball moving in image y at v px/frame (down first), direction flipped at each frame in turns."""
    y, cur, d = np.empty(n), y0, 1
    for t in range(n):
        d = -d if t in turns else d
        cur += d * v
        y[t] = cur
    return _bt(np.stack([np.full(n, x), y], 1))


def _frames(hits: list[Hit]) -> list[int]:
    return [h.frame for h in hits]


def test_a_hit_within_a_few_frames_of_a_segment_edge_is_lost_and_segments_do_not_leak() -> None:
    n, P = 100, _players(100)
    ball = _zig(n, [40])  # reversal peak at frame 39
    assert _frames(detect_hits_in_segments(ball, P, FPS, 720, [(0, 100)])) == [39]
    assert _frames(detect_hits_in_segments(ball, P, FPS, 720, [(0, 46)])) == [39]
    assert _frames(detect_hits_in_segments(ball, P, FPS, 720, [(0, 45)])) == []  # direction window cut off
    assert _frames(detect_hits_in_segments(ball, P, FPS, 720, [(35, 100)])) == [39]
    assert _frames(detect_hits_in_segments(ball, P, FPS, 720, [(36, 100)])) == []  # no state before it
    # the reversal sits exactly on the boundary: neither half sees it, nothing leaks across the mask
    assert detect_hits_in_segments(ball, P, FPS, 720, [(0, 40), (40, 100)]) == []
    two = _zig(n, [30, 70])
    assert [(h.frame, h.side) for h in detect_hits_in_segments(two, P, FPS, 720, [(0, 50), (50, 100)])] == [
        (29, "near"), (69, "far")]


def test_two_reversals_closer_than_0_4_s_give_one_hit_and_the_limit_scales_with_fps() -> None:
    n, P = 100, _players(100)
    assert _frames(detect_hits(_zig(n, [40, 50]), P, 30.0, 720)) == [39]  # 10 frames < 12
    assert _frames(detect_hits(_zig(n, [40, 52]), P, 30.0, 720)) == [39, 51]  # 12 frames kept
    # 20 frames apart: two hits at 30 fps (limit 12) but one at 60 fps (limit 24)
    assert _frames(detect_hits(_zig(n, [40, 60]), P, 30.0, 720)) == [39, 59]
    assert _frames(detect_hits(_zig(n, [40, 60]), P, 60.0, 720)) == [39]
    assert _frames(detect_hits(_zig(n, [40, 66]), P, 60.0, 720)) == [39, 65]


def test_hit_side_follows_the_direction_before_the_reversal_and_near_player_the_box() -> None:
    n, P = 100, _players(100)
    down = detect_hits(_zig(n, [40]), P, 30.0, 720)  # towards the camera, then away: the near player hit
    assert [(h.side, h.near_player) for h in down] == [("near", False)]  # ball at x=600, near box at x>=1000
    beside = detect_hits(_zig(n, [40], y0=50, x=1030), P, 30.0, 720)  # peak y=650 inside the near box
    assert [(h.frame, h.side, h.near_player) for h in beside] == [(39, "near", True)]
    up = detect_hits(_zig(n, [40], v=-15, y0=730, x=615), P, 30.0, 720)  # away, then back: far player
    assert [(h.frame, h.side, h.near_player) for h in up] == [(39, "far", True)]  # peak y=130 in far box
    assert up[0].px.tolist() == [615.0, 130.0]
    no_near = detect_hits(_zig(n, [40]), {"far": P["far"]}, 30.0, 720)  # side without a track: no box
    assert [(h.side, h.near_player) for h in no_near] == [("near", False)]


def _padded_far_serve(pad: int = 40, mod=None) -> BallTrack:
    xy = np.vstack([np.full((pad, 2), np.nan), _far_serve_rally().xy])
    if mod:
        mod(xy)
    return _bt(xy)


def test_dead_ball_window_edges_and_its_clipping_at_the_segment_start() -> None:
    n = 180  # toss starts at frame 52, window = frames 22..51 at 30 fps
    P = _players(n)

    def serve_frames(ball, segs=None, **kw):
        return [s.frame for s in detect_serves(ball, P, FPS, segs or [(0, n)], **kw)]

    def inplay(f):
        def m(xy):
            xy[f] = (300, 420)  # a ball in play, far from the server
        return m

    assert serve_frames(_padded_far_serve()) == [80]
    assert serve_frames(_padded_far_serve(mod=inplay(21))) == [80]  # just outside the window
    assert serve_frames(_padded_far_serve(mod=inplay(22))) == []  # first frame inside it
    assert serve_frames(_padded_far_serve(mod=inplay(22)), dead_ball_s=0.5) == [80]
    assert serve_frames(_padded_far_serve(mod=inplay(10)), dead_ball_s=2.0) == []
    # the check never looks before the segment: the same ball is ignored when the segment starts after it
    assert serve_frames(_padded_far_serve(mod=inplay(22)), [(25, n)]) == [80]


def test_a_segment_cut_through_the_toss_still_serves_but_the_point_starts_at_the_cut() -> None:
    n = 180
    P = _players(n)
    sv = detect_serves(_padded_far_serve(), P, FPS, [(0, 65), (65, n)])
    assert [(s.frame, s.toss_start) for s in sv] == [(80, 65)]  # toss_start is 52 when uncut


def test_toss_without_a_serve_direction_detections_and_feet_band() -> None:
    n = 180
    P = _players(n)

    def run(mod=None, players=None):
        return [s.frame for s in detect_serves(_padded_far_serve(mod=mod), players or P, FPS, [(0, n)])]

    def up(xy):  # struck ball goes up the image: a far serve goes down it
        for t in range(41, 59):
            xy[40 + t] = (606 + 16 * (t - 40), 67 - 6 * (t - 40))

    def after(k):  # only k detections after the toss run's end
        def m(xy):
            xy[40 + 41 + k:] = np.nan
        return m

    assert run() == [80]
    assert run(up) == []
    assert run(after(1)) == [] and run(after(2)) == []
    assert run(after(4)) == [80]
    assert run(players=_players(n, far_y=13.5)) == [80]  # feet band is inclusive at 13.5 m
    assert run(players=_players(n, far_y=13.6)) == []
    assert run(players=_players(n, far_y=10.8)) == []
    assert run(players=_players(n, far_y=-12.0)) == []  # a far server standing on the near baseline


def test_toss_runs_gap_run_length_and_speed_limits_scale_with_frame_scale() -> None:
    n, T = 60, _players(60)["far"]

    def hold(missing=(), upto=40):
        xy = np.full((n, 2), np.nan)
        xy[10:upto] = (603, 60)
        for m in missing:
            xy[m] = np.nan
        return _bt(xy)

    assert _toss_runs(hold(range(20, 22)), T, 0, n) == [(10, 39)]  # 2 missed frames are bridged
    assert _toss_runs(hold(range(20, 23)), T, 0, n) == [(10, 19), (23, 39)]  # 3 split the run
    assert _toss_runs(hold(range(20, 24)), T, 0, n, frame_scale=2) == [(10, 39)]  # 4 bridged at 60 fps
    assert _toss_runs(hold(range(20, 25)), T, 0, n, frame_scale=2) == [(10, 19), (25, 39)]
    assert _toss_runs(hold(upto=12), T, 0, n) == []  # two detections are no run
    assert _toss_runs(hold(upto=13), T, 0, n) == [(10, 12)]

    def moving(sp):
        xy = np.full((n, 2), np.nan)
        for t in range(10, 16):
            xy[t] = (603, 10 + sp * (t - 10))
        return _bt(xy)

    assert _toss_runs(moving(10.7), T, 0, n) == [(10, 15)]  # limit 0.15 * 72 = 10.8 px per frame
    assert _toss_runs(moving(10.8), T, 0, n) == []
    assert _toss_runs(moving(5.0), T, 0, n, frame_scale=2) == [(10, 15)]  # limit halves at 60 fps
    assert _toss_runs(moving(5.5), T, 0, n, frame_scale=2) == []


def test_merge_serves_removal_window_ordering_and_fps() -> None:
    px = np.zeros(2)
    sv = Hit(80, "far", px, True, serve=True, toss_start=52)

    def hh(f):
        return Hit(f, "near", px, False)

    def merged(hits, serves, fps=30.0):
        return [h.frame for h in merge_serves(hits, serves, fps)]

    # removed on [toss_start, contact + 0.4 s): 52..91 at 30 fps
    assert [merged([hh(f)], [sv]) for f in (51, 52, 91, 92)] == [[51, 80], [80], [80], [80, 92]]
    assert merged([hh(103), hh(104)], [sv], 60.0) == [80, 104]  # window is 24 frames at 60 fps
    out = merge_serves([hh(120), hh(10), hh(60)], [sv], FPS)
    assert [(h.frame, h.serve) for h in out] == [(10, False), (80, True), (120, False)]
    sv2 = Hit(150, "near", px, True, serve=True, toss_start=125)
    out = merge_serves([hh(60), hh(100), hh(130), hh(165), hh(170)], [sv2, sv], FPS)  # serves given unsorted
    assert [(h.frame, h.serve) for h in out] == [(80, True), (100, False), (150, True), (165, False),
                                                 (170, False)]
    assert merged([hh(60), hh(70)], [Hit(80, "far", px, True, serve=True, toss_start=None)]) == [60, 70, 80]
    assert merged([hh(30), hh(10)], []) == [10, 30] and merged([], []) == []
