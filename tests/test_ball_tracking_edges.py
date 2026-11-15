"""Ball tracking at its edges (perception/ball.py): the gate's scale with frame width, how long a miss is bridged,
a wild pick removed, court marks (frame count, window, radius) and gap filling at the ends. Values checked by running
the code."""

from __future__ import annotations

import numpy as np

from courtvision.perception.ball import BallCandidate as B
from courtvision.perception.ball import _fill_gaps, drop_court_fixed, track_ball


def line(n: int) -> list[list[B]]:
    return [[B(100 + 10 * i, 400.0, 5.0)] for i in range(n)]


def test_the_gate_scales_with_the_frame_width() -> None:
    for width, jump, kept in ((640, 200, False), (1280, 200, True), (1280, 300, True), (1280, 301, False)):
        c = line(5)
        c[4] = [B(c[3][0].x + jump, 400.0, 5.0)]
        assert bool(track_ball(c, width).source[4] == 1) is kept, (width, jump)


def _after_miss(missed: int):
    c = line(16 + missed)
    for i in range(10, 10 + missed):
        c[i] = []
    c[10 + missed] = [B(700, 100, 12.0), c[10 + missed][0]]  # a brighter decoy far from the ball's path
    return track_ball(c, 1280)


def test_a_miss_of_up_to_five_frames_is_bridged() -> None:
    # Bridging keeps the track for max_gap - 1 = 5 missed frames, while gap filling fills up to max_gap = 6:
    # one frame apart. Changing either changes the tracker's output and needs its own rule.
    assert _after_miss(5).xy[15, 0] == 250.0  # the ball, by its predicted path
    assert _after_miss(6).xy[16, 0] == 700.0  # too long: the decoy


def test_a_wild_pick_is_dropped_and_filled_with_no_score() -> None:
    c = line(20)
    c[10] = [B(c[10][0].x + 100, 700, 200.0)]
    t = track_ball(c, 1280)
    assert t.source[10] == 2 and t.score[10] == 0.0 and abs(t.xy[10, 0] - 200.0) < 1e-6


def _court_mark_kept(frames: int) -> int:
    cs = [[B(100 + 20.0 * i, 300.0, 10.0)] for i in range(40)]
    for j in range(10, 10 + frames):
        cs[j].append(B(500, 500, 8.0))
    return len(drop_court_fixed(cs, [np.diag([0.02, 0.02, 1.0])] * 40)[10])


def test_a_court_mark_is_dropped_from_eleven_frames_on() -> None:
    assert (_court_mark_kept(10), _court_mark_kept(11)) == (2, 1)  # (the ball and the mark, the ball alone)


def _mark_with_others_at(offsets) -> bool:
    n, i0 = 60, 30
    cs = [[B(100 + 20.0 * i, 300.0, 10.0)] for i in range(n)]
    for o in [0, *offsets]:
        cs[i0 + o].append(B(500, 500, 8.0))
    return len(drop_court_fixed(cs, [np.diag([0.02, 0.02, 1.0])] * n)[i0]) == 2  # True: the mark was kept


def test_court_marks_are_counted_within_fifteen_frames_either_side() -> None:
    assert _mark_with_others_at(range(-15, -5)) is False and _mark_with_others_at(range(6, 16)) is False
    assert _mark_with_others_at(range(-16, -6)) is True and _mark_with_others_at(range(7, 17)) is True


def test_court_marks_within_a_quarter_metre() -> None:
    def kept(dx: float) -> bool:  # a mark in 18 frames, alternating between two spots dx px (0.02 m/px) apart
        cs = [[B(100 + 20.0 * i, 300.0, 10.0)] for i in range(40)]
        for j in range(10, 28):
            cs[j].append(B(500 + (dx if j % 2 else 0), 500, 8.0))
        return len(drop_court_fixed(cs, [np.diag([0.02, 0.02, 1.0])] * 40)[18]) == 2
    assert kept(12) is False and kept(13) is True  # 0.24 m: one spot seen 18 times; 0.26 m: two spots, 9 each


def test_gap_filling_leaves_the_ends_and_long_gaps() -> None:
    xy = np.array([[10.0 * i, 5.0 * i] for i in range(10)])
    xy[[0, 1, 8, 9]] = np.nan
    src = np.where(np.isfinite(xy[:, 0]), 1, 0)
    _, s = _fill_gaps(xy, src, 6)
    assert s.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  # nothing to fill from on one side
    gappy = np.array([[10.0 * i, 5.0 * i] for i in range(10)])
    gappy[4:6] = np.nan
    gsrc = np.where(np.isfinite(gappy[:, 0]), 1, 0)
    x, s = _fill_gaps(gappy, gsrc, 2)
    assert s[4:6].tolist() == [2, 2] and np.allclose(x[4:6], [[40, 20], [50, 25]])
    _, s = _fill_gaps(gappy, gsrc, 1)
    assert s[4:6].tolist() == [0, 0]
    x, s = _fill_gaps(np.full((5, 2), np.nan), np.zeros(5, int), 6)
    assert s.tolist() == [0] * 5
