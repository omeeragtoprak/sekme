"""Candidate --far-stroke-breaks (events/hits.py): a break in the ball's 2D image velocity next to the far player is
a far stroke, added where no hit is within 8 frames at 30 fps."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from courtvision.events.hits import Hit, add_far_breaks, far_breaks
from courtvision.perception.ball import BallTrack

FAR_BOX = np.array([600.0, 100.0, 640.0, 180.0])  # 40 x 80 px; grown by 0.7 x 80 px around it


def _track(points: list[tuple[float, float]]) -> BallTrack:
    xy = np.array(points, float)
    return BallTrack(xy, np.ones(len(xy)), np.ones(len(xy), int))


def _players(n: int, box=FAR_BOX) -> dict:
    return {"far": SimpleNamespace(boxes=np.tile(box, (n, 1)))}


def _turn(at: int, n: int = 30, x0: float = 620.0, y0: float = 150.0) -> list[tuple[float, float]]:
    """In along +x at 10 px/frame, out along +y (towards the camera) after frame ``at``."""
    return [(x0 + 10 * (i - at), y0) if i <= at else (x0, y0 + 10 * (i - at)) for i in range(n)]


def test_a_sharp_turn_next_to_the_far_player_is_a_far_hit() -> None:
    hits = far_breaks(_track(_turn(15)), _players(30), 30.0, [(0, 30)])
    assert [h.frame for h in hits] == [15] and hits[0].side == "far" and hits[0].near_player


def test_a_straight_ball_a_turn_away_from_the_player_or_no_far_player_gives_none() -> None:
    straight = [(620.0 + 10 * i, 150.0) for i in range(30)]
    assert far_breaks(_track(straight), _players(30), 30.0, [(0, 30)]) == []
    away = _turn(15, x0=200.0, y0=600.0)  # far from the box
    assert far_breaks(_track(away), _players(30), 30.0, [(0, 30)]) == []
    assert far_breaks(_track(_turn(15)), {}, 30.0, [(0, 30)]) == []
    nan_box = _players(30, np.full(4, np.nan))
    assert far_breaks(_track(_turn(15)), nan_box, 30.0, [(0, 30)]) == []


def test_a_speed_jump_counts_and_the_segment_bounds_the_search() -> None:
    slow_fast = [(620.0 + (2 * i if i <= 15 else 30 + 12 * (i - 15)), 150.0) for i in range(30)]  # 2 -> 12 px/frame
    assert [h.frame for h in far_breaks(_track(slow_fast), _players(30), 30.0, [(0, 30)])] == [15]
    assert far_breaks(_track(_turn(15)), _players(30), 30.0, [(16, 30)]) == []  # the break is outside the segment


def test_breaks_join_only_where_no_hit_is_within_eight_frames() -> None:
    old = [Hit(10, "near", np.zeros(2), True)]
    breaks = [Hit(f, "far", np.zeros(2), True) for f in (18, 19, 40)]
    assert [h.frame for h in add_far_breaks(old, breaks, 30.0)] == [10, 19, 40]  # 18 is within 8 of 10
    assert [h.frame for h in add_far_breaks(old, breaks, 60.0)] == [10, 40]  # 16 frames at 60 fps
    assert add_far_breaks([], [], 30.0) == []


def test_the_window_is_a_tenth_of_a_second_rounded_up_so_25_fps_uses_three_frames() -> None:
    # A turn after 3 frames in and 3 out is found at 25 fps (round(2.5) would have given a 2-frame window).
    hits = far_breaks(_track(_turn(15)), _players(30), 25.0, [(0, 30)])
    assert [h.frame for h in hits] == [15]
