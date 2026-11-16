"""Net clearance (events/flights.py): the net's height across the court and where a path crosses it. The
clearance's uncertainty is pinned in tests/test_events_gates.py; these pin the value itself."""

import numpy as np
import pytest

from courtvision.events.flights import _clearance_on_path, net_height
from courtvision.geometry.court import BALL_RADIUS, NET_HEIGHT_CENTER, NET_HEIGHT_POST, NET_POST_X


def test_the_net_is_lowest_at_the_centre_strap_rises_linearly_to_the_posts_and_is_flat_beyond() -> None:
    assert (NET_HEIGHT_CENTER, NET_HEIGHT_POST, BALL_RADIUS) == (0.914, 1.07, 0.0335)
    assert abs(NET_POST_X - 6.399) < 1e-9  # 0.914 m outside the doubles sideline
    assert net_height(0.0) == NET_HEIGHT_CENTER
    assert net_height(NET_POST_X / 2) == pytest.approx((NET_HEIGHT_CENTER + NET_HEIGHT_POST) / 2)
    assert net_height(3.2) == net_height(-3.2) == pytest.approx(0.992012, abs=1e-6)  # symmetric
    for x in (NET_POST_X, 7.5, -9.0):
        assert net_height(x) == NET_HEIGHT_POST


def test_clearance_is_the_balls_underside_above_the_tape_where_the_path_crosses_the_net_plane() -> None:
    # Crossing halfway between two samples: x = 1 m, z = 1.3 m.
    path = np.array([[1.0, -2.0, 1.5], [1.0, 2.0, 1.1]])
    assert _clearance_on_path(path) == pytest.approx(1.3 - BALL_RADIUS - net_height(1.0))
    # A sample exactly on the net plane is the crossing point itself.
    on = np.array([[0.0, -1.0, 1.0], [0.0, 0.0, 1.2], [0.0, 1.0, 1.4]])
    assert _clearance_on_path(on) == pytest.approx(1.2 - BALL_RADIUS - NET_HEIGHT_CENTER)
    # A ball under the tape has a negative clearance.
    low = np.array([[0.0, -1.0, 0.5], [0.0, 1.0, 0.5]])
    under = _clearance_on_path(low)
    assert under == pytest.approx(0.5 - BALL_RADIUS - NET_HEIGHT_CENTER) and under < 0


def test_a_path_that_never_crosses_has_no_clearance_and_only_the_first_crossing_counts() -> None:
    assert _clearance_on_path(np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 1.0]])) is None
    assert _clearance_on_path(np.array([[0.0, -1.0, 1.0], [0.0, -2.0, 1.0]])) is None
    back = np.array([[0.0, 3.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, 2.0]])  # crosses at z = 1, then again at 1.5
    assert _clearance_on_path(back) == pytest.approx(1.0 - BALL_RADIUS - NET_HEIGHT_CENTER)
