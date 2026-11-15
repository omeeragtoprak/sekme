"""Line calls at their edges (officiating/line_call.py): the ball's footprint on a line, singles and doubles, both
halves, service boxes, the decisive threshold, seeds, degenerate inputs and play evidence. Values checked by running
the code."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.geometry.court import Half, ZoneKind, rally_zone, service_box
from courtvision.officiating.line_call import call_bounce, with_play_evidence

T = np.diag([1e-6] * 2)  # 1 mm: decisive


def P(x: float, y: float) -> np.ndarray:  # noqa: N802 - a point, as in the formulas
    return np.array([x, y])


def test_a_ball_touching_the_line_is_in() -> None:
    z = rally_zone(Half.FAR)
    assert call_bounce(P(4.12, 6), T, z).verdict == "IN"  # within the 2 cm footprint beyond the line
    assert call_bounce(P(4.14, 6), T, z).verdict == "OUT"
    assert call_bounce(P(4.12, 6), T, z, footprint=0).verdict == "OUT"


def test_doubles_and_singles_lines() -> None:
    assert call_bounce(P(5.5, 6), T, rally_zone(Half.FAR, ZoneKind.DOUBLES)).verdict == "IN"
    assert call_bounce(P(5.5, 6), T, rally_zone(Half.FAR)).verdict == "OUT"


def test_the_near_half() -> None:
    zn = rally_zone(Half.NEAR)
    assert call_bounce(P(-4.12, -11.9), T, zn).verdict == "IN"
    out = call_bounce(P(0, -11.93), T, zn)
    assert out.verdict == "OUT" and out.line == "near baseline"


def test_service_boxes() -> None:
    sb = service_box(Half.FAR, 2.0)
    assert call_bounce(P(-2, 6.41), T, sb).verdict == "IN"  # on the service line
    assert call_bounce(P(-2, 6.43), T, sb).verdict == "OUT"
    assert call_bounce(P(4.0, 3), T, sb).verdict == "OUT"  # the other box
    assert call_bounce(P(0.0451, 3), T, sb).verdict == "REVIEW"  # half a footprint over the centre line, 1 mm σ


def test_the_decisive_threshold() -> None:
    z, c = rally_zone(Half.FAR), np.diag([0.03 ** 2] * 2)
    assert call_bounce(P(4.09, 6), c, z, decisive=0.9).verdict == "IN"  # P(in) about 0.93
    assert call_bounce(P(4.09, 6), c, z).verdict == "REVIEW"
    assert call_bounce(P(4.18, 6), c, z, decisive=0.9).verdict == "OUT"  # P(in) about 0.07


def test_a_seed_makes_a_call_reproducible() -> None:
    z, c = rally_zone(Half.FAR), np.diag([0.05 ** 2] * 2)
    a = call_bounce(P(4.12, 6), c, z, seed=1)
    assert a == call_bounce(P(4.12, 6), c, z, seed=1)
    assert a.p_in != call_bounce(P(4.12, 6), c, z, seed=2).p_in


def test_degenerate_inputs() -> None:
    z = rally_zone(Half.FAR)
    r = call_bounce(P(0, 8), np.zeros((2, 2)), z)  # no uncertainty: clipped to 0.1 mm
    assert r.verdict == "IN" and r.sigma_m == pytest.approx(1e-4)
    with pytest.raises(FloatingPointError):
        call_bounce(P(0, 8), np.full((2, 2), np.nan), z)
    with pytest.raises(FloatingPointError):
        call_bounce(P(np.nan, 8), T, z)


def test_play_continuing_only_questions_an_out() -> None:
    z = rally_zone(Half.FAR)
    out = call_bounce(P(0, 12), T, z)
    r = with_play_evidence(out, True)
    assert r.verdict == "REVIEW" and r.conflict and r.p_in == out.p_in
    assert with_play_evidence(out, None).verdict == with_play_evidence(out, False).verdict == "OUT"
    inside = call_bounce(P(0, 8), T, z)
    assert with_play_evidence(inside, True) is inside
