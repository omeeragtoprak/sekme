"""Far contact (events/hits.py; adopted 2026-11-17, on by default): a far rally hit found at the outgoing ball's
image apex moves back to where a line through the incoming ball and a curve through the outgoing one meet, next to
the far player at racket height."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from courtvision import cli
from courtvision.events.hits import Hit, far_contact, move_far_contacts
from courtvision.perception.ball import BallTrack

FAR_BOX = np.array([600.0, 100.0, 640.0, 180.0])  # 40 x 80 px; racket height: at least 24 px above the bottom
SEG = [(0, 40)]  # one camera segment


def _ball(n: int, points: dict[int, tuple[float, float]]) -> BallTrack:
    xy = np.full((n, 2), np.nan)
    src = np.zeros(n, int)
    for i, p in points.items():
        xy[i], src[i] = p, 1
    return BallTrack(xy, src.astype(float), src)


def _players(n: int, box=FAR_BOX) -> dict:
    return {"far": SimpleNamespace(boxes=np.tile(box, (n, 1)))}


def _stroke(contact: int = 20, at: tuple[float, float] = (620.0, 140.0), gap: int = 2, last: int = 28,
            fps_scale: int = 1) -> dict[int, tuple[float, float]]:
    """In straight up the image at 6 px/frame, undetected for ``gap`` frames after the contact, out at (4, -1)
    px/frame (the far player's shot still rising in the image) up to ``last``; frame rates scale the speeds."""
    x0, y0 = at
    pts = {}
    for i in range(contact - 12 * fps_scale, contact):
        pts[i] = (x0, y0 + 6.0 / fps_scale * (contact - i))
    for i in range(contact + gap + 1, last + 1):
        pts[i] = (x0 + 4.0 / fps_scale * (i - contact), y0 - 1.0 / fps_scale * (i - contact))
    return pts


def test_the_contact_is_where_the_incoming_and_outgoing_lines_meet() -> None:
    got = far_contact(_ball(40, _stroke()), _players(40)["far"].boxes, 28, 30.0)
    assert got is not None
    frame, px = got
    assert frame == 20 and np.allclose(px, [620.0, 140.0], atol=0.5)


def test_a_straight_ball_a_turn_at_the_feet_or_away_from_the_player_stays() -> None:
    straight = {i: (620.0 + 4 * i, 140.0 - i) for i in range(10, 29)}
    boxes = _players(40)["far"].boxes
    assert far_contact(_ball(40, straight), boxes, 28, 30.0) is None
    assert far_contact(_ball(40, _stroke(at=(620.0, 170.0))), boxes, 28, 30.0) is None  # a bounce at the feet
    assert far_contact(_ball(40, _stroke(at=(200.0, 600.0))), boxes, 28, 30.0) is None  # nowhere near the box
    assert far_contact(_ball(40, _stroke()), np.full((40, 4), np.nan), 28, 30.0) is None  # no far player


def test_a_shallow_turn_is_not_a_contact() -> None:
    # Out at (1, -5) after in at (0, -6): an 11 degree turn, below the 35 degree minimum.
    pts = {i: (620.0, 140.0 + 6.0 * (20 - i)) for i in range(8, 20)}
    pts.update({i: (620.0 + (i - 20), 140.0 - 5.0 * (i - 20)) for i in range(23, 29)})
    assert far_contact(_ball(40, pts), _players(40)["far"].boxes, 28, 30.0) is None


def test_the_line_needs_two_detections_and_the_curve_three_at_30_fps_four_and_six_at_60() -> None:
    boxes = _players(80)["far"].boxes
    few = {i: p for i, p in _stroke().items() if 18 <= i <= 24}  # 2 in + 2 out
    assert far_contact(_ball(40, few), boxes[:40], 24, 30.0) is None
    few[25] = _stroke()[25]
    assert far_contact(_ball(40, few), boxes[:40], 25, 30.0)[0] == 20
    s60 = _stroke(contact=40, gap=4, last=56, fps_scale=2)
    nine = {i: p for i, p in s60.items() if 36 <= i <= 49}  # 4 in + 5 out
    assert far_contact(_ball(80, nine), boxes, 49, 60.0) is None
    nine[50] = s60[50]
    assert far_contact(_ball(80, nine), boxes, 50, 60.0)[0] == 40


def test_the_outgoing_ball_may_pass_its_image_apex_at_the_hit() -> None:
    # Out at (3, -3) px/frame and slowing in y: the image apex (the reversal hit) is at frame 28, the contact at 20,
    # and the ball is not detected in between until frame 25.
    pts = {i: (620.0, 140.0 + 6.0 * (20 - i)) for i in range(8, 20)}
    pts.update({i: (620.0 + 3 * (i - 20), 140.0 - 3 * (i - 20) + 0.1875 * (i - 20) ** 2) for i in range(25, 32)})
    got = far_contact(_ball(40, pts), _players(40)["far"].boxes, 28, 30.0)
    assert got is not None and got[0] == 20 and np.allclose(got[1], [620.0, 140.0], atol=0.5)
    past = {i: p for i, p in pts.items() if i < 20 or i >= 28}  # even from the apex on, the curve leads back to it
    assert far_contact(_ball(40, past), _players(40)["far"].boxes, 28, 30.0)[0] == 20


def test_move_far_contacts_moves_far_rally_hits_only_and_keeps_the_order() -> None:
    ball, players = _ball(40, _stroke()), _players(40)
    near = Hit(5, "near", np.zeros(2), True)
    far = Hit(28, "far", ball.xy[28].copy(), False)
    out = move_far_contacts([near, far], ball, players, 30.0, SEG)
    assert [(h.frame, h.side, h.near_player) for h in out] == [(5, "near", True), (20, "far", True)]
    serve = Hit(28, "far", ball.xy[28].copy(), False, serve=True, toss_start=10)
    assert move_far_contacts([serve], ball, players, 30.0, SEG) == [serve]  # a serve's contact comes from the toss
    near_only = [Hit(28, "near", ball.xy[28].copy(), False)]
    assert move_far_contacts(near_only, ball, players, 30.0, SEG) == near_only
    assert move_far_contacts([far], ball, {}, 30.0, SEG) == [far]


def test_a_contact_never_moves_onto_or_before_the_previous_hit() -> None:
    ball, players = _ball(40, _stroke()), _players(40)
    far = Hit(28, "far", ball.xy[28].copy(), False)
    for prev in (20, 22):
        hits = [Hit(prev, "near", np.zeros(2), True), far]
        assert move_far_contacts(hits, ball, players, 30.0, SEG) == hits
    assert [h.frame for h in move_far_contacts([Hit(19, "near", np.zeros(2), True), far], ball, players, 30.0, SEG)] \
        == [19, 20]


def test_25_fps_uses_a_ten_frame_window_and_two_points_a_side() -> None:
    assert far_contact(_ball(40, _stroke()), _players(40)["far"].boxes, 28, 25.0)[0] == 20  # 18-19 in, 23-28 out


def test_the_search_stays_inside_the_hits_camera_segment() -> None:
    ball, players = _ball(40, _stroke()), _players(40)
    far = Hit(28, "far", ball.xy[28].copy(), False)
    assert far_contact(ball, players["far"].boxes, 28, 30.0, start=21) is None  # only the outgoing line is left
    assert move_far_contacts([far], ball, players, 30.0, [(0, 21), (21, 40)]) == [far]
    assert [h.frame for h in move_far_contacts([far], ball, players, 30.0, [(0, 40)])] == [20]
    assert move_far_contacts([far], ball, players, 30.0, [(0, 25)]) == [far]  # in no segment: left as found


def test_the_still_ball_floor_is_30_px_per_second() -> None:
    # Out at (0.6, -0.3) px/frame, 0.67 px/frame: 40 px/s at 60 fps (moving), 20 px/s at 30 fps (about still).
    pts = {i: (620.0, 140.0 + 3.0 * (40 - i)) for i in range(16, 40)}
    pts.update({i: (620.0 + 0.6 * (i - 40), 140.0 - 0.3 * (i - 40)) for i in range(45, 57)})
    boxes = _players(80)["far"].boxes
    assert far_contact(_ball(80, pts), boxes, 56, 60.0)[0] == 40

    def at_30(speed: float) -> dict[int, tuple[float, float]]:  # out along (2, -1) at ``speed`` px/frame
        out = {i: (620.0, 140.0 + 6.0 * (20 - i)) for i in range(8, 20)}
        d = np.array([2.0, -1.0]) / np.sqrt(5.0) * speed
        out.update({i: (620.0 + d[0] * (i - 20), 140.0 + d[1] * (i - 20)) for i in range(23, 29)})
        return out
    assert far_contact(_ball(80, at_30(0.67)), boxes, 28, 30.0) is None  # 20 px/s
    assert far_contact(_ball(80, at_30(1.2)), boxes, 28, 30.0)[0] == 20  # 36 px/s


def test_no_far_box_at_the_contact_frame_gives_none() -> None:
    boxes = _players(40)["far"].boxes.copy()
    boxes[20] = np.nan
    assert far_contact(_ball(40, _stroke()), boxes, 28, 30.0) is None


def test_video_faster_than_30_fps_is_left_as_it_is() -> None:
    s60 = _stroke(contact=40, gap=4, last=56, fps_scale=2)
    ball, players = _ball(80, s60), _players(80)
    far = Hit(56, "far", ball.xy[56].copy(), False)
    assert far_contact(ball, players["far"].boxes, 56, 59.94)[0] == 40  # the estimate itself works at 60 fps
    assert move_far_contacts([far], ball, players, 59.94, [(0, 80)]) == [far]  # but nothing moves there
    b30 = _ball(40, _stroke())
    hit30 = Hit(28, "far", b30.xy[28].copy(), False)
    assert [h.frame for h in move_far_contacts([hit30], b30, _players(40), 29.97, SEG)] == [20]  # 29.97 is 30


def test_far_contact_fit_is_on_by_default_and_has_no_switch_left() -> None:
    from courtvision.pipeline import PipelineConfig
    assert PipelineConfig().far_contact_fit is True  # adopted 2026-11-17 (docs/EVAL_SPOT.md, test slot 2)
    with pytest.raises(SystemExit):
        cli.main(["analyze", "v.mp4", "--far-contact-fit"])
