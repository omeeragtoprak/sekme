"""Stroke type rules on synthetic poses (image pixels, y down)."""

import numpy as np

from courtvision.events.strokes import classify, serve_hand, vote_hand
from courtvision.perception.pose import L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, Pose

BOX = np.array([100.0, 100.0, 160.0, 250.0])  # 60 px wide player


def _pose(**pts: tuple[float, float]) -> Pose:
    kps = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    base = {NOSE: (130, 110), L_SHOULDER: (120, 130), R_SHOULDER: (140, 130), L_HIP: (122, 180), R_HIP: (138, 180),
            L_WRIST: (118, 175), R_WRIST: (142, 175)}
    names = {"nose": NOSE, "lw": L_WRIST, "rw": R_WRIST}
    for k, v in base.items():
        kps[k], conf[k] = v, 0.9
    for name, v in pts.items():
        kps[names[name]] = v
    return Pose(kps, conf)


def test_the_raised_wrist_at_a_serve_gives_the_hand() -> None:
    assert serve_hand(_pose(rw=(145, 60))) == "right"
    assert serve_hand(_pose(lw=(115, 60))) == "left"
    assert serve_hand(_pose()) is None  # no arm up: not a contact frame
    assert vote_hand(["right", "right", "left"]) == "right" and vote_hand(["right", "left"]) is None


def test_ball_side_is_read_in_the_players_own_frame() -> None:
    # Ball on image right of the hips: near player (back to camera) -> their right -> forehand for a right-hander;
    # far player (facing the camera) -> their left -> backhand.
    ball = np.array([175.0, 170.0])
    assert classify(False, _pose(), ball, BOX, "near", "right", True, -11.0).kind == "forehand"
    assert classify(False, _pose(), ball, BOX, "far", "right", True, 11.0).kind == "backhand"
    assert classify(False, _pose(), ball, BOX, "near", "left", True, -11.0).kind == "backhand"


def test_a_ball_in_line_with_the_body_falls_back_to_the_wrist_and_is_not_confident() -> None:
    s = classify(False, _pose(rw=(160, 170)), np.array([131.0, 170.0]), BOX, "near", "right", True, -11.0)
    assert s.kind == "forehand" and not s.confident
    in_line = np.array([131.0, 170.0])
    assert classify(False, _pose(rw=(131, 170)), in_line, BOX, "near", "right", True, -11.0).kind is None


def test_overhead_needs_both_the_wrist_and_the_ball_above_the_head() -> None:
    high = np.array([150.0, 90.0])
    assert classify(False, _pose(rw=(150, 95)), high, BOX, "near", "right", True, -9.0).kind == "overhead"
    # Follow-through: wrist up but the ball at hip height -> still a groundstroke.
    hip_height = np.array([175.0, 170.0])
    assert classify(False, _pose(rw=(150, 95)), hip_height, BOX, "near", "right", True, -9.0).kind == "forehand"


def test_volley_needs_no_incoming_bounce_and_a_position_near_the_net() -> None:
    ball = np.array([175.0, 160.0])
    v = classify(False, _pose(), ball, BOX, "near", "right", False, -5.0)
    assert v.kind == "volley" and v.wing == "forehand"
    assert classify(False, _pose(), ball, BOX, "near", "right", False, -11.5).kind == "forehand"
    assert classify(True, None, None, None, "near", "right", None, None).kind == "serve"
