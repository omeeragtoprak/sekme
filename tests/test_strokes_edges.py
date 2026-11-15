"""Stroke type and hitting hand at their edges (events/strokes.py): handedness times side, missing pose, box or
ball, the ambiguity threshold, the wrist fallback, overheads and volleys, serve hand and hand votes, and the
Turkish cue texts. Values checked by running the code (gaps found by a test review)."""

import numpy as np
import pytest

from courtvision.events.strokes import AMBIGUOUS_FRACTION, VOLLEY_MAX_Y_M, classify, serve_hand, vote_hand
from courtvision.perception.pose import L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, Pose

BOX = np.array([100.0, 100.0, 160.0, 250.0])  # 60 px wide; the hip centre of _pose() is x=130
BASE = {NOSE: (130, 110), L_SHOULDER: (120, 130), R_SHOULDER: (140, 130), L_HIP: (122, 180), R_HIP: (138, 180),
        L_WRIST: (118, 175), R_WRIST: (142, 175)}
IDX = {"nose": NOSE, "ls": L_SHOULDER, "rs": R_SHOULDER, "lh": L_HIP, "rh": R_HIP, "lw": L_WRIST, "rw": R_WRIST}


def _pose(drop: tuple[str, ...] = (), **pts: tuple[float, float]) -> Pose:
    kps = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    for k, v in BASE.items():
        kps[k], conf[k] = v, 0.9
    for name, v in pts.items():
        kps[IDX[name]] = v
    for name in drop:
        conf[IDX[name]] = 0.0
    return Pose(kps, conf)


RIGHT_OF_BODY = np.array([175.0, 170.0])
LEFT_OF_BODY = np.array([85.0, 170.0])


@pytest.mark.parametrize(
    ("side", "hand", "ball", "expected"),
    [
        ("near", "right", RIGHT_OF_BODY, "forehand"),
        ("near", "left", LEFT_OF_BODY, "forehand"),
        ("far", "right", LEFT_OF_BODY, "forehand"),
        ("far", "left", RIGHT_OF_BODY, "forehand"),
        ("far", "left", LEFT_OF_BODY, "backhand"),
        ("near", "right", LEFT_OF_BODY, "backhand"),
    ],
)
def test_handedness_and_side_flip_independently(side: str, hand: str, ball: np.ndarray, expected: str) -> None:
    s = classify(False, _pose(), ball, BOX, side, hand, True, 11.0)
    assert (s.kind, s.wing, s.confident) == (expected, expected, True)


def test_the_wrist_fallback_mirrors_for_the_far_left_hander() -> None:
    # Ball in line; left wrist 30 px image-right of the hips. Far player (sgn -1) and left hand (hs -1): forehand.
    s = classify(False, _pose(lw=(160, 170)), np.array([131.0, 170.0]), BOX, "far", "left", True, 11.0)
    assert (s.kind, s.confident, s.cue) == ("forehand", False, "vuran bileğin konumu")
    # Same wrist for the near right-hander's *left* wrist is ignored: only the hitting wrist counts.
    s = classify(False, _pose(lw=(160, 170), rw=(131, 170)), np.array([131.0, 170.0]), BOX, "near", "right", True,
                 -11.0)
    assert s.kind is None


def test_the_ambiguity_threshold_is_inclusive() -> None:
    assert AMBIGUOUS_FRACTION == 0.15
    at = np.array([130.0 + 0.15 * 60, 170.0])  # exactly 9 px = 0.15 of the box width
    just_under = np.array([130.0 + 8.9, 170.0])
    s = classify(False, _pose(), at, BOX, "near", "right", True, -11.0)
    assert (s.kind, s.confident) == ("forehand", True)
    assert classify(False, _pose(), just_under, BOX, "near", "right", True, -11.0).confident is False


def test_missing_inputs_give_an_undetermined_stroke_with_its_reason() -> None:
    ball = RIGHT_OF_BODY
    no_box = classify(False, None, ball, None, "near", "right", True, -11.0)
    assert (no_box.kind, no_box.wing, no_box.confident) == (None, None, False)
    assert no_box.cue == "top ya da gövde görülmedi"
    # No ball but a pose: the hitting wrist alone decides, never confidently.
    s = classify(False, _pose(), None, BOX, "near", "right", True, -11.0)
    assert (s.kind, s.confident, s.cue) == ("forehand", False, "vuran bileğin konumu")
    assert classify(False, _pose(rw=(131, 170)), None, BOX, "near", "right", True, -11.0).cue == "top ya da gövde görülmedi"  # noqa: E501
    nan_ball = np.array([np.nan, 170.0])
    assert classify(False, _pose(rw=(131, 170)), nan_ball, BOX, "near", "right", True, -11.0).kind is None
    nan_box = np.array([100.0, 100.0, np.nan, 250.0])
    assert classify(False, None, ball, nan_box, "near", "right", True, -11.0).kind is None
    # A zero-width box cannot give a fraction.
    flat = np.array([100.0, 100.0, 100.0, 250.0])
    assert classify(False, _pose(rw=(131, 170)), ball, flat, "near", "right", True, -11.0).kind is None


def test_without_a_pose_the_box_centre_and_ball_still_decide() -> None:
    # Box centre x=130; ball 45 px to its image right; near right-hander.
    s = classify(False, None, RIGHT_OF_BODY, BOX, "near", "right", True, -11.0)
    assert (s.kind, s.confident, s.cue) == ("forehand", True, "topun gövdeye göre tarafı")
    # Ball in line and no pose: no wrist to fall back on.
    assert classify(False, None, np.array([131.0, 170.0]), BOX, "near", "right", True, -11.0).kind is None


def test_hips_missing_falls_back_to_shoulders_then_box() -> None:
    # Shoulder centre is also x=130, so the result is the same as with hips.
    s = classify(False, _pose(drop=("lh", "rh")), RIGHT_OF_BODY, BOX, "near", "right", True, -11.0)
    assert s.kind == "forehand"
    # One hip only: that hip (x=138) is the centre, so a ball at 140 is in line, not a forehand.
    s = classify(False, _pose(drop=("lh",)), np.array([140.0, 170.0]), BOX, "near", "right", True, -11.0)
    assert (s.kind, s.confident) == (None, False)


def test_a_missing_hitting_wrist_blocks_the_wrist_fallback_and_the_overhead() -> None:
    in_line = np.array([131.0, 170.0])
    assert classify(False, _pose(drop=("rw",), rw=(165, 170)), in_line, BOX, "near", "right", True, -11.0).kind is None
    high = np.array([150.0, 90.0])
    s = classify(False, _pose(drop=("rw",), rw=(150, 95)), high, BOX, "near", "right", True, -9.0)
    assert s.kind != "overhead"
    s = classify(False, _pose(drop=("nose",), rw=(150, 95)), high, BOX, "near", "right", True, -9.0)
    assert s.kind != "overhead"


def test_overhead_uses_the_hitting_hand_wrist_and_beats_a_volley() -> None:
    high = np.array([150.0, 90.0])
    # Left-hander: only the left wrist raised -> overhead; the same pose for a right-hander is not.
    left_up = _pose(lw=(115, 95))
    s = classify(False, left_up, high, BOX, "near", "left", False, -5.0)
    assert (s.kind, s.wing, s.confident, s.cue) == ("overhead", None, True, "bilek ve top başın üstünde")
    assert classify(False, left_up, high, BOX, "near", "right", False, -5.0).kind == "volley"


def test_the_volley_conditions_each_matter() -> None:
    ball = RIGHT_OF_BODY
    args = (False, _pose(), ball, BOX, "near", "right")
    assert classify(*args, None, -5.0).kind == "forehand"  # bounce unknown is not a volley
    assert classify(*args, True, -5.0).kind == "forehand"  # groundstroke
    assert classify(*args, False, None).kind == "forehand"  # no court position
    assert VOLLEY_MAX_Y_M == 7.0
    assert classify(*args, False, 7.0).kind == "forehand"  # limit is exclusive
    assert classify(*args, False, 6.99).kind == "volley"
    far = classify(False, _pose(), LEFT_OF_BODY, BOX, "far", "right", False, 5.0)  # far side has +y
    assert (far.kind, far.wing) == ("volley", "forehand")


def test_a_volley_keeps_the_wing_confidence_and_may_have_no_wing() -> None:
    v = classify(False, _pose(rw=(131, 170)), np.array([131.0, 170.0]), BOX, "near", "right", False, -5.0)
    assert (v.kind, v.wing, v.confident) == ("volley", None, False)
    assert v.cue == "gelen top sekmeden, fileye yakın vuruldu"


def test_a_serve_ignores_every_other_input_and_the_serve_cue_is_fixed() -> None:
    s = classify(True, _pose(), RIGHT_OF_BODY, BOX, "far", "left", False, 3.0)
    assert (s.kind, s.wing, s.confident, s.cue) == ("serve", None, True, "servis tespiti")


def test_groundstroke_cue_names_the_deciding_evidence() -> None:
    s = classify(False, _pose(), RIGHT_OF_BODY, BOX, "near", "right", True, -11.0)
    assert s.cue == "topun gövdeye göre tarafı"


def test_serve_hand_missing_data_and_thresholds() -> None:
    assert serve_hand(None) is None
    assert serve_hand(_pose(drop=("rw",), lw=(115, 60))) is None  # a missing wrist: cannot compare
    assert serve_hand(_pose(drop=("ls", "rs"), rw=(145, 60))) is None  # no shoulders
    # Torso 50 px (shoulders y=130, hips y=180): wrists must differ by 15 px.
    assert serve_hand(_pose(rw=(145, 100), lw=(115, 116))) == "right"  # 16 px apart
    assert serve_hand(_pose(rw=(145, 100), lw=(115, 114))) is None  # 14 px apart
    # A wrist level with the shoulders still counts as raised; both below do not.
    assert serve_hand(_pose(rw=(145, 130), lw=(115, 175))) == "right"
    assert serve_hand(_pose(rw=(145, 131), lw=(115, 175))) is None
    # Without hips the torso length defaults to 20 px (threshold 6 px).
    assert serve_hand(_pose(drop=("lh", "rh"), lw=(115, 100), rw=(145, 108))) == "left"
    assert serve_hand(_pose(drop=("lh", "rh"), lw=(115, 100), rw=(145, 104))) is None


def test_vote_hand_edges() -> None:
    assert vote_hand([]) is None
    assert vote_hand(["left"]) == "left" and vote_hand(["right"]) == "right"
    assert vote_hand(["left", "left", "right", "right"]) is None
    assert vote_hand(["left", "left", "right"]) == "left"
    assert vote_hand(["right"] * 3 + ["left"] * 2) == "right"
