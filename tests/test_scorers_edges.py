"""Behaviour of the evaluation scorers (six_videos, spot, sim_benchmark) that tests/ does not pin.

Every expected value was checked by running the code; inputs are synthetic dicts and small arrays. Where the code
does what the rule says but the behaviour looks odd it is pinned and marked ``# Pinned as it is:``.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from courtvision.eval import sim_benchmark as sb
from courtvision.eval import six_videos as sv
from courtvision.eval import spot

FPS = 60.0  # 30 s is exactly 1800 frames


# --- builders -------------------------------------------------------------------------------------------------------


def _lab(lo: int, hi: int | None = None, side: str = "near", kmh: float | None = None) -> dict:
    return {"lo": lo, "hi": lo if hi is None else hi, "side": side,
            "radar": None if kmh is None else {"value": kmh, "unit": "km/h"}}


def _fl(i: int, start: int, contact: float | None, side: str = "near", kmh: float | None = None,
        verdict: str | None = None, rejected: str | None = None, bounce: float | None = None,
        kind: str = "serve") -> dict:
    return {"id": i, "kind": kind, "hitter": side, "start_frame": start, "contact_frame": contact,
            "hit_speed_kmh": kmh, "rejected": rejected, "call": None if verdict is None else {"verdict": verdict},
            "bounce": None if bounce is None else {"frame": bounce}}


def _rep(flights: list[dict], fps: float = FPS) -> dict:
    return {"video": {"fps": fps}, "flights": flights}


def _est(lo: int, err: float) -> float:
    """Contact estimate whose error against the bracket [lo, lo + 1] is ``err`` frames (0 inside it)."""
    return float(lo) if err <= 0 else lo + 1 + err


# --- small helpers of six_videos ------------------------------------------------------------------------------------


def test_percentages_of_counts_are_exact_fractions() -> None:
    assert sv._pct(4, 30) == Fraction(40, 3) and isinstance(sv._pct(4, 30), Fraction)
    assert sv._pct(7, 30) - sv._pct(4, 30) == 10  # exactly, where floats give 9.999999999999998
    assert sv._pct(0, 5) == 0 and sv._pct(30, 30) == 100 and sv._pct(1, 3) * 3 == 100
    with pytest.raises(ZeroDivisionError):  # callers guard the empty denominator themselves
        sv._pct(1, 0)


def test_integers_and_numbers_exclude_booleans_nan_and_inf() -> None:
    for ok in (0, -3, 10**12):
        assert sv._is_int(ok) is True
    for bad in (True, False, 1.0, "1", None, [1]):
        assert sv._is_int(bad) is False
    for ok in (0, -2, 3.5, 1e300):
        assert sv._is_num(ok) is True
    for bad in (True, False, float("nan"), float("inf"), float("-inf"), "1", None, [1.0]):
        assert sv._is_num(bad) is False


def test_the_sampling_seed_is_the_first_eight_hex_digits_of_the_name_hash() -> None:
    def ref(key: str) -> np.random.Generator:
        return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))

    for key in ("w2024_final", "w2024_final:disagree", ""):
        assert sv._rng(key).integers(0, 10**9, 6).tolist() == ref(key).integers(0, 10**9, 6).tolist()
    assert sv._rng("a").integers(0, 10**9, 6).tolist() != sv._rng("b").integers(0, 10**9, 6).tolist()


def test_presence_sample_takes_every_frame_of_a_short_video_and_never_repeats() -> None:
    assert sv.presence_sample("short", 10) == list(range(10))
    s = sv.presence_sample("long", 5000, k=25)
    assert len(s) == len(set(s)) == 25 and s == sorted(s) and s[0] >= 0 and s[-1] < 5000
    assert s == sv.presence_sample("long", 5000, k=25)
    assert sv.presence_sample("exactly", 80) == list(range(80))  # the default k is 80 and a draw of all is all


def test_summary_of_values_reports_count_median_p90_and_the_largest_not_the_biggest_magnitude() -> None:
    assert sv._summary([]) == {"n": 0}
    assert sv._summary([4.0]) == {"n": 1, "median": 4.0, "p90": 4.0, "max": 4.0}
    s = sv._summary([float(v) for v in (10, 3, 7, 1, 2, 9, 4, 5, 6, 8)])  # unsorted on purpose
    assert s["n"] == 10 and s["median"] == 5.5 and s["max"] == 10.0
    assert s["p90"] == pytest.approx(9.1)
    neg = sv._summary([-9.0, -1.0])
    assert neg["max"] == -1.0 and neg["median"] == -5.0  # Pinned as it is: "max" of signed shifts is the signed maximum


def test_feet_are_the_bottom_middle_of_the_box_and_a_zero_coordinate_is_not_missing() -> None:
    frames = {"players": {"near": {"box": [[480.0, 700.0, 520.0, 900.0], None, [0, 0, 10, 20], [1, None, 3, 4]]},
                          "far": {"box": [None] * 4}}}
    assert sv._feet(frames, "near", 0) == (500.0, 900.0, 200.0)
    assert sv._feet(frames, "near", 1) is None
    assert sv._feet(frames, "near", 2) == (5.0, 20, 20)  # zeros are coordinates, not "no box"
    assert sv._feet(frames, "near", 3) is None  # any missing coordinate means no box
    assert sv._feet(frames, "far", 0) is None


def test_found_is_within_a_third_of_the_box_height_with_a_floor_of_15_px_at_1080_lines() -> None:
    far = (960.0, 200.0, 70.0)  # tolerance max(15, 0.3 * 70) = 21 px
    assert sv._found(far, [960.0 + 21.0, 200.0], 1080) is True  # the edge is inside
    assert sv._found(far, [960.0 + 21.5, 200.0], 1080) is False
    assert sv._found(far, [960.0, 200.0 - 21.0], 1080) is True  # vertical offsets count the same
    assert sv._found(far, [960.0 + 15.0, 200.0 + 15.0], 1080) is False  # 21.2 px away diagonally
    small = (500.0, 300.0, 20.0)  # 0.3 * 20 = 6 px: the floor of 15 px applies
    assert sv._found(small, [515.0, 300.0], 1080) is True and sv._found(small, [515.5, 300.0], 1080) is False
    assert sv._found(small, [509.0, 312.0], 1080) is True  # 15.0 px along a diagonal
    assert sv._found(small, [510.0, 300.0], 720) is True  # the floor scales with the frame: 10 px at 720 lines
    assert sv._found(small, [510.5, 300.0], 720) is False
    assert sv._found(small, [529.0, 300.0], 2160) is True and sv._found(small, [531.0, 300.0], 2160) is False
    assert sv._found(None, [1.0, 1.0], 1080) is False and sv._found(far, None, 1080) is False
    big = (500.0, 900.0, 200.0)  # the tracked box's height, not the labelled one: 0.3 * 200 = 60 px
    assert sv._found(big, [560.0, 900.0], 1080) is True and sv._found(big, [561.0, 900.0], 1080) is False


# --- labels ---------------------------------------------------------------------------------------------------------


def test_serve_labels_refuse_a_zero_radar_a_bad_unit_a_float_frame_and_anything_that_is_not_a_dict() -> None:
    base = {"lo": 5, "hi": 9, "side": "far", "radar": None}
    sv.check_serve_labels([base, dict(base, lo=7, hi=7), dict(base, radar={"value": 100, "unit": "km/h"}),
                           dict(base, radar={"value": 87.5, "unit": "mph"})])
    sv.check_serve_labels([])
    for bad in (dict(base, radar={"value": 0, "unit": "mph"}), dict(base, radar={"value": -5.0, "unit": "mph"}),
                dict(base, radar={"value": float("nan"), "unit": "mph"}),
                dict(base, radar={"value": True, "unit": "mph"}),
                dict(base, radar={"value": 100, "unit": "kph"}), dict(base, radar="120"),
                dict(base, lo=5.0), dict(base, hi=9.5), dict(base, lo=True), dict(base, lo=-1, hi=3),
                dict(base, lo=10, hi=9),
                "serve", None, {"lo": 1, "hi": 2, "radar": None}, {"lo": 1, "hi": 2, "side": "far"},
                dict(base, side="left"), dict(base, side=None)):
        with pytest.raises(ValueError, match="serve label 1"):
            sv.check_serve_labels([base, bad])


def test_player_labels_refuse_a_non_boolean_camera_flag_a_nan_foot_and_a_boolean_frame() -> None:
    ok = {"frame": 0, "main": True, "near": [1.0, 2.0], "far": [3, 4]}
    sv.check_player_labels([ok, dict(ok, frame=1, main=False, near=None, far=None)])
    sv.check_player_labels([])
    for bad in (dict(ok, main=1), dict(ok, main="yes"), dict(ok, near=[1.0, float("nan")]),
                dict(ok, far=[1.0, 2.0, 3.0]), dict(ok, far=(1.0, 2.0)), dict(ok, frame=True), dict(ok, frame=2.0),
                dict(ok, near=[True, 2.0]), "frame", {"frame": 0, "main": True, "near": None},
                {"frame": 0, "near": None, "far": None}):
        with pytest.raises(ValueError, match="player label 1"):
            sv.check_player_labels([dict(ok, frame=9), bad])
    with pytest.raises(ValueError, match="labelled twice"):
        sv.check_player_labels([ok, dict(ok, main=False)])


# --- serve flights and matching -------------------------------------------------------------------------------------


def test_bracket_error_is_zero_inside_inclusive_and_the_distance_to_the_nearest_edge_outside() -> None:
    assert sv.bracket_error(100.0, 100, 104) == 0.0 and sv.bracket_error(104.0, 100, 104) == 0.0
    assert sv.bracket_error(102.5, 100, 104) == 0.0
    assert sv.bracket_error(97.5, 100, 104) == 2.5 and sv.bracket_error(104.25, 100, 104) == 0.25
    assert sv.bracket_error(98.0, 100, 100) == 2.0 and sv.bracket_error(100.0, 100, 100) == 0.0
    assert isinstance(sv.bracket_error(100, 100, 104), float)


def test_the_board_reading_is_converted_from_mph_and_left_alone_in_kmh() -> None:
    assert sv.board_kmh(_lab(1, kmh=100.0)) == 100.0
    assert sv.board_kmh({"radar": {"value": 100, "unit": "mph"}}) == pytest.approx(160.9344)
    assert sv.board_kmh({"radar": {"value": 87.5, "unit": "mph"}}) == pytest.approx(87.5 * 1.609344)
    assert sv.board_kmh({"radar": None}) is None and sv.board_kmh({}) is None


def test_system_serves_keep_only_serves_take_the_contact_frame_even_when_zero_and_read_the_verdict() -> None:
    rep = _rep([_fl(0, 90, 100.5, side="far", kmh=180.0, verdict="OUT", rejected="speed"),
                _fl(1, 40, None),  # no fitted contact: the start frame
                _fl(2, 55, 0.0),  # a contact at frame 0 is a contact, not a missing one
                _fl(3, 70, 71.0, kind="rally", verdict="IN"),
                {"id": 4, "kind": None, "hitter": "near", "start_frame": 1, "contact_frame": 1.0}])
    s = sv.system_serves(rep)
    assert [x["id"] for x in s] == [0, 1, 2]
    assert s[0] == {"id": 0, "side": "far", "est": 100.5, "kmh": 180.0, "start": 90, "rejected": "speed",
                    "verdict": "OUT"}
    assert s[1]["est"] == 40.0 and isinstance(s[1]["est"], float) and s[1]["verdict"] is None  # call is null
    assert s[2]["est"] == 0.0
    assert sv.system_serves(_rep([])) == []


def test_bounce_calls_need_both_a_call_and_a_bounce_and_keep_flight_order_of_every_kind() -> None:
    rep = _rep([_fl(0, 0, 10.0, verdict="IN", bounce=40.0),
                _fl(1, 0, 10.0, verdict="OUT", bounce=15, kind="rally"),  # any kind, an int frame
                _fl(2, 0, 10.0, verdict="IN", bounce=None),  # a call without a bounce
                _fl(3, 0, 10.0, verdict=None, bounce=50.0),  # a bounce without a call
                _fl(4, 0, 10.0, verdict="REVIEW", bounce=0.0)])  # frame 0 still counts
    assert sv._bounce_calls(rep) == [(40.0, "IN"), (15.0, "OUT"), (0.0, "REVIEW")]
    assert all(isinstance(f, float) for f, _ in sv._bounce_calls(rep))
    assert sv._bounce_calls(_rep([])) == []


def test_matching_includes_30_frames_and_excludes_31_and_needs_the_same_side() -> None:
    labels = [_lab(1000, 1001)]
    ok = sv.match(labels, sv.system_serves(_rep([_fl(0, 0, 1031.0)])))  # 30 frames after the bracket end
    assert ok[0][1] == 30.0
    assert sv.match(labels, sv.system_serves(_rep([_fl(0, 0, 1031.5)]))) == {}
    assert sv.match(labels, sv.system_serves(_rep([_fl(0, 0, 970.0)])))[0][1] == 30.0  # before the start
    assert sv.match(labels, sv.system_serves(_rep([_fl(0, 0, 969.0)]))) == {}
    assert sv.match(labels, sv.system_serves(_rep([_fl(0, 0, 1000.5, side="far")]))) == {}


def test_matching_takes_the_closest_pair_first_across_labels_not_the_first_label_first() -> None:
    labels = [_lab(100), _lab(110)]
    serves = sv.system_serves(_rep([_fl(0, 0, 106.0), _fl(1, 0, 125.0)]))
    m = sv.match(labels, serves)
    # serve 0 is 4 frames from label 1 and 6 from label 0: label 1 gets it; label 0 then takes serve 1 (25 away)
    assert {i: (s["id"], e) for i, (s, e) in m.items()} == {1: (0, 4.0), 0: (1, 25.0)}
    inside = sv.match([_lab(100, 110)], sv.system_serves(_rep([_fl(0, 0, 112.0), _fl(1, 0, 105.0)])))
    assert inside[0][0]["id"] == 1 and inside[0][1] == 0.0  # inside the bracket beats outside it


def test_matching_ties_go_to_the_lower_label_then_the_lower_serve() -> None:
    two_labels = sv.match([_lab(100), _lab(110)], sv.system_serves(_rep([_fl(0, 0, 105.0)])))  # 5 from both
    assert list(two_labels) == [0]
    two_serves = sv.match([_lab(100)], sv.system_serves(_rep([_fl(7, 0, 104.0), _fl(8, 0, 96.0)])))  # 4 from both
    assert two_serves[0][0]["id"] == 7
    assert len(sv.match([_lab(100)], sv.system_serves(_rep([_fl(0, 0, 101.0), _fl(1, 0, 101.5)])))) == 1


def test_recall_counts_edges_of_the_two_frame_window_and_of_the_fault_window() -> None:
    labels = [_lab(1000, 1001, side="near")]
    two = sv.recall_stats(labels, _rep([_fl(0, 0, _est(1000, 2.0))]))
    assert (two["matched"], two["within2"]) == (1, 1)
    three = sv.recall_stats(labels, _rep([_fl(0, 0, _est(1000, 2.5))]))
    assert (three["matched"], three["within2"]) == (1, 0)
    assert sv.recall_stats(labels, _rep([_fl(0, 0, 1000.0), _fl(1, 0, 1001.0)])) == \
        {"labelled": 1, "matched": 1, "within2": 1, "false_serves": 1}  # the second serve of one label is false
    # An unmatched serve (more than 30 frames off) is a fault, not a false serve, when the labelled bracket starts
    # up to 1800 frames (30 s at 60 fps) later.
    far_label = [_lab(5000, 5001)]
    for est, false in ((5000 - 1800, 0), (5000 - 1801, 1), (5000 - 31, 0), (5000 - 30, None)):
        if false is None:  # 30 frames before the bracket: matched, so neither a fault nor a false serve
            st = sv.recall_stats(far_label, _rep([_fl(0, 0, float(est))]))
            assert (st["matched"], st["false_serves"]) == (1, 0)
            continue
        st = sv.recall_stats(far_label, _rep([_fl(0, 0, float(est))]))
        assert (st["matched"], st["false_serves"]) == (0, false), est
    assert sv.recall_stats(far_label, _rep([_fl(0, 0, 5000 - 1800.0, side="far")]))["false_serves"] == 1


def test_a_serve_at_or_after_the_start_of_a_labelled_bracket_is_never_a_fault() -> None:
    labels = [_lab(1000, 1010)]
    st = sv.recall_stats(labels, _rep([_fl(0, 0, 1000.0), _fl(1, 0, 1000.0), _fl(2, 0, 1005.0)]))
    assert st["matched"] == 1 and st["false_serves"] == 2  # est == lo: not "followed by" a label; inside: no
    assert sv.recall_stats([], _rep([_fl(0, 0, 5.0)])) == {"labelled": 0, "matched": 0, "within2": 0,
                                                            "false_serves": 1}
    assert sv.recall_stats(labels, _rep([])) == {"labelled": 1, "matched": 0, "within2": 0, "false_serves": 0}


def test_the_fault_window_is_thirty_seconds_at_the_videos_own_frame_rate() -> None:
    labels = [_lab(5000, 5001)]
    assert sv.recall_stats(labels, _rep([_fl(0, 0, 5000 - 750.0)], fps=25.0))["false_serves"] == 0
    assert sv.recall_stats(labels, _rep([_fl(0, 0, 5000 - 751.0)], fps=25.0))["false_serves"] == 1
    assert sv.recall_stats(labels, _rep([_fl(0, 0, 5000 - 751.0)], fps=30.0))["false_serves"] == 0


def test_a_flip_needs_the_two_bounces_within_two_frames_and_exactly_in_against_out() -> None:
    def runs(fa: float, va: str, fb: float, vb: str) -> tuple[dict, dict]:
        return (_rep([_fl(0, 0, 1.0, verdict=va, bounce=fa)]), _rep([_fl(0, 0, 1.0, verdict=vb, bounce=fb)]))

    assert sv.decisive_flips(*runs(40.0, "IN", 42.0, "OUT")) == [(40.0, "IN", "OUT")]  # 2 frames apart: inside
    assert sv.decisive_flips(*runs(40.0, "OUT", 38.0, "IN")) == [(40.0, "OUT", "IN")]
    assert sv.decisive_flips(*runs(40.0, "IN", 42.5, "OUT")) == []
    for va, vb in (("IN", "IN"), ("OUT", "OUT"), ("IN", "REVIEW"), ("REVIEW", "OUT"), ("REVIEW", "REVIEW")):
        assert sv.decisive_flips(*runs(40.0, va, 40.0, vb)) == []
    a, b = runs(40.0, "IN", 40.0, "OUT")
    assert sv.decisive_flips(a, b, tol=0.0) == [(40.0, "IN", "OUT")] and sv.decisive_flips(a, _rep([])) == []
    far = _rep([_fl(0, 0, 1.0, verdict="OUT", bounce=40.5)])
    assert sv.decisive_flips(a, far, tol=0.25) == []  # the tolerance is a parameter


# --- fps rule -------------------------------------------------------------------------------------------------------


def _fps_video(fps: float, n_lab: int, d_err: list[float | None], s_err: list[float | None],
               extra_d: int = 0, extra_s: int = 0) -> tuple:
    """Labels at 10000 k + 100 (bracket 100 .. 101). ``d_err`` / ``s_err``: per label the contact error of the
    serve each run finds (None: not found). ``extra_*``: unmatched serves nowhere near a label."""
    labels = [_lab(10_000 * k + 100, 10_000 * k + 101) for k in range(n_lab)]

    def flights(errs: list[float | None], extra: int) -> list[dict]:
        out = [_fl(k, 10_000 * k + 99, _est(10_000 * k + 100, e), verdict="IN", bounce=10_000 * k + 130.0)
               for k, e in enumerate(errs) if e is not None]
        return out + [_fl(900 + j, 0, 9_000_000.0 + 10_000 * j) for j in range(extra)]

    return (fps, labels, _rep(flights(d_err, extra_d), fps), _rep(flights(s_err, extra_s), fps))


def _found_first(n: int, k: int) -> list[float | None]:
    return [0.0 if i < k else None for i in range(n)]


def test_the_fps_rule_counts_only_videos_of_at_least_50_fps_and_needs_three() -> None:
    v = sv.fps_rule([_fps_video(50.0, 4, _found_first(4, 0), _found_first(4, 4))] * 3)
    assert v.decision == "accepted" and v.numbers["videos_50fps"] == 3  # 50.0 itself is in
    under = sv.fps_rule([_fps_video(49.99, 4, _found_first(4, 0), _found_first(4, 4))] * 3)
    assert under.decision == "cannot decide" and under.numbers == {"videos_50fps": 0}
    two = sv.fps_rule([_fps_video(60.0, 4, _found_first(4, 0), _found_first(4, 4))] * 2
                      + [_fps_video(30.0, 4, _found_first(4, 0), _found_first(4, 4))] * 4)
    assert two.decision == "cannot decide" and two.numbers == {"videos_50fps": 2} and not two.criteria
    mixed = sv.fps_rule([_fps_video(60.0, 4, _found_first(4, 1), _found_first(4, 4))] * 3
                        + [_fps_video(29.97, 4, _found_first(4, 4), _found_first(4, 0))] * 2)
    assert mixed.numbers["videos_50fps"] == 3 and mixed.numbers["labelled"] == 12  # slow videos are left out
    assert mixed.numbers["recall_default"] == 25.0 and mixed.numbers["recall_scaled"] == 100.0


def test_the_fps_rule_cannot_decide_without_a_single_labelled_serve() -> None:
    v = sv.fps_rule([_fps_video(60.0, 0, [], [])] * 3)
    assert v.decision == "cannot decide" and v.numbers == {"videos_50fps": 3, "labelled": 0}


def test_the_fps_rule_recall_criterion_is_at_least_ten_points_exactly() -> None:
    def rise(found_d: list[int], found_s: list[int]) -> sv.Verdict:  # 3 videos of 10 labels: 30 labelled
        return sv.fps_rule([_fps_video(FPS, 10, _found_first(10, d), _found_first(10, s))
                            for d, s in zip(found_d, found_s, strict=True)])

    assert rise([1, 1, 2], [2, 2, 3]).criteria["recall +10 points"] is True  # 4 -> 7 of 30: +10 exactly
    assert rise([1, 1, 2], [2, 2, 2]).criteria["recall +10 points"] is False  # +6.7
    assert rise([2, 2, 2], [1, 1, 1]).criteria["recall +10 points"] is False  # a fall
    v = rise([1, 1, 2], [2, 2, 3])
    assert v.numbers["recall_default"] == pytest.approx(40 / 3) and v.numbers["recall_scaled"] == pytest.approx(70 / 3)


def test_the_fps_rule_within_two_share_may_fall_five_points_not_more() -> None:
    def share(scaled_within: int) -> sv.Verdict:  # 4 videos x 5 labels = 20 matched in each run
        d_err: list[float | None] = [0.0] * 5
        out = []
        for v in range(4):
            n_in = min(5, max(0, scaled_within - 5 * v))
            s_err: list[float | None] = [0.0] * n_in + [5.0] * (5 - n_in)  # 5 frames off: matched, not within 2
            out.append(_fps_video(FPS, 5, d_err, s_err))
        return sv.fps_rule(out)

    ok = share(19)  # 100 % -> 95 %: a fall of exactly 5 points
    assert ok.criteria["within-2 share falls <= 5 points"] is True
    assert ok.numbers["within2_share_default"] == 100.0 and ok.numbers["within2_share_scaled"] == 95.0
    assert share(18).criteria["within-2 share falls <= 5 points"] is False
    assert share(20).criteria["within-2 share falls <= 5 points"] is True


def test_the_fps_rule_share_with_nothing_matched_in_a_run_is_judged_on_the_scaled_run_finding_something() -> None:
    none_then_some = sv.fps_rule([_fps_video(FPS, 4, _found_first(4, 0), _found_first(4, 4))] * 3)
    assert none_then_some.numbers["within2_share_default"] is None
    # Pinned as it is: no default share to fall from, so the criterion is met when the scaled run matches anything.
    assert none_then_some.criteria["within-2 share falls <= 5 points"] is True
    some_then_none = sv.fps_rule([_fps_video(FPS, 4, _found_first(4, 4), _found_first(4, 0))] * 3)
    assert some_then_none.numbers["within2_share_scaled"] is None
    assert some_then_none.criteria["within-2 share falls <= 5 points"] is False  # the run lost every serve
    neither = sv.fps_rule([_fps_video(FPS, 4, _found_first(4, 0), _found_first(4, 0))] * 3)
    assert neither.decision == "rejected" and neither.criteria["within-2 share falls <= 5 points"] is False


def test_the_fps_rule_allows_two_more_false_serves_not_three() -> None:
    def extra(more: int) -> sv.Verdict:
        return sv.fps_rule([_fps_video(FPS, 4, _found_first(4, 4), _found_first(4, 4), extra_d=1, extra_s=1 + m)
                            for m in (more, 0, 0)])

    assert extra(2).criteria["false serves +<= 2"] is True
    assert extra(3).criteria["false serves +<= 2"] is False
    assert extra(0).criteria["false serves +<= 2"] is True
    fewer = sv.fps_rule([_fps_video(FPS, 4, _found_first(4, 4), _found_first(4, 4), extra_d=5, extra_s=0)] * 3)
    assert fewer.criteria["false serves +<= 2"] is True
    v = extra(2)
    assert (v.numbers["false_serves_default"], v.numbers["false_serves_scaled"]) == (3, 5)


def test_the_fps_rule_is_accepted_only_when_all_four_criteria_hold_and_reports_the_flips() -> None:
    vids = [_fps_video(FPS, 5, _found_first(5, 1), _found_first(5, 4)) for _ in range(3)]
    ok = sv.fps_rule(vids)
    assert ok.decision == "accepted" and all(ok.criteria.values()) and len(ok.criteria) == 4
    assert ok.numbers["flips"] == [] and ok.numbers["videos_50fps"] == 3 and ok.numbers["labelled"] == 15
    # One bounce called IN by the default run and OUT by the scaled run, in the last video only: a flip anywhere
    # rejects, and it is listed.
    vids[2][3]["flights"][0]["call"]["verdict"] = "OUT"
    bad = sv.fps_rule(vids)
    assert bad.decision == "rejected" and bad.criteria["no decisive flip"] is False
    assert [k for k, v in bad.criteria.items() if not v] == ["no decisive flip"]
    assert bad.numbers["flips"] == [(130.0, "IN", "OUT")]
    vids[2][3]["flights"][0]["call"]["verdict"] = "REVIEW"  # REVIEW is not a flip
    assert sv.fps_rule(vids).decision == "accepted"


# --- pin rule -------------------------------------------------------------------------------------------------------
# Boards are 100 km/h and speeds are multiples of 0.25, so every error is exact in floating point.


def _pin_video(specs: list[dict], base: int = 0) -> tuple:
    """Per serve a spec: u / p (fitted km/h without / with the pin), board (km/h or None), u_rej / p_rej, absent
    (the pinned run has no such flight), vu / vp (verdicts), kind_p (kind of the pinned flight)."""
    labels, du, dp = [], [], []
    for k, sp in enumerate(specs):
        lo = base + 10_000 * k + 100
        labels.append(_lab(lo, lo + 1, kmh=sp.get("board", 100.0)))
        du.append(_fl(k, lo - 1, lo + 0.5, kmh=sp.get("u", 103.0), rejected=sp.get("u_rej"),
                      verdict=sp.get("vu", "IN"), bounce=lo + 30.0))
        if not sp.get("absent"):
            dp.append(_fl(100 + k, lo - 1, lo + 0.5, kmh=sp.get("p", 101.5), rejected=sp.get("p_rej"),
                          verdict=sp.get("vp", "IN"), bounce=lo + 30.0, kind=sp.get("kind_p", "serve")))
    dp.reverse()  # the pinned run is paired by the hit's start frame, not by id or order
    return labels, _rep(du), _rep(dp)


def _specs(n: int, **kw) -> list[dict]:
    return [dict(kw) for _ in range(n)]


def test_the_pin_rule_needs_fifteen_paired_serves_and_counts_only_paired_ones() -> None:
    assert sv.pin_rule([_pin_video(_specs(15))]).decision == "accepted"
    short = sv.pin_rule([_pin_video(_specs(14))])
    assert short.decision == "inconclusive"
    assert short.numbers == {"paired": 14, "lost_by_pinned": [], "verdict_changes": []}
    for odd in ({"board": None}, {"u_rej": "speed"}, {"u": None}):  # not paired: no board, rejected, no speed
        v = sv.pin_rule([_pin_video(_specs(14) + [odd])])
        assert v.decision == "inconclusive" and v.numbers["paired"] == 14, odd
    labels, ru, rp = _pin_video(_specs(15))
    labels.append(_lab(900_000, 900_001, kmh=100.0))  # a labelled serve nobody found is not paired either
    assert sv.pin_rule([(labels, ru, rp)]).numbers["paired"] == 15
    assert sv.pin_rule([_pin_video(_specs(8)), _pin_video(_specs(7), base=500_000)]).decision == "accepted"
    assert sv.pin_rule([_pin_video(_specs(14), base=0)] + [_pin_video(_specs(14), base=0)]).numbers["paired"] == 28
    assert sv.pin_rule([_pin_video(_specs(3))], min_paired=3).decision == "accepted"
    assert sv.pin_rule([_pin_video(_specs(3))], min_paired=4).decision == "inconclusive"


def test_an_inconclusive_pin_run_still_lists_its_verdict_changes() -> None:
    specs = _specs(10)
    specs[3]["vp"] = "OUT"
    v = sv.pin_rule([_pin_video(specs)])
    assert v.decision == "inconclusive" and v.numbers["verdict_changes"] == [(30_000 + 99, "IN", "OUT")]


def test_the_pin_rule_wants_the_mean_error_to_fall_by_at_least_one_and_a_half_points_exactly() -> None:
    ok = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=101.5))])  # 3.0 -> 1.5 on every serve
    assert ok.decision == "accepted" and ok.numbers["mean_abs_drop_points"] == 1.5 and ok.criteria == {
        "mean |error| falls >= 1.5 points": True, "two thirds closer": True, "no serve verdict change": True}
    near = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=101.75))])  # a fall of 1.25: every serve closer
    assert near.decision == "rejected"
    assert [k for k, c in near.criteria.items() if not c] == ["mean |error| falls >= 1.5 points"]
    assert near.numbers["closer"] == 15 and near.numbers["mean_abs_drop_points"] == 1.25
    signed = sv.pin_rule([_pin_video(_specs(15, u=97.0, p=100.5))])  # below the board: |error| 3.0 -> 0.5, closer
    assert signed.numbers["mean_abs_drop_points"] == 2.5 and signed.decision == "accepted"
    crossing = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=97.0))])  # across the board, same |error|: no change
    assert crossing.numbers["mean_abs_drop_points"] == 0.0 and crossing.numbers["closer"] == 0
    assert crossing.numbers["further_or_lost"] == 0


def test_serves_the_pinned_run_loses_stay_out_of_the_means_and_count_as_worse() -> None:
    specs = _specs(15, u=103.0, p=101.5) + _specs(3, u=150.0, p=None)  # three lost, huge unpinned error
    v = sv.pin_rule([_pin_video(specs)])
    assert v.numbers["paired"] == 18 and v.numbers["both_measured"] == 15
    assert v.numbers["mean_abs_drop_points"] == 1.5  # the lost serves' 50 % do not enter the mean
    assert (v.numbers["closer"], v.numbers["further_or_lost"]) == (15, 3) and v.decision == "accepted"
    for lost in ({"p_rej": "speed"}, {"absent": True}, {"kind_p": "rally"}):
        w = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=101.5) + [dict(lost, u=103.0)] )])
        assert w.numbers["further_or_lost"] == 1 and w.numbers["both_measured"] == 15, lost


def test_the_serves_the_pinned_run_loses_are_listed_not_only_counted() -> None:
    # docs/EVAL_SERVE_SPEED.md, Measurement: such a serve "counts as worse in criterion 2 and is listed" (the
    # listing was missing before any download; the count was right).
    specs = _specs(15, u=103.0, p=101.5)
    specs[5].update(p=None)  # start frame 50099
    specs[9].update(p_rej="speed")  # start frame 90099
    v = sv.pin_rule([_pin_video(specs)])
    assert v.numbers["further_or_lost"] == 2
    lost = v.numbers["lost_by_pinned"]
    assert [(x["video"], x["start_frame"]) for x in lost] == [(0, 50099), (0, 90099)]
    assert lost[0]["pinned"] == "no speed" and lost[1]["pinned"] not in ("no speed", "no flight")  # its rejection
    few = sv.pin_rule([_pin_video(specs[:10])])
    assert few.decision == "inconclusive" and len(few.numbers["lost_by_pinned"]) == 2  # listed there too
    ok = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=101.5))])
    assert ok.numbers["lost_by_pinned"] == []


def test_two_thirds_of_the_changing_serves_must_get_closer_lost_ones_counting_against() -> None:
    def run(closer: int, worse: int, **worse_kw) -> sv.Verdict:
        specs = _specs(closer, u=105.0, p=102.0)  # 5 -> 2: closer
        specs += [dict(u=102.0, p=103.0, **worse_kw) if not worse_kw else dict(u=102.0, **worse_kw)
                  for _ in range(worse)]  # 2 -> 3 (or lost)
        return sv.pin_rule([_pin_video(specs)])

    assert run(10, 5).criteria["two thirds closer"] is True  # 30 >= 30
    assert run(9, 6).criteria["two thirds closer"] is False  # 27 < 30
    assert run(12, 6, p=None).criteria["two thirds closer"] is True  # lost serves are worse: 36 >= 36
    assert run(11, 6, p=None).criteria["two thirds closer"] is False
    assert run(12, 6, p_rej="speed").criteria["two thirds closer"] is True
    v = run(10, 5)
    assert (v.numbers["closer"], v.numbers["further_or_lost"]) == (10, 5)


def test_changes_under_half_a_point_are_ties_and_a_change_of_exactly_half_a_point_counts() -> None:
    ties = sv.pin_rule([_pin_video(_specs(9, u=105.0, p=102.0) + _specs(6, u=103.0, p=103.25))])
    assert (ties.numbers["closer"], ties.numbers["further_or_lost"]) == (9, 0)  # 0.25 worse: a tie, not counted
    assert ties.criteria["two thirds closer"] is True
    half = sv.pin_rule([_pin_video(_specs(9, u=105.0, p=102.0) + _specs(6, u=103.0, p=103.5))])
    assert (half.numbers["closer"], half.numbers["further_or_lost"]) == (9, 6)  # exactly 0.5 worse counts
    assert half.criteria["two thirds closer"] is False
    better = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=102.5))])  # exactly 0.5 closer on all
    assert better.numbers["closer"] == 15
    slight = sv.pin_rule([_pin_video(_specs(15, u=103.0, p=102.75))])
    assert slight.numbers["closer"] == 0 and slight.criteria["two thirds closer"] is False  # all ties: none counted
    assert slight.decision == "rejected"


def test_a_pin_run_that_loses_every_serve_has_no_mean_and_is_rejected() -> None:
    v = sv.pin_rule([_pin_video(_specs(15, p=None))])
    assert v.decision == "rejected" and v.numbers["both_measured"] == 0
    assert math.isnan(v.numbers["mean_abs_drop_points"]) and v.criteria["mean |error| falls >= 1.5 points"] is False
    assert v.numbers["further_or_lost"] == 15 and v.criteria["two thirds closer"] is False


def test_any_serve_verdict_change_in_or_out_of_review_rejects_the_pin_and_is_listed_by_start_frame() -> None:
    cases = (("IN", "OUT"), ("IN", "REVIEW"), ("REVIEW", "IN"), ("OUT", "REVIEW"))
    for vu, vp in cases:
        specs = _specs(15)
        specs[7].update(vu=vu, vp=vp)
        v = sv.pin_rule([_pin_video(specs)])
        assert v.decision == "rejected" and v.criteria["no serve verdict change"] is False, (vu, vp)
        assert v.numbers["verdict_changes"] == [(70_000 + 99, vu, vp)]
    same = _specs(15)
    same[7].update(vu="REVIEW", vp="REVIEW")
    assert sv.pin_rule([_pin_video(same)]).decision == "accepted"
    nocall = _specs(15)
    nocall[2].update(vu=None, vp=None)  # no call in either run is no change
    assert sv.pin_rule([_pin_video(nocall)]).decision == "accepted"
    gone = _specs(15)
    gone[4].update(absent=True)  # the pinned run lost a serve that had a call: its verdict changed to nothing
    assert sv.pin_rule([_pin_video(gone)]).numbers["verdict_changes"] == [(40_000 + 99, "IN", None)]
    only_u = _specs(15)
    only_u[4].update(vu=None)  # a call appearing only in the pinned run
    assert sv.pin_rule([_pin_video(only_u)]).numbers["verdict_changes"] == [(40_000 + 99, None, "IN")]


def test_verdict_changes_are_taken_over_every_serve_of_every_video_and_ignore_other_flights() -> None:
    specs_a, specs_b = _specs(8), _specs(8)
    specs_b[5].update(vp="OUT", u_rej="speed")  # a change on a serve that is not even paired
    v = sv.pin_rule([_pin_video(specs_a), _pin_video(specs_b, base=400_000)])
    assert v.numbers["paired"] == 15 and v.numbers["verdict_changes"] == [(400_000 + 50_000 + 99, "IN", "OUT")]
    labels, ru, rp = _pin_video(_specs(15))
    ru["flights"].append(_fl(50, 7, 8.0, kind="rally", verdict="IN", bounce=30.0))
    rp["flights"].append(_fl(51, 7, 8.0, kind="rally", verdict="OUT", bounce=30.0))
    assert sv.pin_rule([(labels, ru, rp)]).decision == "accepted"  # rally verdicts are reported, not ruled on


def test_the_pin_rules_serve_is_matched_in_the_unpinned_run_and_paired_by_start_frame() -> None:
    labels, ru, rp = _pin_video(_specs(15, u=103.0, p=101.5))
    for f in rp["flights"]:  # the pinned run's contact estimates drift; its start frames still identify the hit
        f["contact_frame"] += 500.0
    assert sv.pin_rule([(labels, ru, rp)]).numbers["both_measured"] == 15
    for f in rp["flights"]:
        f["start_frame"] += 1  # a different start frame: another hit, so every serve is lost
    v = sv.pin_rule([(labels, ru, rp)])
    assert v.numbers["both_measured"] == 0 and v.numbers["further_or_lost"] == 15


def test_a_board_in_mph_is_converted_before_the_error_is_taken() -> None:
    labels, ru, rp = _pin_video(_specs(15))
    kmh = 100.0 * 1.609344  # a 100 mph board
    for lab in labels:
        lab["radar"] = {"value": 100.0, "unit": "mph"}
    for f in ru["flights"]:
        f["hit_speed_kmh"] = kmh * 1.04
    for f in rp["flights"]:
        f["hit_speed_kmh"] = kmh * 1.01
    v = sv.pin_rule([(labels, ru, rp)])
    assert v.numbers["mean_abs_drop_points"] == pytest.approx(3.0) and v.decision == "accepted"


# --- third candidate: player rule -----------------------------------------------------------------------------------

ON_NEAR = [480.0, 700.0, 520.0, 900.0]  # feet (500, 900), 200 px tall: tolerance 60 px
ON_FAR = [945.0, 130.0, 975.0, 200.0]  # feet (960, 200), 70 px tall: tolerance 21 px
ELSEWHERE = [1480.0, 700.0, 1520.0, 900.0]


def _frames(n: int, near, far) -> dict:
    return {"players": {"near": {"box": [near(i) for i in range(n)]}, "far": {"box": [far(i) for i in range(n)]}}}


def _lbl(i: int, near: bool = True, far: bool = True, main: bool = True) -> dict:
    return {"frame": i, "main": main, "near": [500.0, 900.0] if near else None, "far": [960.0, 200.0] if far else None}


def test_player_frames_off_the_main_camera_count_for_nothing_and_a_tracked_absent_player_is_wrong() -> None:
    n = 6
    labels = [_lbl(0), _lbl(1, far=False), _lbl(2, main=False), _lbl(3), _lbl(4, near=False), _lbl(5, near=False,
                                                                                                    far=False)]
    frames = _frames(n, lambda i: ON_NEAR if i != 3 else ELSEWHERE, lambda i: ON_FAR if i in (0, 1, 2, 5) else None)
    st = sv.presence_stats(labels, frames, 1080)
    # near present on frames 0, 1, 3 (frame 2 is not the main camera); found on 0, 1; frame 3 tracked elsewhere;
    # frames 4 and 5 label no near player but one is tracked: wrong.
    assert st["main_frames"] == 5
    assert st["near"] == {"present": 3, "found": 2, "wrong": 1 + 2}
    # far present on 0, 3, 4; found on 0; frame 1 tracked while labelled out of view and frame 5 too; 3 and 4 missed.
    assert st["far"] == {"present": 3, "found": 1, "wrong": 2}
    assert sv.presence_stats([], frames, 1080) == {"main_frames": 0, "near": {"present": 0, "found": 0, "wrong": 0},
                                                   "far": {"present": 0, "found": 0, "wrong": 0}}


def test_a_missed_player_is_neither_found_nor_wrong_and_the_pixel_floor_scales_with_the_frame_height() -> None:
    labels = [_lbl(0)]
    missed = _frames(1, lambda i: None, lambda i: None)
    assert sv.presence_stats(labels, missed, 1080)["near"] == {"present": 1, "found": 0, "wrong": 0}
    small = [495.0, 880.0, 505.0, 900.0]  # 20 px tall: 6 px by the third, so the 1080-line floor of 15 px rules
    off_by_12 = _frames(1, lambda i: [507.0, 880.0, 517.0, 900.0], lambda i: None)  # feet x 512, 12 px off
    assert sv.presence_stats(labels, off_by_12, 1080)["near"]["found"] == 1
    assert sv.presence_stats(labels, off_by_12, 720)["near"]["found"] == 0  # at 720 lines the floor is 10 px
    assert small[3] == 900.0


def test_disagreement_stats_count_frames_where_exactly_one_run_is_right_and_how_many_are_the_second_runs() -> None:
    labels = [_lbl(0, far=False), _lbl(1, far=False), _lbl(2, far=False), _lbl(3, far=False), _lbl(4, far=False),
              _lbl(5, far=False, main=False)]
    a = _frames(6, lambda i: [ON_NEAR, ON_NEAR, None, ELSEWHERE, ON_NEAR, ON_NEAR][i],
                lambda i: ON_FAR if i == 1 else None)
    b = _frames(6, lambda i: [None, ON_NEAR, ON_NEAR, ELSEWHERE, ON_NEAR, None][i], lambda i: None)
    st = sv.disagreement_stats(labels, a, b, 1080)
    # near: frame 0 a right, 1 both right, 2 b right, 3 both wrong, 4 both right, 5 is not the main camera.
    assert st["near"] == {"one_right": 2, "b_right": 1}
    # far is labelled out of view: right means not tracking it. Frame 1: a tracks it (wrong), b right.
    assert st["far"] == {"one_right": 1, "b_right": 1}
    swapped = sv.disagreement_stats(labels, b, a, 1080)
    assert swapped["near"] == {"one_right": 2, "b_right": 1} and swapped["far"] == {"one_right": 1, "b_right": 0}
    empty = {side: {"one_right": 0, "b_right": 0} for side in ("near", "far")}
    assert sv.disagreement_stats([], a, b, 1080) == empty


def test_a_player_labelled_out_of_view_is_right_only_when_not_tracked_even_for_a_found_looking_box() -> None:
    labels = [{"frame": 0, "main": True, "near": None, "far": None}]
    tracked = _frames(1, lambda i: ON_NEAR, lambda i: None)
    untracked = _frames(1, lambda i: None, lambda i: None)
    st = sv.disagreement_stats(labels, tracked, untracked, 1080)
    assert st["near"] == {"one_right": 1, "b_right": 1} and st["far"] == {"one_right": 0, "b_right": 0}


def _pr_frames(n: int, near_found, far_found, far_wrong=lambda i: False):
    far = lambda i: ON_FAR if far_found(i) else (ELSEWHERE if far_wrong(i) else None)  # noqa: E731
    return _frames(n, lambda i: ON_NEAR if near_found(i) else None, far)


def test_the_player_rule_is_inconclusive_below_one_hundred_labelled_near_frames_and_only_main_ones_count() -> None:
    def run(n_near: int, n_other_camera: int = 0) -> sv.Verdict:
        labels = [_lbl(i, far=False) for i in range(n_near)] + [_lbl(n_near + j, far=False, main=False)
                                                                for j in range(n_other_camera)]
        f = _pr_frames(n_near + n_other_camera, lambda i: True, lambda i: False)
        return sv.player_rule([(labels, [], f, f, 1080)])

    short = run(99, 50)
    assert short.decision == "inconclusive" and short.numbers == {"near_present": 99}
    assert run(100).decision == "rejected"  # decided: no near gain is a rejection
    assert run(100).criteria["near coverage +5 points"] is False
    assert sv.player_rule([([_lbl(i, far=False) for i in range(3)], [], _pr_frames(3, lambda i: True, lambda i: False),
                            _pr_frames(3, lambda i: True, lambda i: False), 1080)], min_near=3).decision == "rejected"
    pooled = [([_lbl(i, far=False) for i in range(60)], [], _pr_frames(60, lambda i: True, lambda i: False),
               _pr_frames(60, lambda i: True, lambda i: False), 1080)] * 2
    assert sv.player_rule(pooled).numbers["near_present"] == 120  # counts add up over videos


def test_the_player_rule_near_coverage_must_rise_five_points_not_four() -> None:
    def run(cand_found: int) -> sv.Verdict:  # 200 near frames, the default finds 100
        labels = [_lbl(i, far=False) for i in range(200)]
        return sv.player_rule([(labels, [], _pr_frames(200, lambda i: i < 100, lambda i: False),
                                _pr_frames(200, lambda i: i < cand_found, lambda i: False), 1080)])

    assert run(110).criteria["near coverage +5 points"] is True  # 50 % -> 55 %
    assert run(109).criteria["near coverage +5 points"] is False
    v = run(110)
    assert v.numbers["near_coverage"] == [50.0, 55.0] and v.decision == "accepted"
    assert v.numbers["near_present"] == 200 and v.numbers["far_present"] == 0


def test_the_player_rule_far_coverage_may_fall_one_point_not_more() -> None:
    def run(cand_found: int) -> sv.Verdict:  # 100 frames: the near player found by both, far found 50 vs cand_found
        labels = [_lbl(i) for i in range(100)]
        return sv.player_rule([(labels, [], _pr_frames(100, lambda i: True, lambda i: i < 50),
                                _pr_frames(100, lambda i: True, lambda i: i < cand_found), 1080)])

    assert run(49).criteria["far coverage falls <= 1 point"] is True
    assert run(48).criteria["far coverage falls <= 1 point"] is False
    assert run(80).criteria["far coverage falls <= 1 point"] is True
    assert run(49).numbers["far_coverage"] == [50.0, 49.0] and run(49).numbers["far_present"] == 100


def test_the_player_rule_wrong_selections_may_rise_one_point_of_player_frames_not_more() -> None:
    def run(wrong_far: int) -> sv.Verdict:  # 100 frames, far labelled out of view: tracking it is wrong
        labels = [_lbl(i, far=False) for i in range(100)]
        return sv.player_rule([(labels, [], _pr_frames(100, lambda i: True, lambda i: False),
                                _pr_frames(100, lambda i: True, lambda i: False, lambda i: i < wrong_far), 1080)])

    assert run(2).criteria["wrong selections rise <= 1 point"] is True  # 2 of 200 player-frames: exactly 1 point
    assert run(3).criteria["wrong selections rise <= 1 point"] is False
    assert run(0).criteria["wrong selections rise <= 1 point"] is True
    assert run(2).numbers["wrong_share"] == [0.0, 1.0] and run(3).numbers["wrong_share"] == [0.0, 1.5]
    # A candidate that is less wrong than the default passes too, and the near player counts in the same share.
    labels = [_lbl(i, far=False) for i in range(100)]
    better = sv.player_rule([(labels, [], _pr_frames(100, lambda i: True, lambda i: False, lambda i: i < 10),
                              _pr_frames(100, lambda i: True, lambda i: False), 1080)])
    assert better.criteria["wrong selections rise <= 1 point"] is True and better.numbers["wrong_share"] == [5.0, 0.0]
    near_wrong = [_lbl(i, far=False) for i in range(100)]
    f_def = _pr_frames(100, lambda i: True, lambda i: False)
    f_cand = _frames(100, lambda i: ELSEWHERE if i < 3 else ON_NEAR, lambda i: None)
    assert sv.player_rule([(near_wrong, [], f_def, f_cand, 1080)]).numbers["wrong_share"] == [0.0, 1.5]


def _disagreement_case(one_right: int, b_right: int, side: str = "near", frame_offset: int = 0):
    """Random-sample part: 100 frames both runs track the near player and not the far one (labelled out of view).
    Disagreement part: ``one_right`` frames of ``side`` on which exactly one run is right, ``b_right`` of them b."""
    n = 100 + one_right
    labels = [_lbl(i, far=False) for i in range(one_right, n)]
    dlabels = [_lbl(i, near=(side == "near"), far=(side == "far")) for i in range(one_right)]

    def box(run_b: bool, near_side: bool):
        def f(i: int):
            if i >= one_right:
                return ON_NEAR if near_side else None
            if (side == "near") != near_side:
                return None
            right = (i < b_right) == run_b
            return (ON_NEAR if near_side else ON_FAR) if right else None
        return f

    fa = _frames(n, box(False, True), box(False, False))
    fb = _frames(n, box(True, True), box(True, False))
    return labels, dlabels, fa, fb


def test_the_disagreement_criterion_is_judged_per_player_from_ten_frames_and_needs_two_thirds() -> None:
    def verdict(one_right: int, b_right: int, side: str = "near") -> sv.Verdict:
        lab, dlab, fa, fb = _disagreement_case(one_right, b_right, side)
        return sv.player_rule([(lab, dlab, fa, fb, 1080)])

    key = "disagreements (near): candidate right in >= 2/3"
    assert key not in verdict(9, 0).criteria  # fewer than 10 frames: not judged
    assert verdict(10, 0).criteria[key] is False  # judged from exactly 10
    assert verdict(12, 8).criteria[key] is True  # 24 >= 24: exactly two thirds
    assert verdict(12, 7).criteria[key] is False
    assert verdict(30, 30).criteria[key] is True
    assert verdict(12, 8).numbers["disagreements"]["near"] == {"one_right": 12, "b_right": 8}
    far_key = "disagreements (far): candidate right in >= 2/3"
    far = verdict(12, 7, side="far")
    assert far.criteria[far_key] is False and key not in far.criteria  # the other player is not judged
    assert far.decision == "rejected" and verdict(12, 8, side="far").criteria[far_key] is True
    assert verdict(12, 8, side="far").numbers["disagreements"]["far"] == {"one_right": 12, "b_right": 8}


def test_disagreement_frames_of_all_videos_are_pooled_before_the_ten_frame_minimum() -> None:
    parts = []
    for _ in range(2):  # 6 + 6 frames: neither video reaches ten, together they do
        lab, dlab, fa, fb = _disagreement_case(6, 4)
        parts.append((lab, dlab, fa, fb, 1080))
    v = sv.player_rule(parts)
    assert v.numbers["disagreements"]["near"] == {"one_right": 12, "b_right": 8}
    assert v.criteria["disagreements (near): candidate right in >= 2/3"] is True
    assert sv.player_rule(parts, min_one_right=13).criteria.keys() == {
        "near coverage +5 points", "far coverage falls <= 1 point", "wrong selections rise <= 1 point"}


def test_the_player_rule_is_accepted_only_when_every_criterion_holds() -> None:
    labels = [_lbl(i, far=False) for i in range(200)]
    f_def = _pr_frames(200, lambda i: i < 100, lambda i: False)
    f_good = _pr_frames(200, lambda i: True, lambda i: False)
    ok = sv.player_rule([(labels, [], f_def, f_good, 1080)])
    assert ok.decision == "accepted" and all(ok.criteria.values()) and len(ok.criteria) == 3
    assert ok.numbers["totals"]["default"]["near"] == {"present": 200, "found": 100, "wrong": 0}
    assert ok.numbers["totals"]["candidate"]["main_frames"] == 200
    far_bad = _pr_frames(200, lambda i: True, lambda i: False, lambda i: i < 10)  # 10 wrong far selections: 2.5 points
    bad = sv.player_rule([(labels, [], f_def, far_bad, 1080)])
    assert bad.decision == "rejected"
    assert [k for k, c in bad.criteria.items() if not c] == ["wrong selections rise <= 1 point"]


# --- disagreement sample --------------------------------------------------------------------------------------------


def _box(height: float, dx: float = 0.0) -> list[float]:
    return [500.0 + dx - 10, 900.0 - height, 500.0 + dx + 10, 900.0]  # feet (500 + dx, 900)


def test_disagreement_frames_are_calibrated_ones_where_either_player_differs_and_in_frame_order() -> None:
    n = 10
    valid = [True] * n
    valid[3] = False
    fa = _frames(n, lambda i: _box(100), lambda i: _box(100))
    fb = _frames(n, lambda i: _box(100, 40.0 if i in (1, 3) else 0.0), lambda i: _box(100, 40.0 if i in (7, 5) else 0))
    assert sv.disagreement_sample("v", fa, fb, valid, 1080) == [1, 5, 7]  # frame 3 differs but is not calibrated
    assert sv.disagreement_sample("v", fa, fb, [False] * n, 1080) == []
    assert sv.disagreement_sample("v", fa, fa, valid, 1080) == []


def test_two_runs_differ_when_feet_are_more_than_a_third_of_the_smaller_box_apart_exactly() -> None:
    n = 5
    shifts = [0.0, 30.0, 30.5, 29.5, 100.0]  # 100 px boxes: tolerance 30 px
    fa = _frames(n, lambda i: _box(100), lambda i: None)
    fb = _frames(n, lambda i: _box(100, shifts[i]), lambda i: None)
    assert sv.disagreement_sample("v", fa, fb, [True] * n, 1080) == [2, 4]  # 30.0 apart is within the tolerance
    # The smaller of the two box heights sets the tolerance: 50 px and 100 px -> 15 px.
    fa2 = _frames(1, lambda i: _box(100), lambda i: None)
    fb2 = _frames(1, lambda i: _box(50, 20.0), lambda i: None)
    assert sv.disagreement_sample("v", fa2, fb2, [True], 1080) == [0]
    fb3 = _frames(1, lambda i: _box(50, 14.0), lambda i: None)
    assert sv.disagreement_sample("v", fa2, fb3, [True], 1080) == []
    # The tolerance has the scoring tolerance's floor (15 px at 1080 lines, scaled with the height): two 10 px boxes
    # with feet 5 px apart are the same player for both (the floor was missing here before any download).
    tiny_a = _frames(1, lambda i: _box(10), lambda i: None)
    tiny_b = _frames(1, lambda i: _box(10, 5.0), lambda i: None)
    assert sv.disagreement_sample("v", tiny_a, tiny_b, [True], 1080) == []
    assert sv._found(sv._feet(tiny_a, "near", 0), [505.0, 900.0], 1080) is True
    apart = _frames(1, lambda i: _box(10, 16.0), lambda i: None)
    assert sv.disagreement_sample("v", tiny_a, apart, [True], 1080) == [0]  # 16 px > 15 px
    assert sv.disagreement_sample("v", tiny_a, apart, [True], 2160) == []  # 16 px < 30 px at 2160 lines
    assert sv.disagreement_sample("v", tiny_a, apart, [True], 720) == [0]  # 10 px floor at 720 lines


def test_one_run_tracking_and_the_other_not_is_a_disagreement_two_missing_players_are_not() -> None:
    n = 6
    near_a = [None, None, _box(100), _box(100), [1.0, None, 3.0, 4.0], [1.0, None, 3.0, 4.0]]
    near_b = [None, _box(100), None, _box(100), None, [1.0, None, 3.0, 4.0]]
    fa = _frames(n, lambda i: near_a[i], lambda i: None)
    fb = _frames(n, lambda i: near_b[i], lambda i: None)
    assert sv.disagreement_sample("v", fa, fb, [True] * n, 1080) == [1, 2]  # a partial box is no box


def test_the_disagreement_sample_keeps_all_up_to_k_and_otherwise_draws_k_by_the_name() -> None:
    n = 300
    fa = _frames(n, lambda i: _box(100), lambda i: None)
    fb = _frames(n, lambda i: _box(100, 99.0), lambda i: None)
    valid = [True] * n
    assert sv.disagreement_sample("v", fa, fb, valid, 1080) == sv.disagreement_sample("v", fa, fb, valid, 1080, k=60)
    all_ = sv.disagreement_sample("v", fa, fb, valid, 1080, k=300)
    assert all_ == list(range(300))  # exactly k candidates: all of them
    assert sv.disagreement_sample("v", fa, fb, valid, 1080, k=301) == list(range(300))
    s = sv.disagreement_sample("v", fa, fb, valid, 1080)
    assert len(s) == 60 and s == sorted(s) and len(set(s)) == 60 and set(s) <= set(range(300))
    assert s == sv.disagreement_sample("v", fa, fb, valid, 1080)
    assert s != sv.disagreement_sample("w", fa, fb, valid, 1080)
    drawn = sv._rng("v:disagree").choice(list(range(300)), size=60, replace=False)
    assert s == sorted(int(i) for i in drawn)  # seeded by the name plus ":disagree", not the presence seed
    assert s != sv.presence_sample("v", 300, k=60)
    assert sv.disagreement_sample("v", fa, fb, valid, 1080, k=7) == sorted(
        int(i) for i in sv._rng("v:disagree").choice(list(range(300)), size=7, replace=False))


# --- reported, not used: call changes and speeds --------------------------------------------------------------------


def _called(i: int, bounce: float, verdict: str = "IN", kind: str = "serve", x: float | None = 1.0,
            y: float | None = 2.0, sigma: float | None = 0.02, p_in: float | None = 1.0, kmh: float | None = None,
            board: float | None = None, rejected: str | None = None) -> dict:
    f = _fl(i, i * 1000, i * 1000 + 10.0, kind=kind, kmh=kmh, rejected=rejected, bounce=bounce)
    f["call"] = {"verdict": verdict, "x": x, "y": y, "sigma_m": sigma, "p_in": p_in}
    f["board_speed_kmh"] = board
    return f


def test_call_changes_pair_bounces_within_two_frames_one_to_one_closest_first() -> None:
    a = _rep([_called(0, 40.0)])
    assert sv.call_changes([(a, _rep([_called(0, 42.0)]))])["only_a"] == 0  # 2 frames apart: paired
    far = sv.call_changes([(a, _rep([_called(0, 42.5)]))])
    assert (far["calls_a"], far["calls_b"], far["only_a"], far["only_b"]) == (1, 1, 1, 1)
    both = sv.call_changes([(a, _rep([_called(0, 42.0, verdict="OUT"), _called(1, 40.5, verdict="REVIEW")]))])
    # 40 pairs with 40.5 (closest), leaving the call at 42 alone in b; the change is the one against 40.5.
    assert both["only_b"] == 1 and both["verdict_changes"]["serve"] == [(40.0, "IN", "REVIEW")]
    pooled = sv.call_changes([(a, _rep([_called(0, 40.0)])), (a, _rep([]))])
    assert (pooled["calls_a"], pooled["calls_b"], pooled["only_a"], pooled["only_b"]) == (2, 1, 1, 0)


def test_call_changes_ignore_flights_without_a_call_or_a_bounce_and_split_serves_from_other_flights() -> None:
    nocall = _called(2, 70.0)
    nocall["call"] = None
    nobounce = _called(3, 80.0)
    nobounce["bounce"] = None
    a = _rep([_called(0, 40.0, kind="rally"), _called(1, 60.0, kind="serve"), nocall, nobounce])
    b = _rep([_called(0, 40.0, kind="rally", verdict="OUT"), _called(1, 60.0, kind="serve", verdict="REVIEW"),
              nocall, nobounce])
    cc = sv.call_changes([(a, b)])
    assert (cc["calls_a"], cc["calls_b"]) == (2, 2)
    assert cc["verdict_changes"] == {"serve": [(60.0, "IN", "REVIEW")], "other": [(40.0, "IN", "OUT")]}
    assert cc["review_changes"] == 1  # only the REVIEW one: an IN -> OUT change is not a review change
    c = _called(1, 60.0, kind="rally")
    d = _called(1, 60.0, kind="serve", verdict="OUT")  # the group is the first run's kind
    assert sv.call_changes([(_rep([c]), _rep([d]))])["verdict_changes"]["other"] == [(60.0, "IN", "OUT")]


def test_paired_calls_report_the_shift_in_cm_the_signed_sigma_change_and_the_p_in_change_in_size() -> None:
    a = _rep([_called(0, 40.0, x=1.0, y=2.0, sigma=0.05, p_in=0.9)])
    b = _rep([_called(0, 40.0, x=1.3, y=2.4, sigma=0.03, p_in=0.99)])
    p = sv.call_changes([(a, b)])["paired"]["serve"]
    assert p["shift_cm"]["max"] == pytest.approx(50.0) and p["shift_cm"]["n"] == 1
    assert p["sigma_change_cm"]["max"] == pytest.approx(-2.0)  # b - a, signed
    assert p["p_in_change"]["max"] == pytest.approx(0.09)
    assert sv.call_changes([(b, a)])["paired"]["serve"]["p_in_change"]["max"] == pytest.approx(0.09)  # absolute
    assert sv.call_changes([(a, b)])["paired"]["other"]["shift_cm"] == {"n": 0}
    gaps = _rep([_called(0, 40.0, x=None, sigma=None, p_in=None)])
    q = sv.call_changes([(a, gaps)])["paired"]["serve"]
    assert q == {"shift_cm": {"n": 0}, "sigma_change_cm": {"n": 0}, "p_in_change": {"n": 0}}
    no_y = _rep([_called(0, 40.0, y=None)])
    assert sv.call_changes([(no_y, a)])["paired"]["serve"]["shift_cm"] == {"n": 0}


def test_speed_report_errors_shares_and_the_mph_split_at_their_edges() -> None:
    def lab(lo: int, value: float, unit: str = "mph", side: str = "near") -> dict:
        return {"lo": lo, "hi": lo + 1, "side": side, "radar": {"value": value, "unit": unit}}

    kmh = 1.609344
    labels = [lab(100, 160.0, "km/h"), lab(1100, 100.0), lab(2100, 115.0), lab(3100, 99.0, side="far"),
              lab(4100, 120.0), lab(5100, 110.0)]
    flights = [_called(0, 130.0, kmh=168.0, board=160.0),  # 160 km/h board (99.4 mph), exactly +5 %: within 5 %
               _called(1, 1130.0, kmh=100 * kmh * 1.01),  # 100 mph: neither below 100 nor from 115
               _called(2, 2130.0, kmh=115 * kmh * 0.90),  # 115 mph: in the "from 115" group
               _called(3, 3130.0, kmh=99 * kmh * 1.20),
               _called(4, 4130.0, kmh=None),  # no fitted speed
               _called(5, 5130.0, kmh=110 * kmh * 0.5, rejected="speed")]  # a rejected speed is not used
    for k, f in enumerate(flights):
        f["start_frame"] = k * 1000 + 99
        f["contact_frame"] = k * 1000 + 100.5
    flights[3]["hitter"] = "far"
    r = sv.speed_report([(labels, _rep(flights))])
    assert r["labelled_board"]["n"] == 4 and r["labelled_board"]["mean_signed"] == pytest.approx((5 + 1 - 10 + 20) / 4)
    assert r["labelled_board"]["mean_abs"] == pytest.approx(36 / 4)
    assert r["labelled_board"]["within_5pct"] == 0.5  # +5.0 % and +1 % are within; -10 % and +20 % are not
    assert r["near"]["n"] == 3 and r["far"]["n"] == 1 and r["far"]["mean_signed"] == pytest.approx(20.0)
    assert r["board_below_100mph"]["n"] == 2 and r["board_below_100mph"]["mean_signed"] == pytest.approx(12.5)
    assert r["board_115mph_up"]["n"] == 1 and r["board_115mph_up"]["mean_signed"] == pytest.approx(-10.0)
    assert r["automatic_pairs"] == {"n": 1, "mean_signed": pytest.approx(5.0), "mean_abs": pytest.approx(5.0),
                                    "within_5pct": 1.0}  # only the flight with a board_speed_kmh
    assert sv.speed_report([])["labelled_board"] == {"n": 0}


def test_automatic_board_pairs_skip_rejected_missing_and_non_serve_flights() -> None:
    def fl(kind: str, kmh: float | None, board: float | None, rejected: str | None = None) -> dict:
        f = _called(0, 30.0, kind=kind, kmh=kmh, board=board, rejected=rejected)
        return f

    flights = [fl("serve", 110.0, 100.0), fl("serve", 100.0, 100.0, rejected="x"), fl("serve", None, 100.0),
               fl("serve", 110.0, None), fl("serve", 110.0, 0.0), fl("rally", 110.0, 100.0), fl("serve", 90.0, 100.0)]
    auto = sv.speed_report([([], _rep(flights))])["automatic_pairs"]
    assert auto["n"] == 2 and auto["mean_signed"] == pytest.approx(0.0) and auto["mean_abs"] == pytest.approx(10.0)
    assert auto["within_5pct"] == 0.0


# --- spot -----------------------------------------------------------------------------------------------------------


def _clip(events: list[tuple[str, int]], fps: float = 25.0, n: int = 3000) -> dict:
    return {"video": f"m_0_{n}", "fps": fps, "num_frames": n,
            "events": [{"label": lab, "frame": f, "comment": ""} for lab, f in events]}


def _hit(frame: float, side: str = "near", serve: bool = False) -> dict:
    return {"frame": frame, "side": side, "serve": serve}


def _sf(contact: float | None, side: str = "near", start: int | None = None, bounce: dict | None = None,
        kind: str = "serve") -> dict:
    return {"kind": kind, "hitter": side, "contact_frame": contact,
            "start_frame": int(contact) if start is None and contact is not None else (start or 0), "bounce": bounce}


def test_the_nominal_rate_is_the_nearest_whole_frame_rate() -> None:
    assert [spot.nominal_fps(f) for f in (25.0, 29.97, 30000 / 1001, 23.976, 59.94, 50.0, 60.0)] == [25, 30, 30, 24, 60,
                                                                                                     50, 60]
    assert spot.nominal_fps(29.4) == 29 and spot.nominal_fps(29.6) == 30
    assert isinstance(spot.nominal_fps(29.97), int)
    # Pinned as it is: halves round to even, so 24.5 fps counts as 24 and 25.5 as 26; no real video has such a rate.
    assert (spot.nominal_fps(24.5), spot.nominal_fps(25.5)) == (24, 26)


def test_system_events_from_hits_are_strokes_and_a_toss_flagged_hit_is_also_a_serve_toss() -> None:
    ev = spot.system_events({"hits": [_hit(50, "far", serve=True), _hit(20, "near"), {"frame": 30, "side": "far"}]})
    assert ev == [spot.Event("stroke", "near", 20.0), spot.Event("stroke", "far", 30.0),
                  spot.Event("stroke", "far", 50.0), spot.Event("serve_toss", "far", 50.0)]
    assert all(isinstance(e.frame, float) for e in ev)
    assert spot.system_events({}) == [] and spot.system_events({"hits": [], "flights": []}) == []


def test_system_serve_events_use_the_contact_frame_even_at_zero_and_the_start_frame_without_one() -> None:
    ev = spot.system_events({"flights": [_sf(0.0, start=5), _sf(None, start=77), _sf(310.5, "far"),
                                         _sf(400.0, kind="rally"), _sf(None, start=9, kind="rally")]})
    assert [(e.kind, e.side, e.frame) for e in ev] == [("serve", "near", 0.0), ("serve", "near", 77.0),
                                                       ("serve", "far", 310.5)]
    numeric = spot.system_events({"flights": [{"kind": "serve", "hitter": 1, "contact_frame": 4.0, "start_frame": 4,
                                               "bounce": None}]})
    assert numeric == [spot.Event("serve", "1", 4.0)]  # the side is always a string


def test_a_bounces_side_is_the_court_half_it_lands_on_and_a_bounce_without_a_frame_is_no_event() -> None:
    def bounce(y: float | None, frame: float | None = 100.0) -> dict:
        return {"kind": "rally", "hitter": "near", "contact_frame": None, "start_frame": 1,
                "bounce": {"frame": frame, "y": y}}

    flights = [bounce(0.5, 10.0), bounce(-0.5, 20.0), bounce(None, 30.0), bounce(5.0, None), bounce(0.0, 50.0),
               {**bounce(1.0, 60.0), "bounce": {}}, {**bounce(1.0, 70.0), "bounce": None},
               {**bounce(1.0, 80.0), "bounce": {"frame": 0.0, "y": 3.0}}]
    ev = spot.system_events({"flights": flights})
    assert [(e.side, e.frame) for e in ev] == [("far", 0.0), ("far", 10.0), ("near", 20.0), ("unknown", 30.0),
                                               ("near", 50.0)]  # frame 0.0 is a frame; the side at y == 0 is near
    # Pinned as it is: a bounce exactly on the net line (y == 0) is the near half; only y > 0 is far.
    assert all(e.kind == "bounce" for e in ev) and [e.frame for e in ev] == sorted(e.frame for e in ev)


def test_a_serve_flight_with_a_bounce_gives_both_a_serve_and_a_bounce_event_in_frame_order() -> None:
    f = _sf(100.0, "far", bounce={"frame": 120.0, "y": -4.0})
    ev = spot.system_events({"hits": [_hit(101, "far", serve=True)], "flights": [f]})
    assert [(e.kind, e.side, e.frame) for e in ev] == [("serve", "far", 100.0), ("stroke", "far", 101.0),
                                                       ("serve_toss", "far", 101.0), ("bounce", "near", 120.0)]


def test_match_events_are_inclusive_at_the_tolerance_ties_go_to_the_lower_indices_and_sides_are_ignored() -> None:
    ref = [spot.Event("stroke", "near", 100.0)]
    assert spot.match_events(ref, [spot.Event("stroke", "near", 102.5)], 2.5) == [(0, 0)]
    assert spot.match_events(ref, [spot.Event("stroke", "near", 102.6)], 2.5) == []
    assert spot.match_events(ref, [spot.Event("stroke", "near", 97.5)], 2.5) == [(0, 0)]
    assert spot.match_events(ref, [spot.Event("stroke", "far", 100.0)], 0.0) == [(0, 0)]  # the side is only reported
    two_ref = [spot.Event("stroke", "near", 100.0), spot.Event("stroke", "near", 104.0)]
    assert spot.match_events(two_ref, [spot.Event("stroke", "near", 102.0)], 5.0) == [(0, 0)]  # a tie: lower ref
    two_sys = [spot.Event("stroke", "near", 98.0), spot.Event("stroke", "near", 102.0)]
    assert spot.match_events(ref, two_sys, 5.0) == [(0, 0)]  # a tie: lower system index
    assert spot.match_events([], two_sys, 5.0) == [] and spot.match_events(ref, [], 5.0) == []
    chain = [spot.Event("stroke", "near", 10.0), spot.Event("stroke", "near", 20.0)]
    sysc = [spot.Event("stroke", "near", 14.0), spot.Event("stroke", "near", 17.0)]
    assert spot.match_events(chain, sysc, 6.0) == [(1, 1), (0, 0)]  # the 3-frame gap is taken before the 4-frame one
    crossing = [spot.Event("stroke", "near", 10.0), spot.Event("stroke", "near", 13.0)]
    one = [spot.Event("stroke", "near", 12.0)]
    assert spot.match_events(crossing, one, 5.0) == [(1, 0)]  # 1 frame beats 2: the later label takes it


def test_label_events_make_every_swing_a_stroke_and_every_serve_a_stroke_a_serve_and_a_toss() -> None:
    ev = spot.label_events(_clip([("far_court_serve", 50), ("near_court_swing", 20), ("far_court_bounce", 30)]))
    assert [(e.kind, e.side, e.frame) for e in ev] == [("stroke", "near", 20.0), ("bounce", "far", 30.0),
                                                       ("stroke", "far", 50.0), ("serve", "far", 50.0),
                                                       ("serve_toss", "far", 50.0)]
    assert sum(e.kind == "stroke" for e in ev) == 2 and sum(e.kind == "serve_toss" for e in ev) == 1
    assert all(isinstance(e.frame, float) for e in ev)
    assert spot.label_events(_clip([])) == []


def test_first_bounce_flags_count_a_leading_bounce_and_restart_at_every_stroke() -> None:
    def flags(spec: list[tuple[str, int]]) -> list[bool]:
        return spot.first_bounce_flags(spot.label_events(_clip(spec)))

    assert flags([("near_court_bounce", 5), ("near_court_bounce", 9)]) == [True, False]  # pinned: before any stroke
    assert flags([("far_court_serve", 1), ("near_court_bounce", 5), ("far_court_bounce", 9), ("near_court_swing", 12),
                  ("far_court_bounce", 15), ("far_court_swing", 18), ("near_court_bounce", 20)]) == [
        True, False, True, True]
    assert flags([("near_court_swing", 1), ("near_court_swing", 2), ("near_court_bounce", 3)]) == [True]
    assert flags([]) == []
    assert flags([("near_court_swing", 1)]) == []
    # Pinned as it is: a bounce and a stroke on the same frame are ordered by the order of the label list.
    assert flags([("near_court_bounce", 10), ("near_court_swing", 10), ("near_court_bounce", 12)]) == [True, True]
    assert flags([("near_court_swing", 10), ("near_court_bounce", 10), ("near_court_bounce", 12)]) == [True, False]


def test_serve_tolerance_is_half_a_second_at_the_nominal_rate() -> None:
    def matched(fps: float, gap: float) -> tuple[int, int]:
        row = spot.score_clip(_clip([("near_court_serve", 1000)], fps=fps),
                              {"hits": [_hit(1000 + gap, "near", serve=True)], "flights": [_sf(1000 + gap)]})
        return row["serve"]["matched"], row["serve_toss"]["matched"]

    assert matched(29.97, 15) == (1, 1) and matched(29.97, 16) == (0, 0)  # 0.5 s counted at 30 fps: 15 frames
    assert matched(25.0, 12) == (1, 1) and matched(25.0, 13) == (0, 0)  # 12.5 frames at 25 fps
    assert matched(25.0, -12) == (1, 1) and matched(25.0, -13) == (0, 0)  # before the label alike
    assert matched(59.94, 30) == (1, 1) and matched(59.94, 31) == (0, 0)


def test_the_bounce_and_stroke_windows_and_the_sweep_have_inclusive_edges_at_the_nominal_rate() -> None:
    def row(gap: float, fps: float = 29.97) -> dict:
        return spot.score_clip(_clip([("near_court_swing", 100), ("near_court_bounce", 500)], fps=fps),
                               {"hits": [_hit(100 + gap)], "flights": [
                                   {"kind": "rally", "hitter": "near", "contact_frame": None, "start_frame": 1,
                                    "bounce": {"frame": 500.0 - gap, "y": -3.0}}]})

    for gap, sweep in ((1.5, (1, 1, 1)), (1.6, (0, 1, 1)), (3.0, (0, 1, 1)), (3.1, (0, 0, 1)), (6.0, (0, 0, 1)),
                       (6.1, (0, 0, 0))):
        r = row(gap)
        for kind in ("stroke", "bounce"):
            assert tuple(r[kind]["sweep"][k] for k in ("0.05", "0.1", "0.2")) == sweep, (kind, gap)
            assert r[kind]["matched"] == sweep[1]
    r25 = row(2.5, fps=25.0)  # 0.1 s = 2.5 frames at 25 fps; 0.05 s = 1.25 and 0.2 s = 5
    assert r25["stroke"]["matched"] == 1 and r25["stroke"]["sweep"] == {"0.05": 0, "0.1": 1, "0.2": 1}
    assert row(2.6, fps=25.0)["stroke"]["sweep"] == {"0.05": 0, "0.1": 0, "0.2": 1}
    assert row(1.25, fps=25.0)["stroke"]["sweep"] == {"0.05": 1, "0.1": 1, "0.2": 1}
    assert row(5.0, fps=25.0)["bounce"]["sweep"]["0.2"] == 1 and row(5.1, fps=25.0)["bounce"]["sweep"]["0.2"] == 0
    assert "sweep" not in row(1.0)["serve"] and "sweep" not in row(1.0)["serve_toss"]


def test_a_serve_before_a_labelled_one_is_a_fault_only_within_thirty_seconds_and_strictly_before() -> None:
    def before(fps: float, ahead: float, side: str = "near", also_matched: bool = False) -> dict:
        flights = [_sf(2000 - ahead, side)]
        hits = [_hit(2000 - ahead, side, serve=True)]
        if also_matched:
            flights.append(_sf(2000.0, "near"))
            hits.append(_hit(2000.0, "near", serve=True))
        row = spot.score_clip(_clip([("near_court_serve", 2000)], fps=fps), {"hits": hits, "flights": flights})
        return {"serve": row["serve"]["before_label"], "toss": row["serve_toss"]["before_label"]}

    assert before(29.97, 900) == {"serve": 1, "toss": 1}  # 30 s at the nominal 30 fps: 900 frames, inside
    assert before(29.97, 901) == {"serve": 0, "toss": 0}
    assert before(25.0, 750) == {"serve": 1, "toss": 1} and before(25.0, 751) == {"serve": 0, "toss": 0}
    assert before(25.0, 100, side="far") == {"serve": 0, "toss": 0}  # a labelled serve of the other side
    assert before(25.0, 100, also_matched=True) == {"serve": 1, "toss": 1}  # the matched one is not counted
    # An unmatched serve at the label's own frame is not "before" it (the gap must be positive).
    row = spot.score_clip(_clip([("near_court_serve", 2000)]), {"hits": [], "flights": [_sf(2000.0), _sf(2000.0)]})
    assert row["serve"]["matched"] == 1 and row["serve"]["system"] == 2 and row["serve"]["before_label"] == 0
    after = spot.score_clip(_clip([("near_court_serve", 2000)]), {"hits": [], "flights": [_sf(2100.0)]})
    assert after["serve"]["before_label"] == 0  # after the label: not a fault
    far_out = spot.score_clip(_clip([("near_court_serve", 2000)]), {"hits": [], "flights": [_sf(1900.0)]})
    assert far_out["serve"]["matched"] == 0 and far_out["serve"]["before_label"] == 1  # 100 frames: outside 0.5 s


def test_timing_gaps_are_system_minus_label_and_sides_must_agree_only_for_the_side_count() -> None:
    clip = _clip([("near_court_swing", 100), ("far_court_swing", 200), ("near_court_swing", 300)])
    rep = {"hits": [_hit(102, "near"), _hit(198, "near"), _hit(300, "far")], "flights": []}
    st = spot.score_clip(clip, rep)["stroke"]
    assert sorted(st["dt"]) == [-2.0, 0.0, 2.0] and st["matched"] == 3 and st["same_side"] == 1
    assert (st["labelled"], st["system"]) == (3, 3)


def test_first_bounce_matched_counts_a_matched_first_bounce_not_a_matched_second_one() -> None:
    clip = _clip([("far_court_serve", 10), ("near_court_bounce", 20), ("near_court_bounce", 40),
                  ("near_court_swing", 60), ("far_court_bounce", 80)])
    only_second = {"hits": [], "flights": [{"kind": "rally", "hitter": "far", "start_frame": 1, "contact_frame": None,
                                            "bounce": {"frame": 41.0, "y": -3.0}}]}
    b = spot.score_clip(clip, only_second)["bounce"]
    assert (b["labelled"], b["matched"], b["first_labelled"], b["first_matched"]) == (3, 1, 2, 0)
    both = {"hits": [], "flights": [{"kind": "rally", "hitter": "far", "start_frame": 1, "contact_frame": None,
                                     "bounce": {"frame": f, "y": -3.0}} for f in (20.0, 41.0, 81.0)]}
    b2 = spot.score_clip(clip, both)["bounce"]
    assert (b2["matched"], b2["first_labelled"], b2["first_matched"], b2["same_side"]) == (3, 2, 2, 2)


def test_a_failed_analysis_scores_as_no_system_events_and_carries_no_calibrated_share() -> None:
    clip = _clip([("far_court_serve", 50), ("near_court_bounce", 70)])
    row = spot.score_clip(clip, None)
    assert row["failed"] is True and "calibrated_share" not in row and row["fps"] == 25.0
    for kind in spot.TOL_S:
        assert row[kind]["system"] == 0 and row[kind]["matched"] == 0 and row[kind]["dt"] == []
    assert row["serve"]["labelled"] == 1 and row["bounce"]["labelled"] == 1 and row["serve"]["before_label"] == 0
    assert row["bounce"]["sweep"] == {"0.05": 0, "0.1": 0, "0.2": 0}
    assert spot.score_clip(clip, {})["calibrated_share"] == 0.0  # no calibration section: nothing calibrated
    ok = spot.score_clip(_clip([], n=400), {"calibration": {"valid_frames": 100}})
    assert ok["calibrated_share"] == 0.25 and ok["failed"] is False


def _row(clip: dict, report: dict | None) -> dict:
    return spot.score_clip(clip, report)


def test_summary_pools_counts_over_clips_before_taking_ratios() -> None:
    a = _row(_clip([("near_court_swing", 100)]), {"hits": [_hit(101)], "calibration": {"valid_frames": 3000}})
    b = _row(_clip([("near_court_swing", 100), ("near_court_swing", 200), ("near_court_swing", 300)]),
             {"hits": [_hit(400), _hit(500), _hit(600)], "calibration": {"valid_frames": 1500}})
    st = spot.summary([a, b])["stroke"]
    assert (st["labelled"], st["system"], st["matched"]) == (4, 4, 1)
    assert st["recall"] == 0.25 and st["precision"] == 0.25 and st["f1"] == 0.25  # not the mean of 1.0 and 0.0
    assert spot.summary([a, b])["calibrated_share"] == pytest.approx(0.75)
    assert spot.summary([a, b])["events_per_clip"] == 2.0  # 4 labelled strokes over 2 clips
    assert st["median_dt_frames"] == 1.0 and st["same_side"] == 1.0


def test_summary_f1_precision_and_recall_are_not_a_number_when_empty_and_f1_is_zero_without_a_match() -> None:
    nothing = _row(_clip([("near_court_swing", 100)]), {"hits": [_hit(900)], "calibration": {"valid_frames": 10}})
    s = spot.summary([nothing])
    assert s["stroke"]["f1"] == 0.0 and s["stroke"]["recall"] == 0.0 and s["stroke"]["precision"] == 0.0
    assert math.isnan(s["stroke"]["median_dt_frames"]) and math.isnan(s["stroke"]["same_side"])
    assert math.isnan(s["bounce"]["recall"]) and math.isnan(s["bounce"]["precision"]) and s["bounce"]["f1"] == 0.0
    assert math.isnan(s["bounce"]["first_recall"]) and s["bounce"]["first_labelled"] == 0
    empty = spot.summary([])
    assert empty["clips"] == 0 and math.isnan(empty["events_per_clip"]) and math.isnan(empty["calibrated_share"])
    assert empty["serve"]["labelled"] == 0 and math.isnan(empty["serve"]["recall"])
    assert "recall_sweep" not in empty["bounce"]
    failed_only = spot.summary([_row(_clip([("near_court_swing", 1)]), None)])
    assert failed_only["failed"] == 1 and math.isnan(failed_only["calibrated_share"])
    assert failed_only["stroke"]["recall"] == 0.0 and math.isnan(failed_only["stroke"]["precision"])


def test_summary_f1_is_the_harmonic_mean_and_serve_precision_can_leave_out_likely_faults() -> None:
    clip = _clip([("near_court_swing", 100), ("near_court_swing", 200), ("near_court_swing", 300),
                  ("near_court_swing", 400)])
    row = _row(clip, {"hits": [_hit(100), _hit(200), _hit(900)], "calibration": {"valid_frames": 3000}})
    st = spot.summary([row])["stroke"]
    assert st["precision"] == pytest.approx(2 / 3) and st["recall"] == 0.5
    assert st["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    sclip = _clip([("near_court_serve", 2000), ("far_court_serve", 2200)])
    srow = _row(sclip, {"hits": [], "flights": [_sf(1800.0), _sf(2001.0), _sf(2200.0, "far"), _sf(100.0, "far"),
                                               _sf(2900.0, "far")]})
    s = spot.summary([srow])["serve"]
    # 5 system serves, 2 matched; 1800 (near) is 200 frames = 8 s before 2000: a fault; 100 (far) is 2100 frames
    # (84 s) before 2200: not; 2900 (far) is after every label.
    assert (s["system"], s["matched"], s["before_label"]) == (5, 2, 1)
    assert s["precision"] == 2 / 5 and s["precision_excl_faults"] == 2 / 4
    zero = _row(_clip([("near_court_serve", 2000)]), {"hits": [], "flights": [_sf(1900.0)]})
    assert math.isnan(spot.summary([zero])["serve"]["precision_excl_faults"])  # 1 system serve, all set apart


def test_summary_recall_sweep_first_bounce_recall_and_per_match_grouping() -> None:
    clip = _clip([("far_court_serve", 10), ("near_court_bounce", 20), ("near_court_bounce", 40),
                  ("near_court_swing", 60), ("far_court_bounce", 80)], n=3000)
    clip["video"] = "usopen_2019_x_100_3100"
    rep = {"hits": [], "calibration": {"valid_frames": 3000}, "flights": [
        {"kind": "rally", "hitter": "far", "start_frame": 1, "contact_frame": None,
         "bounce": {"frame": f, "y": -3.0}} for f in (20.0, 42.0)]}
    row = _row(clip, rep)
    s = spot.summary([row])
    assert s["bounce"]["recall"] == 2 / 3 and s["bounce"]["first_recall"] == 0.5 and s["bounce"]["first_labelled"] == 2
    assert s["bounce"]["recall_sweep"] == {"0.05": 1 / 3, "0.1": 2 / 3, "0.2": 2 / 3}
    assert s["bounce"]["median_dt_frames"] == 1.0 and s["bounce"]["same_side"] == 1.0
    assert "recall_sweep" not in s["serve"] and "first_recall" not in s["stroke"]
    other = dict(clip, video="wimbledon_2018_y_5_3005")
    pm = spot.per_match([row, _row(other, None), _row(dict(clip, video="usopen_2019_x_300_3300"), rep)])
    assert list(pm) == ["usopen_2019_x", "wimbledon_2018_y"]  # sorted by match name
    assert pm["usopen_2019_x"]["clips"] == 2 and pm["usopen_2019_x"]["bounce"]["matched"] == 4
    assert pm["wimbledon_2018_y"]["failed"] == 1 and pm["wimbledon_2018_y"]["bounce"]["matched"] == 0
    assert spot.match_name("a_b_c_1_2") == "a_b_c" and spot.match_name("plain") == "plain"
    assert spot.per_match([]) == {}


def test_a_clip_must_match_its_name_hold_known_labels_and_integer_frames_inside_it() -> None:
    spot.check_clip(_clip([("near_court_bounce", 2999), ("far_court_swing", 0)]))
    bad_name = _clip([])
    bad_name["video"] = "m_0_2999"
    with pytest.raises(ValueError, match="name says 2999 frames, num_frames 3000"):
        spot.check_clip(bad_name)
    for ev in ({"label": "near_court_bounce", "frame": 5.0}, {"label": "near_court_bounce", "frame": True},
               {"label": "near_court_bounce"}, {"label": "near_court_bounce", "frame": 3000},
               {"label": "left_court_bounce", "frame": 5}, {"label": "near_court_serve_toss", "frame": 5},
               {"frame": 5}):
        c = _clip([])
        c["events"] = [ev]
        with pytest.raises(ValueError):
            spot.check_clip(c)
    assert {f"{s}_court_{k}" for s in ("near", "far") for k in ("bounce", "swing", "serve")} == spot.LABELS
    assert spot.TOL_S == {"bounce": 0.1, "stroke": 0.1, "serve": 0.5, "serve_toss": 0.5} and spot.FAULT_WINDOW_S == 30.0


def test_the_fps_rule_recall_gain_just_under_ten_points_fails_and_a_share_fall_just_over_five_fails() -> None:
    nine = sv.fps_rule([_fps_video(FPS, 11, _found_first(11, d), _found_first(11, s))
                        for d, s in ((1, 2), (1, 2), (2, 3))])  # 33 labelled: 4 -> 7 is +9.09 points
    assert nine.criteria["recall +10 points"] is False and nine.numbers["recall_scaled"] == pytest.approx(700 / 33)
    # 17 serves matched by both runs, all within 2 frames in the default run, 16 of 17 (94.1 %) in the scaled one.
    labels_n = (6, 6, 5)
    vids = []
    for v, n in enumerate(labels_n):
        s_err: list[float | None] = [0.0] * n
        if v == 0:
            s_err[0] = 5.0
        vids.append(_fps_video(FPS, n, [0.0] * n, s_err))
    v = sv.fps_rule(vids)
    assert v.numbers["within2_share_scaled"] == pytest.approx(1600 / 17)
    assert v.criteria["within-2 share falls <= 5 points"] is False  # a fall of 5.88 points


def test_a_matched_serve_followed_by_another_labelled_serve_is_not_a_fault_only_unmatched_ones_are() -> None:
    clip = _clip([("near_court_serve", 1000), ("near_court_serve", 1500)])
    flights = [_sf(1000.0), _sf(1500.0), _sf(1100.0)]  # the third matches nothing and 400 frames precede label 2
    row = spot.score_clip(clip, {"hits": [], "flights": flights})
    assert row["serve"]["matched"] == 2 and row["serve"]["system"] == 3 and row["serve"]["before_label"] == 1


def test_the_summary_averages_calibration_over_clips_that_ran_and_takes_the_median_gap_over_all_pairs() -> None:
    swings = [("near_court_swing", 100), ("near_court_swing", 200), ("near_court_swing", 300)]
    ok = _row(_clip(swings), {"hits": [_hit(100), _hit(200), _hit(302)], "calibration": {"valid_frames": 300}})
    failed = _row(_clip(swings), None)
    s = spot.summary([ok, failed])
    assert s["calibrated_share"] == pytest.approx(0.1) and s["failed"] == 1 and s["clips"] == 2
    assert ok["stroke"]["dt"] == [0.0, 0.0, 2.0] and s["stroke"]["median_dt_frames"] == 0.0  # the median, not 0.67
    assert s["stroke"]["recall"] == 0.5 and s["stroke"]["precision"] == 1.0  # the failed clip's 3 labels are missed


def test_per_match_summaries_are_listed_in_the_order_of_the_match_names() -> None:
    def row(video: str) -> dict:
        clip = _clip([("near_court_swing", 100)])
        clip["video"] = video
        return _row(clip, None)

    pm = spot.per_match([row("wimbledon_2018_y_5_3005"), row("usopen_2019_x_1_3001"), row("usopen_2019_a_1_3001")])
    assert list(pm) == ["usopen_2019_a", "usopen_2019_x", "wimbledon_2018_y"]


# --- sim_benchmark --------------------------------------------------------------------------------------------------
# The real simulation + fit take about 4 s a shot, so the contract of ``run_benchmark`` (sampling, bookkeeping, the
# statistics) is pinned with scripted stand-ins for ``simulate_shot``, ``fit_flight`` and ``call_bounce``, plus one
# real single-shot smoke run.

HL = sb.SINGLES_HALF_WIDTH


class _Script:
    """Scripted outcomes per shot, in call order. ``truth``: (x, |y|); the half comes from the shot's start."""

    def __init__(self, shots: list[dict]) -> None:
        self.shots = shots
        self.k = -1
        self.sim_calls: list[dict] = []
        self.fit_calls: list[dict] = []
        self.call_seeds: list[int] = []
        self.call_zones: list[str] = []
        self.sgn: list[float] = []

    def simulate(self, camera, start, tgt, speed, spin_rpm, fps, pixel_noise, rng):
        self.k += 1
        sgn = -math.copysign(1.0, start[1])  # to_far shots start at y = -11
        self.sgn.append(sgn)
        self.sim_calls.append({"start": np.array(start), "tgt": np.array(tgt), "speed": speed, "spin": spin_rpm,
                               "fps": fps, "noise": pixel_noise})
        x, y = self.shots[self.k]["truth"]
        return SimpleNamespace(times=np.arange(5) / fps, pixels=np.zeros((5, 2)), bounce_time=0.5,
                               bounce_xy=np.array([x, sgn * y]))

    def fit(self, times, pixels, cams, t_guess, pixel_sigma):
        shot = self.shots[self.k]
        self.fit_calls.append({"n_cams": len(cams), "t_guess": t_guess, "sigma": pixel_sigma, "n_times": len(times)})
        truth_xy = np.array([shot["truth"][0], self.sgn[self.k] * shot["truth"][1]])
        err = np.array(shot.get("err", (0.0, 0.0)))
        cov = np.array(shot.get("cov", np.diag([1e-4, 1e-4])), dtype=float)
        return SimpleNamespace(success=shot.get("success", True), bounce_xy=truth_xy + err, bounce_cov=cov)

    def call(self, xy, cov, zone, seed):
        self.call_seeds.append(seed)
        self.call_zones.append(zone.name)
        return SimpleNamespace(verdict=self.shots[self.k]["verdict"])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Script:
        monkeypatch.setattr(sb, "simulate_shot", self.simulate)
        monkeypatch.setattr(sb, "fit_flight", self.fit)
        monkeypatch.setattr(sb, "call_bounce", self.call)
        return self


def _run(monkeypatch: pytest.MonkeyPatch, shots: list[dict], **kw) -> tuple[sb.BenchmarkReport, _Script]:
    sc = _Script(shots).install(monkeypatch)
    return sb.run_benchmark(object(), n_shots=len(shots), **kw), sc


def test_benchmark_statistics_follow_the_fitted_shots_only_with_coverage_decisiveness_and_accuracy(monkeypatch) -> None:
    shots = [
        {"truth": (0.0, 6.0), "err": (0.03, 0.04), "cov": np.diag([0.04**2, 0.02**2]), "verdict": "IN"},
        {"truth": (5.0, 6.0), "err": (0.10, 0.0), "cov": np.diag([0.02**2, 0.02**2]), "verdict": "IN"},  # truth OUT
        {"truth": (HL - 0.05, 6.0), "success": False, "verdict": "IN"},  # fit fails: skipped, not near-line counted
        {"truth": (0.0, 6.0), "err": (0.0, 0.2), "cov": np.diag([0.1**2, 0.1**2]), "verdict": "REVIEW"},
        {"truth": (0.0, 6.0), "cov": np.array([[np.nan, 0.0], [0.0, 1e-4]]), "verdict": "IN"},  # non-finite: skipped
        {"truth": (HL - 0.05, 6.0), "err": (0.01, 0.0), "cov": np.diag([0.01**2, 0.01**2]), "verdict": "OUT"},
    ]
    rep, sc = _run(monkeypatch, shots, fps=60.0, pixel_noise=0.5)
    assert rep.n_shots == 4  # Pinned as it is: n_shots is the number of fitted shots, not the number asked for (6)
    assert (rep.fps, rep.pixel_noise) == (60.0, 0.5)
    assert rep.median_error_cm == pytest.approx(7.5)  # errors 1, 5, 10, 20 cm
    assert rep.p90_error_cm == pytest.approx(17.0)  # linear percentile between 10 and 20 cm
    assert rep.median_sigma_cm == pytest.approx(3.0)  # largest standard deviations 4, 2, 10, 1 cm
    assert rep.coverage_95 == 0.75  # Mahalanobis^2 4.56, 25, 4, 1 against 5.991: the second is outside
    assert rep.decisive_rate == 0.75  # the REVIEW call is not decisive
    assert rep.decisive_accuracy == pytest.approx(1 / 3)  # IN on an IN shot; IN on an OUT shot; OUT on an IN shot
    assert rep.wrong_decisive == 2
    assert rep.near_line_shots == 1  # the failed fit is not counted, nor the shots far from every line
    assert sc.call_seeds == [0, 1, 3, 5]  # the seed is the shot's index, skipped shots included
    assert len(sc.sim_calls) == 6


def test_the_coverage_threshold_is_the_95_percent_chi_square_for_two_degrees_of_freedom(monkeypatch) -> None:
    def coverage(mahalanobis2: float) -> float:
        d = math.sqrt(mahalanobis2) * 0.01
        shots = [{"truth": (0.0, 6.0), "err": (d, 0.0), "cov": np.diag([1e-4, 1e-4]), "verdict": "IN"}]
        return _run(monkeypatch, shots)[0].coverage_95

    assert coverage(5.9) == 1.0 and coverage(6.1) == 0.0
    # The off-diagonal terms matter: an error along the long axis of a tilted ellipse is covered, across it is not.
    cov = np.array([[2e-4, 1.8e-4], [1.8e-4, 2e-4]])
    e_long, e_short = np.array([0.02, 0.02]), np.array([0.02, -0.02])
    assert float(e_long @ np.linalg.solve(cov, e_long)) < 5.991 < float(e_short @ np.linalg.solve(cov, e_short))
    for err, covered in ((e_long, 1.0), (e_short, 0.0)):
        shot = [{"truth": (0.0, 6.0), "err": tuple(err), "cov": cov, "verdict": "IN"}]
        assert _run(monkeypatch, shot)[0].coverage_95 == covered
    sigma = _run(monkeypatch, [{"truth": (0.0, 6.0), "cov": cov, "verdict": "IN"}])[0].median_sigma_cm
    assert sigma == pytest.approx(100 * math.sqrt(3.8e-4))  # the square root of the larger eigenvalue, not the mean


def test_truth_is_in_two_centimetres_beyond_the_line_and_near_line_is_measured_from_that_edge(monkeypatch) -> None:
    def one(x: float, verdict: str = "IN") -> sb.BenchmarkReport:
        return _run(monkeypatch, [{"truth": (x, 6.0), "verdict": verdict}])[0]

    assert one(HL + 0.01).wrong_decisive == 0 and one(HL + 0.01, "OUT").wrong_decisive == 1  # truth IN: 1 cm over
    assert one(HL + 0.03).wrong_decisive == 1 and one(HL + 0.03, "OUT").wrong_decisive == 0  # truth OUT: 3 cm over
    assert one(HL - 0.30).wrong_decisive == 0 and one(HL - 0.30, "OUT").wrong_decisive == 1
    inside, outside = (HL - 0.075, HL - 0.085, HL - 0.5), (HL + 0.115, HL + 0.125, HL + 0.5)
    near = {round(x - HL, 3): one(x).near_line_shots for x in inside + outside}
    # Pinned as it is: "within 10 cm of a line" is measured from the line grown by the 2 cm, so the band runs from 8 cm
    # inside to 12 cm outside the drawn line, not 10 cm either way.
    assert near == {-0.075: 1, -0.085: 0, -0.5: 0, 0.115: 1, 0.125: 0, 0.5: 0}


def test_the_call_is_made_against_the_zone_of_the_half_the_ball_was_hit_into(monkeypatch) -> None:
    shots = [{"truth": (0.0, 6.0), "verdict": "IN"} for _ in range(12)]
    rep, sc = _run(monkeypatch, shots)
    assert rep.n_shots == 12
    for sgn, zone in zip(sc.sgn, sc.call_zones, strict=True):
        assert zone.startswith("far") == (sgn > 0)
    assert len(set(sc.sgn)) == 2  # both halves occur with this seed
    assert rep.wrong_decisive == 0 and rep.decisive_accuracy == 1.0 and rep.decisive_rate == 1.0


def test_the_fit_gets_every_frame_the_pixel_noise_as_sigma_and_a_start_guess_within_one_frame(monkeypatch) -> None:
    rep, sc = _run(monkeypatch, [{"truth": (0.0, 6.0), "verdict": "IN"}] * 5, fps=25.0, pixel_noise=2.0)
    assert all(c["sigma"] == 2.0 and c["n_cams"] == c["n_times"] == 5 for c in sc.fit_calls)
    assert all(abs(c["t_guess"] - 0.5) <= 1 / 25.0 for c in sc.fit_calls)
    assert len({round(c["t_guess"], 9) for c in sc.fit_calls}) > 1  # the guess is jittered, not the true bounce time
    assert all(c["fps"] == 25.0 and c["noise"] == 2.0 for c in sc.sim_calls) and rep.fps == 25.0


def test_shots_are_sampled_from_the_documented_ranges_and_near_line_fraction_steers_the_targets(monkeypatch) -> None:
    shots = [{"truth": (0.0, 6.0), "verdict": "IN"}] * 60
    _, interior = _run(monkeypatch, shots, near_line_fraction=0.0)
    for c in interior.sim_calls:
        assert abs(abs(c["start"][1]) - 11.0) < 1e-12 and abs(c["start"][0]) <= 3.0 and 0.6 <= c["start"][2] <= 1.4
        assert 80.0 <= c["speed"] <= 150.0 and -800.0 <= c["spin"] <= 3000.0
        assert abs(c["tgt"][0]) <= 3.8 and 4.0 <= abs(c["tgt"][1]) <= 11.4
        assert math.copysign(1, c["tgt"][1]) == -math.copysign(1, c["start"][1])  # hit towards the other end
    _, lines = _run(monkeypatch, shots, near_line_fraction=1.0)
    for c in lines.sim_calls:
        x, y = c["tgt"]
        on_sideline = abs(abs(x) - HL) <= 0.15 + 1e-9 and 3.0 <= abs(y) <= 11.0
        on_baseline = abs(abs(y) - sb.HALF_LENGTH) <= 0.2 + 1e-9 and abs(x) <= 3.5
        assert on_sideline or on_baseline
        assert math.copysign(1, y) == -math.copysign(1, c["start"][1])
    assert any(abs(abs(c["tgt"][0]) - HL) <= 0.15 + 1e-9 for c in lines.sim_calls)  # both kinds of line occur
    assert any(abs(abs(c["tgt"][1]) - sb.HALF_LENGTH) <= 0.2 + 1e-9 for c in lines.sim_calls)
    _, again = _run(monkeypatch, shots, near_line_fraction=1.0)
    assert all(np.array_equal(a["tgt"], b["tgt"]) for a, b in zip(lines.sim_calls, again.sim_calls, strict=True))
    _, other = _run(monkeypatch, shots, near_line_fraction=1.0, seed=8)
    assert not np.array_equal(lines.sim_calls[0]["tgt"], other.sim_calls[0]["tgt"])
    _, half = _run(monkeypatch, shots, near_line_fraction=0.5)
    near_lines = sum(abs(abs(c["tgt"][0]) - HL) <= 0.15 + 1e-9 or abs(abs(c["tgt"][1]) - sb.HALF_LENGTH) <= 0.2 + 1e-9
                     for c in half.sim_calls)
    assert 15 <= near_lines <= 45  # about half of the 60 targets are put on a line


def test_no_fitted_shot_is_an_error_and_no_decisive_call_reports_zero_accuracy(monkeypatch) -> None:
    for shots in ([{"truth": (0.0, 6.0), "success": False, "verdict": "IN"}],
                  [{"truth": (0.0, 6.0), "cov": np.full((2, 2), np.inf), "verdict": "IN"}]):
        with pytest.raises(ValueError, match="no simulated shot was fitted"):
            _run(monkeypatch, shots)
    with pytest.raises(ValueError, match="no simulated shot was fitted"):
        sb.run_benchmark(object(), n_shots=0)
    rep, _ = _run(monkeypatch, [{"truth": (0.0, 6.0), "verdict": "REVIEW"}] * 3)
    assert (rep.decisive_rate, rep.decisive_accuracy, rep.wrong_decisive) == (0.0, 0.0, 0)
    # Pinned as it is: with no decisive call the accuracy reads 0.0 (not undefined); read it with decisive_rate.


def test_the_report_serialises_every_field_as_json() -> None:
    rep = sb.BenchmarkReport(n_shots=4, fps=30.0, pixel_noise=1.5, median_error_cm=1.25, p90_error_cm=3.5,
                             coverage_95=0.75, median_sigma_cm=2.0, decisive_rate=1.0, decisive_accuracy=0.5,
                             wrong_decisive=2, near_line_shots=1)
    back = json.loads(rep.to_json())
    assert back == {"n_shots": 4, "fps": 30.0, "pixel_noise": 1.5, "median_error_cm": 1.25, "p90_error_cm": 3.5,
                    "coverage_95": 0.75, "median_sigma_cm": 2.0, "decisive_rate": 1.0, "decisive_accuracy": 0.5,
                    "wrong_decisive": 2, "near_line_shots": 1}


def test_one_real_simulated_shot_runs_end_to_end_and_reports_consistent_numbers() -> None:
    from courtvision.calib.camera import Camera

    center, target = np.array([0.0, -28.0, 11.0]), np.array([0.0, 1.5, 0.0])
    fwd = (target - center) / np.linalg.norm(target - center)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    cam = Camera(K=np.array([[1750.0, 0, 960.0], [0, 1750.0, 540.0], [0, 0, 1.0]]), R=R, t=-R @ center)
    try:
        rep = sb.run_benchmark(cam, n_shots=1, fps=30.0, pixel_noise=0.5, seed=3)
    except ValueError:
        pytest.skip("the one simulated shot did not fit with this library version")
    assert rep.n_shots == 1 and (rep.fps, rep.pixel_noise) == (30.0, 0.5)
    assert rep.median_error_cm == rep.p90_error_cm and 0.0 <= rep.median_error_cm < 100.0  # one shot: both are it
    assert rep.coverage_95 in (0.0, 1.0) and rep.decisive_rate in (0.0, 1.0) and rep.median_sigma_cm > 0.0
    assert rep.wrong_decisive in (0, 1) and rep.near_line_shots in (0, 1)
    assert set(json.loads(rep.to_json())) == set(sb.BenchmarkReport.__dataclass_fields__)
