"""Candidate --score-consistency (eval/broadcast_check.py): which score readings the player names may be counted
from. A reading whose last set is finished while a point is under way has lost the current set's column; a
reading showing fewer sets than one before it has lost a column too."""

from __future__ import annotations

import pytest

from courtvision.broadcast.scoreboard import ScoreReading, ScoreRow
from courtvision.eval.broadcast_check import consistent_scores, current_set_missing
from courtvision.eval.six_videos import check_name_labels, names_rule, names_sample


def _r(t: float, g0: tuple[int, ...], g1: tuple[int, ...], p0: str | None = "15",
       p1: str | None = "30") -> ScoreReading:
    return ScoreReading(int(t * 30), t, (ScoreRow("A", g0, p0), ScoreRow("B", g1, p1)))


def test_a_finished_last_set_with_a_point_under_way_means_the_current_set_was_not_read() -> None:
    assert current_set_missing(_r(0, (2, 6, 1), (6, 3, 6)))  # 1-6 is over, yet 15-30 is being played
    assert current_set_missing(_r(0, (6,), (3,)))
    assert current_set_missing(_r(0, (7,), (5,))) and current_set_missing(_r(0, (6,), (7,)))
    assert current_set_missing(_r(0, (6,), (4,), "AD", "40"))
    assert current_set_missing(_r(0, (6,), (2,), None, "15"))  # one row's points are enough


def test_a_set_still_under_way_or_a_new_set_at_love_all_is_not_a_missing_column() -> None:
    for g0, g1 in (((2, 6, 1), (6, 3, 5)), ((6,), (5,)), ((6,), (6,)), ((5,), (3,)), ((7,), (7,)), ((7,), (4,))):
        assert not current_set_missing(_r(0, g0, g1)), (g0, g1)  # 6-5, 6-6, 5-3 under way; 7-7 and 7-4 impossible
    assert not current_set_missing(_r(0, (6,), (3,), "0", "0"))  # the new set's first point: 0-0 changes no ends
    assert not current_set_missing(_r(0, (6,), (3,), None, None))  # no point under way
    assert not current_set_missing(_r(0, (6, 3), (3,)))  # rows of unequal length are another misread
    assert not current_set_missing(_r(0, (), ()))


def test_consistent_scores_drop_missing_columns_and_readings_with_fewer_sets_than_before() -> None:
    a = _r(1.0, (2, 6, 1), (6, 3, 5))  # three sets, the third under way
    b = _r(2.0, (2, 6), (6, 3), None, None)  # a set column lost, no point under way: kept by the first test only
    c = _r(3.0, (2, 6, 1), (6, 3, 6))  # 1-6 over and 15-30 played: the fourth set's column is missing
    d = _r(4.0, (2, 6, 1, 1), (6, 3, 6, 0))  # four sets
    e = _r(5.0, (2, 6, 1), (6, 3, 6), "0", "0")  # after a four-set reading, a three-set one is a misread
    assert consistent_scores([a, b, c, d, e]) == [a, d]
    assert consistent_scores([d, a]) == [a, d]  # time order, whatever the list order
    assert consistent_scores([]) == []
    first = _r(0.0, (6,), (3,), None, None)
    assert consistent_scores([first, b]) == [first, b]  # 1 then 2 sets: rising is fine


# --- the six-video rule for the candidate (eval/six_videos.py names_rule) ----------------------------------------

def _flight(i: int, near_name: str | None, hitter: str = "near") -> dict:
    return {"id": i, "start_frame": 10 * i, "end_frame": 10 * i + 5, "hitter": hitter, "kind": "rally",
            "call": {"verdict": "IN"}, "bounce": {"frame": 10 * i + 3},
            "hitter_name": near_name if hitter == "near" else "OTHER",
            "receiver_name": near_name if hitter == "far" else "OTHER"}


def _video(truth: list[str], shown_d: list[str | None], shown_c: list[str | None]):
    labels = [{"flight": i, "near": t} for i, t in enumerate(truth)]
    d = {"flights": [_flight(i, s, "near" if i % 2 == 0 else "far") for i, s in enumerate(shown_d)]}
    c = {"flights": [_flight(i, s, "near" if i % 2 == 0 else "far") for i, s in enumerate(shown_c)]}
    return labels, d, c


def test_names_rule_counts_right_wrong_and_none_and_needs_ten_wrong_names_to_decide() -> None:
    truth = ["SINNER"] * 20
    v = names_rule([_video(truth, ["ALCARAZ"] * 9 + ["SINNER"] * 11, [None] * 9 + ["SINNER"] * 11)])
    assert v.decision == "inconclusive" and v.numbers["default"] == {"right": 11, "wrong": 9, "none": 0}
    v = names_rule([_video(truth, ["ALCARAZ"] * 10 + ["SINNER"] * 10, [None] * 10 + ["SINNER"] * 10)])
    assert v.decision == "accepted" and v.numbers["candidate"] == {"right": 10, "wrong": 0, "none": 10}


def test_names_rule_halving_and_85_percent_hold_exactly_at_their_boundary() -> None:
    truth = ["SINNER"] * 40
    d = ["ALCARAZ"] * 10 + ["SINNER"] * 20 + [None] * 10  # 10 wrong, 20 right
    half = ["ALCARAZ"] * 5 + [None] * 5 + ["SINNER"] * 17 + [None] * 13  # 5 wrong (= half), 17 right (= 85 %)
    assert names_rule([_video(truth, d, half)]).decision == "accepted"
    more = ["ALCARAZ"] * 6 + [None] * 4 + ["SINNER"] * 20 + [None] * 10
    assert names_rule([_video(truth, d, more)]).criteria["wrong names at most half"] is False
    fewer = ["ALCARAZ"] * 0 + [None] * 10 + ["SINNER"] * 16 + [None] * 14  # 16 < 17
    assert names_rule([_video(truth, d, fewer)]).criteria["right names at least 85 %"] is False


def test_names_rule_rejects_a_candidate_that_changes_anything_but_names() -> None:
    labels, d, c = _video(["SINNER"] * 12, ["ALCARAZ"] * 12, [None] * 12)
    c["flights"][3]["call"] = {"verdict": "OUT"}
    v = names_rule([(labels, d, c)])
    assert v.decision == "rejected" and v.criteria["nothing but names changes"] is False
    assert v.numbers["videos_with_other_changes"] == [0]


def test_names_rule_skips_unclear_labels_and_matches_a_surname_inside_the_shown_name() -> None:
    labels, d, c = _video(["AUGER"] * 12, ["SINNER"] * 12, ["AUGER-ALIASSIME"] * 12)
    labels[0]["near"] = None  # unclear: not counted
    v = names_rule([(labels, d, c)])
    assert v.numbers["default"]["wrong"] == 11 and v.numbers["candidate"]["right"] == 11


def test_name_labels_and_the_flight_sample() -> None:
    check_name_labels([{"flight": 1, "near": "SINNER"}, {"flight": 2, "near": None}], ("SINNER", "DRAPER"))
    for bad in ([{"flight": 1}], [{"flight": "1", "near": None}], [{"flight": 1, "near": "ALCARAZ"}],
                [{"flight": 1, "near": None}, {"flight": 1, "near": None}]):
        with pytest.raises(ValueError):
            check_name_labels(bad, ("SINNER", "DRAPER"))
    assert names_sample("v", [3, 1, 2]) == [1, 2, 3]  # fewer than 40: all of them
    s = names_sample("v", list(range(200)))
    assert len(s) == 40 and s == sorted(s) and s == names_sample("v", list(range(200)))
    assert s != names_sample("w", list(range(200)))
