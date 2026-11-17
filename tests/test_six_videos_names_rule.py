"""The six-video rule for the player names (eval/six_videos.py names_rule, check_name_labels, names_sample): the
scorer of the rejected --score-consistency candidate, kept with its tests as the record of how its result was
computed (results/six_videos_test.json)."""

from __future__ import annotations

import pytest

from courtvision.eval.six_videos import check_name_labels, names_rule, names_sample


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


def test_name_labels_need_ties_take_any_case_and_come_back_in_upper_case() -> None:
    ties = [{"frame": 10, "server": "Sinner"}, {"frame": 20, "server": "DRAPER"}, {"frame": 30, "server": "sinner"}]
    got = check_name_labels({"ties": ties, "flights": [{"flight": 1, "near": "sinner"}, {"flight": 2, "near": None}]},
                            ("SINNER", "DRAPER"))
    assert got == [{"flight": 1, "near": "SINNER"}, {"flight": 2, "near": None}]  # a lower-case label is fine
    bad_docs = [
        [{"flight": 1, "near": "SINNER"}],  # the old bare list
        {"ties": ties, "flights": [{"flight": 1}]},
        {"ties": ties, "flights": [{"flight": "1", "near": None}]},
        {"ties": ties, "flights": [{"flight": 1, "near": "ALCARAZ"}]},
        {"ties": ties, "flights": [{"flight": 1, "near": None}, {"flight": 1, "near": None}]},
        {"ties": ties[:2], "flights": [{"flight": 1, "near": "SINNER"}]},  # fewer than 3 serves tie the names
        {"ties": [*ties[:2], {"frame": 20, "server": "SINNER"}], "flights": [{"flight": 1, "near": "SINNER"}]},
        {"ties": [{"frame": 1, "server": "NADAL"}, *ties], "flights": [{"flight": 1, "near": "SINNER"}]},
    ]
    for doc in bad_docs:
        with pytest.raises(ValueError):
            check_name_labels(doc, ("SINNER", "DRAPER"))  # type: ignore[arg-type]
    # A video with every flight unclear needs no ties.
    assert check_name_labels({"ties": [], "flights": [{"flight": 1, "near": None}]}, ("SINNER", "DRAPER")) == [
        {"flight": 1, "near": None}]


def test_the_flight_sample_is_all_flights_up_to_forty_and_drawn_by_the_name_beyond() -> None:
    assert names_sample("v", [3, 1, 2]) == [1, 2, 3]
    s = names_sample("v", list(range(200)))
    assert len(s) == 40 and s == sorted(s) and s == names_sample("v", list(range(200)))
    assert s != names_sample("w", list(range(200)))


def test_names_rule_reports_each_video_the_names_source_and_other_name_fields() -> None:
    labels, d, c = _video(["SINNER"] * 12, ["ALCARAZ"] * 12, [None] * 12)
    for k, f in enumerate(d["flights"]):
        f["names_from"] = "serve" if k < 4 else "graphic"
    c["flights"][5]["stroke_wing"] = "backhand"  # the hand vote can follow the names
    c["flights"][6]["server_name"] = "SINNER"
    labels2, d2, c2 = _video(["SINNER"] * 3, ["SINNER"] * 3, ["SINNER"] * 3)
    v = names_rule([(labels, d, c), (labels2, d2, c2)])
    assert v.numbers["per_video"] == [{"default": {"right": 0, "wrong": 12, "none": 0},
                                       "candidate": {"right": 0, "wrong": 0, "none": 12}},
                                      {"default": {"right": 3, "wrong": 0, "none": 0},
                                       "candidate": {"right": 3, "wrong": 0, "none": 0}}]
    assert v.numbers["by_default_names_source"]["serve"]["default"] == {"right": 0, "wrong": 4, "none": 0}
    assert v.numbers["by_default_names_source"]["graphic"]["candidate"] == {"right": 0, "wrong": 0, "none": 8}
    assert v.numbers["by_default_names_source"]["None"]["default"]["right"] == 3  # no names_from in that report
    assert v.numbers["flights_with_other_name_fields_changed"] == {
        "server_name": 1, "hand": 0, "hand_source": 0, "stroke": 0, "stroke_wing": 1}
    assert v.decision == "accepted"  # wrong 12 -> 0, right 3 -> 3; the other name fields are reported, not judged
