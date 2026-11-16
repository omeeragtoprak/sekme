"""Candidate --score-consistency (eval/broadcast_check.py): which score readings the player names may be counted
from. A reading whose last set is finished while a point is under way has lost the current set's column; a
reading showing fewer sets than one before it has lost a column too."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from courtvision.broadcast.ocr import TextBox
from courtvision.broadcast.scoreboard import ScoreReading, ScoreRow, cell_digit, parse_scoreboard
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


# --- the cell recovery: Vision does not read the graphic's "0"s ------------------------------------------------

BAR = (90, 35, 20)  # the graphic's dark blue (BGR)


def _canvas() -> np.ndarray:
    return np.full((720, 1280, 3), BAR, np.uint8)


def _draw(img: np.ndarray, glyph: str, x: float, y: float, w: float = 0.012, h: float = 0.024) -> None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
    cx, cy, rx, ry = (x0 + x1) // 2, (y0 + y1) // 2, (x1 - x0) // 2, (y1 - y0) // 2
    light = (240, 240, 240)
    if glyph == "0":
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, light, 3)
    elif glyph == "1":
        cv2.rectangle(img, (cx - 2, y0), (cx + 2, y1), light, -1)
    elif glyph == "6":  # a lower loop and a left stem: open at the top right
        cv2.ellipse(img, (cx, cy + ry // 3), (rx, (2 * ry) // 3), 0, 0, 360, light, 3)
        cv2.line(img, (x0 + 1, cy + ry // 3), (x0 + 1, y0), light, 3)
    elif glyph == "two":  # two narrow glyphs side by side
        cv2.rectangle(img, (x0, y0), (x0 + 3, y1), light, -1)
        cv2.rectangle(img, (x1 - 3, y0), (x1, y1), light, -1)
    elif glyph in ("box", "AD", "40"):  # a light points box with dark text
        cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 8, y1 + 4), light, -1)
        cv2.putText(img, "15" if glyph == "box" else glyph, (x0, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
    else:  # a wide filled glyph (another digit's ink)
        cv2.rectangle(img, (x0, y0), (x1, y1), light, -1)


def test_a_cell_reads_as_a_ring_a_bar_empty_or_unknown_from_its_pixels() -> None:
    for glyph, want in (("0", "0"), ("1", "1"), ("box", "?"), ("AD", "?"), ("40", "?"), ("solid", "?"),
                        ("6", "?"), ("two", "?")):
        img = _canvas()
        _draw(img, glyph, 0.5, 0.5)
        assert cell_digit(img, 0.495, 0.517, 0.498, 0.028) == want, glyph
    assert cell_digit(_canvas(), 0.495, 0.517, 0.498, 0.028) == "empty"
    assert cell_digit(_canvas(), 1.2, 1.3, 0.5, 0.028) == "?"  # outside the frame: nothing to read


def _graphic(img: np.ndarray, top: list, bottom: list, pts: tuple[str, str], hidden: tuple[str, str] | None):
    """Boxes of a two-row graphic with set columns at x = 0.259 + k * 0.0195, the points after them; ``hidden``
    is drawn as pixels in the next column but given no box (as Vision leaves the "0"s out)."""
    boxes = [TextBox("1 SINNER ITA", 1, 0.06, 0.832, 0.092, 0.026),
             TextBox("2 ALCARAZ ESP", 1, 0.06, 0.872, 0.115, 0.026)]
    for row, (vals, y) in enumerate(((top, 0.834), (bottom, 0.874))):
        for k, v in enumerate(vals):
            boxes.append(TextBox(str(v), 1, 0.259 + k * 0.0195, y, 0.011, 0.024))
        n = len(vals)
        if hidden is not None:
            _draw(img, hidden[row], 0.259 + n * 0.0195, y)
        boxes.append(TextBox(pts[row], 1, 0.259 + (n + (hidden is not None)) * 0.0195 + 0.012, y, 0.016, 0.024))
    return boxes


def test_the_parser_recovers_a_left_out_current_set_from_the_cells_pixels() -> None:
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [2, 6, 1], [6, 3, 6], ("15", "30"), ("1", "0")), img, 10, 1.0)
    assert r is not None and r.rows[0].games == (2, 6, 1) and r.next_set == (1, 0)  # the reading itself is as read
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [2, 6, 1], [6, 3, 6], ("AD", "40"), ("0", "0")), img, 10, 1.0)
    assert r is not None and r.next_set == (0, 0)


def test_no_recovery_without_a_finished_set_a_point_box_room_for_a_cell_or_a_readable_cell() -> None:
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [2, 6, 1], [6, 3, 5], ("15", "30"), ("0", "0")), img, 10, 1.0)
    assert r is not None and r.next_set is None  # 1-5: the set is still being played
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [6], [3], ("15", "30"), ("0", "0")), img, 10, 1.0)
    assert r is not None and r.next_set == (0, 0)  # one set column is enough: the cell is found by the gap
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [2, 6, 1], [6, 3, 6], ("15", "30"), None), img, 10, 1.0)
    assert r is not None and r.next_set is None  # the points box right after the last set: no room for a cell
    img = _canvas()
    r = parse_scoreboard(_graphic(img, [2, 6], [6, 3], ("15", "30"), ("0", "solid")), img, 10, 1.0)
    assert r is not None and r.next_set is None  # one cell unreadable: nothing is guessed
    r = parse_scoreboard(_graphic(_canvas(), [2, 6], [6, 3], ("15", "30"), ("0", "0")), None, 10, 1.0)
    assert r is not None and r.next_set is None  # no image


def test_the_filter_keeps_a_reading_whose_left_out_set_was_recovered_and_counts_it() -> None:
    from dataclasses import replace

    lost = _r(3.0, (2, 6, 1), (6, 3, 6))  # 1-6 over, 15-30 played
    back = replace(lost, next_set=(1, 0))
    assert consistent_scores([back])[0].rows[0].games == (2, 6, 1, 1)
    assert consistent_scores([back])[0].rows[1].games == (6, 3, 6, 0)
    assert consistent_scores([lost]) == []  # not recovered: dropped as before
    later = _r(4.0, (2, 6, 1), (6, 3, 6), "0", "0")  # three sets after a recovered four-set reading: a misread
    assert [r.t for r in consistent_scores([back, later])] == [3.0]


def test_a_blank_points_box_in_one_row_uses_the_other_rows_box_to_find_the_cell() -> None:
    img = _canvas()
    boxes = _graphic(img, [2, 6, 1], [6, 3, 6], ("AD", "40"), ("0", "0"))
    top_points = [b for b in boxes if b.text == "AD"]
    assert len(top_points) == 1
    boxes = [b for b in boxes if b.text != "AD"]  # the other player's box is blank at AD: Vision gives no text box
    r = parse_scoreboard(boxes, img, 10, 1.0)
    assert r is not None and r.rows[0].points is None and r.rows[1].points == "40"
    assert r.next_set == (0, 0)


def test_a_cache_from_before_next_set_is_read_again_only_when_the_candidate_is_on(tmp_path, monkeypatch) -> None:
    import pickle

    import courtvision.pipeline as pl
    from courtvision.broadcast.scoreboard import BroadcastScan

    old = _r(1.0, (6,), (3,))
    object.__delattr__(old, "next_set")  # as unpickled from a cache written before the field existed
    assert "next_set" not in vars(old) and old.next_set is None  # the class default still answers
    with open(tmp_path / "broadcast.pkl", "wb") as f:
        pickle.dump({"video_sha256": "sha", "scan": BroadcastScan(15, 1, scores=[old])}, f)
    fresh = BroadcastScan(15, 1, scores=[_r(1.0, (6,), (3,))])
    calls: list = []
    monkeypatch.setattr(pl, "scan_video", lambda video, fps: calls.append(video) or fresh)
    off = pl._broadcast_scan("v.mp4", 30.0, tmp_path, "sha", pl.PipelineConfig(reuse_perception=True))
    assert off.scores[0] is not None and calls == []  # the default reuses the old cache
    on = pl._broadcast_scan("v.mp4", 30.0, tmp_path, "sha", pl.PipelineConfig(reuse_perception=True,
                                                                               score_consistency=True))
    assert on is fresh and calls == ["v.mp4"]  # the candidate reads the graphic again
