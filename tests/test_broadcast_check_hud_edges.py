"""The broadcast check (eval/broadcast_check.py: end changes, name mapping, the time windows of reading_near and
side_names, the unit conversion, the serve and board pairing, the start-end vote) and the overlay (render/hud.py:
panel, draw_texts, fonts, lean_in, the ground ellipse, the renderer) at their edges. Values checked by running the
code (gaps found by a test review); nothing here depends on which font the machine has (CI has none of the macOS
fonts and falls back to Pillow's)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import courtvision.render.hud as hud
from courtvision.broadcast.scoreboard import BroadcastScan, ScoreReading, ScoreRow, SpeedReading
from courtvision.calib.camera import apply_h
from courtvision.eval.broadcast_check import (
    MPH_TO_KMH,
    Serve,
    check,
    end_changes,
    reading_near,
    serve_anchored_names,
    side_names,
    to_kmh,
)
from courtvision.geometry.court import CourtModel
from courtvision.render.hud import OverlayState, Renderer, TextItem, draw_texts, ground_ellipse_px, lean_in, panel

NAMES = ("SINNER", "ALCARAZ")


def _score(t: float, games0: tuple[int, ...] = (0,), games1: tuple[int, ...] = (0,), server: int | None = 0,
           names: tuple[str, str] = NAMES) -> ScoreReading:
    """A reading at ``t`` s; ``server`` is the row index with the serving marker (None: no marker at all)."""
    rows = (ScoreRow(names[0], games0, "0", None if server is None else server == 0),
            ScoreRow(names[1], games1, "0", None if server is None else server == 1))
    return ScoreReading(int(t * 30), t, rows)


def _speeds(values: list[int], dt: float = 0.5, t0: float = 0.0) -> list[SpeedReading]:
    return [SpeedReading(i, t0 + i * dt, v) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------------------------------------
# end_changes

def test_end_changes_in_one_set_follow_after_games_one_three_five() -> None:
    # g games played give (g + 1) // 2 changes: after game 1, 3, 5 ...; the sides' split of g does not matter.
    got = [end_changes(_score(0, (g,), (0,))) for g in range(0, 14)]
    assert got == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    assert end_changes(_score(0, (3,), (2,))) == end_changes(_score(0, (5,), (0,))) == 3
    assert end_changes(_score(0, (2,), (1,))) == end_changes(_score(0, (1,), (2,))) == 2


def test_end_changes_add_up_over_every_set_shown() -> None:
    assert end_changes(_score(0, (6, 3), (4, 6))) == 5 + 5  # 10 games, then 9 games
    assert end_changes(_score(0, (6, 0), (4, 0))) == 5  # a new set with no game yet adds nothing
    assert end_changes(_score(0, (6, 6, 1), (0, 6, 0))) == 3 + 6 + 1  # an odd set (6-0) still changes at its end
    assert end_changes(_score(0, (6, 4, 2), (3, 6, 2))) == 5 + 5 + 2


def test_end_changes_ignore_a_set_present_in_only_one_row() -> None:
    # zip stops at the shorter row: the extra column is not counted (the parser rejects such readings anyway)
    r = ScoreReading(0, 0.0, (ScoreRow("A", (6, 3), "0"), ScoreRow("B", (4,), "0")))
    assert end_changes(r) == 5


def test_end_changes_in_a_tiebreak_ignore_the_tiebreak_points() -> None:
    # Known issue (docs/ARCHITECTURE.md): a tiebreak counts as one game (7-6 -> 13 games -> 7 changes), but players
    # also change ends after every 6 tiebreak points, which the games columns cannot show. Pinned as it is.
    assert end_changes(_score(0, (7,), (6,))) == end_changes(_score(0, (6,), (7,))) == 7
    assert end_changes(_score(0, (6,), (6,))) == 6  # 6-6 before and during the tiebreak: no change for its points


# ---------------------------------------------------------------------------------------------------------
# to_kmh

def test_mph_is_multiplied_and_kmh_passes_through_as_a_float() -> None:
    assert MPH_TO_KMH == 1.609344
    assert to_kmh(100, "mph") == pytest.approx(160.9344)
    assert to_kmh(0, "mph") == 0.0
    assert to_kmh(100, "km/h") == 100.0 and isinstance(to_kmh(100, "km/h"), float)
    assert to_kmh(-5, "mph") == pytest.approx(-8.04672)  # no clamping: the board reader gates the range


def test_an_unknown_or_oddly_spelt_unit_gives_no_value() -> None:
    assert to_kmh(100, None) is None
    assert to_kmh(100, "MPH") is None and to_kmh(100, "kmh") is None and to_kmh(100, "") is None


# ---------------------------------------------------------------------------------------------------------
# reading_near

def test_reading_near_takes_the_closest_reading_with_a_marked_server() -> None:
    scores = [_score(0.0), _score(10.0), _score(20.0)]
    assert reading_near(scores, 9.0) is scores[1]
    assert reading_near(scores, 4.9) is scores[0] and reading_near(scores, 5.1) is scores[1]
    assert reading_near([], 5.0) is None


def test_reading_near_window_is_inclusive_and_adjustable() -> None:
    scores = [_score(10.0)]
    assert reading_near(scores, 18.0) is scores[0]  # exactly 8 s away
    assert reading_near(scores, 18.01) is None
    assert reading_near(scores, 2.0) is scores[0] and reading_near(scores, 1.99) is None
    assert reading_near(scores, 13.0, window=3.0) is scores[0]
    assert reading_near(scores, 13.5, window=3.0) is None


def test_reading_near_skips_graphics_without_a_server_marker_even_if_closer() -> None:
    marked, unmarked = _score(5.0), _score(9.9, server=None)
    assert reading_near([marked, unmarked], 10.0) is marked
    assert reading_near([unmarked], 10.0) is None
    both_false = ScoreReading(0, 10.0, (ScoreRow("A", (0,), "0", False), ScoreRow("B", (0,), "0", False)))
    assert reading_near([both_false], 10.0) is None  # serving False in both rows is no server either


def test_reading_near_on_a_tie_keeps_the_earlier_reading_in_the_list() -> None:
    a, b = _score(8.0), _score(12.0)
    assert reading_near([a, b], 10.0) is a and reading_near([b, a], 10.0) is b


# ---------------------------------------------------------------------------------------------------------
# side_names

def test_side_names_follow_the_parity_of_the_end_changes() -> None:
    start = {"ALCARAZ": "near", "SINNER": "far"}
    one = _score(0.0, (1,), (1,))  # 2 games: 1 change, the ends are swapped
    assert side_names(BroadcastScan(15, 1, scores=[one]), start, 0.0) == {"far": "ALCARAZ", "near": "SINNER"}
    two = _score(0.0, (2,), (2,))  # 2 changes: back at the start ends
    assert side_names(BroadcastScan(15, 1, scores=[two]), start, 0.0) == {"near": "ALCARAZ", "far": "SINNER"}
    zero = _score(0.0, (0,), (0,))
    assert side_names(BroadcastScan(15, 1, scores=[zero]), start, 0.0) == {"near": "ALCARAZ", "far": "SINNER"}


def test_side_names_need_start_ends_and_a_graphic_within_the_window() -> None:
    scan = BroadcastScan(15, 1, scores=[_score(10.0, (1,), (0,))])
    start = {"ALCARAZ": "near", "SINNER": "far"}
    assert side_names(scan, None, 10.0) is None and side_names(scan, {}, 10.0) is None
    assert side_names(BroadcastScan(15, 0), start, 10.0) is None
    assert side_names(scan, start, 18.0) == {"far": "ALCARAZ", "near": "SINNER"}  # exactly 8 s: still within
    assert side_names(scan, start, 18.01) is None
    assert side_names(scan, start, 14.0, window=3.0) is None and side_names(scan, start, 12.9, window=3.0)


def test_side_names_use_the_nearest_graphic_even_without_a_server_marker() -> None:
    # unlike reading_near, the games are all that is needed here
    far_marked = _score(0.0, (2,), (2,))
    near_unmarked = _score(9.0, (3,), (2,), server=None)
    scan = BroadcastScan(15, 2, scores=[far_marked, near_unmarked])
    start = {"ALCARAZ": "near", "SINNER": "far"}
    assert side_names(scan, start, 10.0) == {"far": "ALCARAZ", "near": "SINNER"}  # 5 games: 3 changes


def test_side_names_with_one_player_or_both_at_one_end_give_nothing() -> None:
    scan = BroadcastScan(15, 1, scores=[_score(0.0)])
    assert side_names(scan, {"ALCARAZ": "near"}, 0.0) is None  # the other player is not named
    assert side_names(scan, {"ALCARAZ": "near", "SINNER": "near"}, 0.0) is None


def test_side_names_on_a_tie_between_graphics_use_the_earlier_one() -> None:
    early, late = _score(8.0, (1,), (0,)), _score(12.0, (3,), (0,))
    start = {"ALCARAZ": "near", "SINNER": "far"}
    assert side_names(BroadcastScan(15, 2, scores=[early, late]), start, 10.0) == {"far": "ALCARAZ", "near": "SINNER"}
    assert side_names(BroadcastScan(15, 2, scores=[late, early]), start, 10.0) == {"near": "ALCARAZ", "far": "SINNER"}


# ---------------------------------------------------------------------------------------------------------
# serve_anchored_names

def _fl(i: int, start: int, hitter: str, kind: str = "rally") -> dict:
    return {"id": i, "start_frame": start, "hitter": hitter, "kind": kind}


def test_anchored_names_need_two_player_names() -> None:
    flights = [_fl(1, 10, "near", "serve")]
    assert serve_anchored_names(flights, {1: "ALCARAZ"}, None, cuts=[]) == {}
    assert serve_anchored_names(flights, {1: "ALCARAZ"}, (), cuts=[]) == {}  # type: ignore[arg-type]


def test_anchored_names_for_a_server_in_either_name_slot_name_the_other_player() -> None:
    flights = [_fl(1, 10, "far", "serve"), _fl(2, 20, "near")]
    got = serve_anchored_names(flights, {1: "SINNER"}, NAMES, cuts=[])
    assert got == {1: {"far": "SINNER", "near": "ALCARAZ"}, 2: {"far": "SINNER", "near": "ALCARAZ"}}
    got = serve_anchored_names(flights, {1: "ALCARAZ"}, NAMES, cuts=[])
    assert got[2] == {"far": "ALCARAZ", "near": "SINNER"}


def test_an_unrecognised_server_name_anchors_nothing_and_drops_the_previous_anchor() -> None:
    flights = [_fl(1, 10, "far", "serve"), _fl(2, 20, "near"), _fl(3, 30, "near", "serve"), _fl(4, 40, "far")]
    got = serve_anchored_names(flights, {1: "SINNER", 3: "CARAZ"}, NAMES, cuts=[])
    assert sorted(got) == [1, 2]  # the truncated name "CARAZ" is not in names: no anchor, flights 3 and 4 unnamed


def test_a_cut_or_new_score_ends_the_anchor_only_after_the_serve_up_to_the_flight() -> None:
    flights = [_fl(1, 100, "far", "serve"), _fl(2, 200, "near"), _fl(3, 300, "far")]
    names = {1: "SINNER"}
    # the serve's own frame is not "after" it; the flight's own frame is
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[100])) == [1, 2, 3]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[101])) == [1]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[200])) == [1]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[199])) == [1]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[], point_ends=[300])) == [1, 2]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[], point_ends=[301])) == [1, 2, 3]
    assert sorted(serve_anchored_names(flights, names, NAMES, cuts=[250], point_ends=[])) == [1, 2]


def test_a_new_serve_on_a_cut_frame_still_gets_its_own_anchor() -> None:
    flights = [_fl(1, 100, "far", "serve"), _fl(2, 300, "near", "serve"), _fl(3, 320, "far")]
    got = serve_anchored_names(flights, {1: "SINNER", 2: "ALCARAZ"}, NAMES, cuts=[300])
    assert got[2] == {"near": "ALCARAZ", "far": "SINNER"} and got[3] == got[2]


def test_anchored_names_sort_flights_by_start_frame_first() -> None:
    flights = [_fl(2, 200, "near"), _fl(1, 100, "far", "serve"), _fl(0, 50, "near")]
    got = serve_anchored_names(flights, {1: "SINNER"}, NAMES, cuts=[])
    assert sorted(got) == [1, 2]  # flight 0 (frame 50) precedes the serve even though it was listed last


# ---------------------------------------------------------------------------------------------------------
# check: serve speed against the board

def _board_scan(unit: str | None = "mph") -> BroadcastScan:
    seq = _speeds([90] * 4 + [120] * 6)  # 90 -> 120 mph between 1.5 s and 2.0 s
    return BroadcastScan(15, len(seq), speeds=seq, speed_unit=unit)


def test_a_serve_is_paired_when_it_lies_inside_the_lag_window_inclusive() -> None:
    # the update lies at 1.5-2.0 s: the serve may be from 6.5 s before 1.5 to 1.0 s before 2.0, i.e. -5.0 .. 1.0
    for t in (-5.0, 0.0, 1.0):
        out = check(_board_scan(), [Serve(1, t, "far", 200.0)])["serve_speed"]
        assert out["n"] == 1, t
    for t in (-5.01, 1.01):
        out = check(_board_scan(), [Serve(1, t, "far", 200.0)])["serve_speed"]
        assert out["n"] == 0 and out["unmatched_updates"][0]["candidates"] == 0, t


def test_the_lag_window_can_be_changed() -> None:
    out = check(_board_scan(), [Serve(1, 1.4, "far", 200.0)], lag=(0.5, 2.0))["serve_speed"]
    assert out["n"] == 1 and out["pairs"][0]["lag_s"] == [0.1, 0.6]
    assert check(_board_scan(), [Serve(1, 1.4, "far", 200.0)])["serve_speed"]["n"] == 0


def test_a_pair_reports_the_board_in_kmh_the_lags_and_the_signed_error() -> None:
    out = check(_board_scan(), [Serve(5, 0.0, "far", 200.0)])["serve_speed"]
    assert out["pairs"] == [{"flight_id": 5, "t": 0.0, "ours_kmh": 200.0, "board": 120, "board_kmh": 193.1,
                             "lag_s": [1.5, 2.0], "error_pct": 3.6}]
    assert out["mean_error_pct"] == 3.6 and out["mean_abs_error_pct"] == 3.6
    under = check(_board_scan(), [Serve(5, 0.0, "far", 180.0)])["serve_speed"]
    assert under["pairs"][0]["error_pct"] == -6.8  # we measure lower than the board: negative
    assert under["mean_error_pct"] == -6.8 and under["mean_abs_error_pct"] == 6.8


def test_the_board_unit_decides_the_conversion() -> None:
    kmh = check(_board_scan("km/h"), [Serve(1, 0.0, "far", 200.0)])["serve_speed"]["pairs"][0]
    assert kmh["board_kmh"] == 120.0 and kmh["error_pct"] == 66.7
    unknown = check(_board_scan(None), [Serve(1, 0.0, "far", 200.0)])["serve_speed"]
    assert unknown["n"] == 0 and unknown["unmatched_updates"][0]["candidates"] == 1  # one serve, but no unit


def test_a_serve_without_a_speed_is_not_a_pair() -> None:
    out = check(_board_scan(), [Serve(1, 0.0, "far", None)], measurable=[(-6.0, 3.0)])["serve_speed"]
    assert out["n"] == 0 and out["unmatched_updates"][0]["reason"] == "missed"


def test_mean_errors_average_the_signed_and_the_absolute_errors() -> None:
    seq = _speeds([90] * 2 + [120] * 2 + [100] * 3 + [110] * 3)
    scan = BroadcastScan(15, len(seq), speeds=seq, speed_unit="km/h")
    # updates: 90 -> 120 (0.5-1.0 s), 120 -> 100 (1.5-2.0 s), 100 -> 110 (3.0-3.5 s, no serve in reach)
    out = check(scan, [Serve(1, -5.5, "far", 132.0), Serve(2, 0.5, "far", 90.0)])["serve_speed"]
    assert [(p["flight_id"], p["error_pct"]) for p in out["pairs"]] == [(1, 10.0), (2, -10.0)]
    assert out["n"] == 2 and out["mean_error_pct"] == 0.0 and out["mean_abs_error_pct"] == 10.0
    assert [u["board"] for u in out["unmatched_updates"]] == [110]


def test_one_serve_is_paired_with_one_update_only() -> None:
    seq = _speeds([90] * 2 + [120] * 2 + [100] * 3)  # 90 -> 120 at 0.5-1.0 s, 120 -> 100 at 1.5-2.0 s
    scan = BroadcastScan(15, len(seq), speeds=seq, speed_unit="mph")
    out = check(scan, [Serve(1, -3.0, "far", 200.0)])["serve_speed"]  # inside both windows
    assert [p["board"] for p in out["pairs"]] == [120]
    assert out["unmatched_updates"] == [{"t": [1.5, 2.0], "board": 100, "candidates": 0, "reason": "cut"}]


def test_an_update_window_of_exactly_four_seconds_is_still_pairable() -> None:
    first = _speeds([90] * 2) + _speeds([120] * 3, t0=4.5)  # seen 0.5 s ... 4.5 s: 4.0 s wide
    out = check(BroadcastScan(15, 5, speeds=first, speed_unit="mph"), [Serve(1, 0.0, "far", 200.0)])["serve_speed"]
    assert out["n"] == 1
    wide = _speeds([90] * 2) + _speeds([120] * 3, t0=4.6)
    out = check(BroadcastScan(15, 5, speeds=wide, speed_unit="mph"), [Serve(1, 0.0, "far", 200.0)])["serve_speed"]
    assert out["n"] == 0 and out["unmatched_updates"][0]["reason"] == "timing"


def test_unmatched_reasons_depend_on_candidates_and_the_measurable_span() -> None:
    def reason(serves: list[Serve], measurable: list[tuple[float, float]] | None) -> str:
        out = check(_board_scan(), serves, measurable=measurable)["serve_speed"]["unmatched_updates"]
        return out[0]["reason"]

    assert reason([], [(-5.0, 1.0)]) == "missed"  # the window [-5, 1] is exactly inside the span
    assert reason([], [(-4.99, 1.0)]) == "cut" and reason([], [(-5.0, 0.99)]) == "cut"
    assert reason([], None) == "cut" and reason([], []) == "cut"
    assert reason([], [(-100.0, -50.0), (-6.0, 3.0)]) == "missed"  # any one span may contain it
    two = [Serve(1, 0.0, "far", 150.0), Serve(2, 0.5, "far", 150.0)]
    assert reason(two, [(-9.0, 9.0)]) == "ambiguous" and reason(two, None) == "ambiguous"


def test_the_first_board_value_seen_is_not_an_update_to_pair() -> None:
    seq = _speeds([90] * 4)
    scan = BroadcastScan(15, 4, speeds=seq, speed_unit="mph")
    out = check(scan, [Serve(1, 0.0, "far", 200.0)])
    assert out["serve_speed"]["pairs"] == [] and out["serve_speed"]["unmatched_updates"] == []
    assert out["speed_updates"] == [{"t": [0.0, 0.0], "value": 90, "previous": None}]


# ---------------------------------------------------------------------------------------------------------
# check: server sides

def _serves_scan(rows: list[tuple[str, tuple[int, int], str]], names: tuple[str, str] | None = NAMES):
    """rows: (server name, (games0, games1), side we saw) per serve, one serve per 10 s."""
    scores, serves = [], []
    for i, (who, games, side) in enumerate(rows):
        server = 0 if who == NAMES[0] else 1
        scores.append(_score(10.0 * i, (games[0],), (games[1],), server=server))
        serves.append(Serve(i, 10.0 * i + 1.0, side, None))
    return BroadcastScan(15, len(scores), scores=scores, names=names), serves


def test_a_serve_without_a_graphic_in_reach_is_left_out_of_the_sides() -> None:
    scan = BroadcastScan(15, 1, scores=[_score(0.0)], names=NAMES)
    out = check(scan, [Serve(0, 20.0, "near", None)])["server_sides"]
    assert out == {"serves": [], "consistent": None, "agreement": None, "start_end": None}
    assert check(scan, [])["server_sides"]["consistent"] is None


def test_each_serve_row_names_the_server_the_end_changes_and_the_start_side() -> None:
    scan, serves = _serves_scan([("SINNER", (0, 0), "near"), ("ALCARAZ", (1, 0), "near"), ("SINNER", (3, 2), "far")])
    rows = check(scan, serves)["server_sides"]["serves"]
    assert rows == [
        {"flight_id": 0, "server": "SINNER", "side": "near", "end_changes": 0, "start_side": "near"},
        {"flight_id": 1, "server": "ALCARAZ", "side": "near", "end_changes": 1, "start_side": "far"},
        {"flight_id": 2, "server": "SINNER", "side": "far", "end_changes": 3, "start_side": "near"},
    ]


def test_players_who_serve_from_the_same_start_end_are_inconsistent_and_get_no_names() -> None:
    scan, serves = _serves_scan([("SINNER", (0, 0), "near"), ("ALCARAZ", (0, 0), "near")])
    ss = check(scan, serves)["server_sides"]
    assert ss["consistent"] is False and ss["agreement"] == 1.0 and ss["start_end"] is None


def test_unanimous_serves_of_two_players_at_opposite_ends_are_consistent() -> None:
    scan, serves = _serves_scan([("SINNER", (0, 0), "near"), ("ALCARAZ", (0, 0), "far"), ("SINNER", (2, 2), "near")])
    ss = check(scan, serves)["server_sides"]
    assert ss["consistent"] is True and ss["agreement"] == 1.0
    assert ss["start_end"] == {"SINNER": "near", "ALCARAZ": "far"}


def test_a_two_of_three_majority_is_enough_but_a_split_is_not() -> None:
    rows = [("SINNER", (0, 0), "near")] * 2 + [("SINNER", (0, 0), "far")]
    scan, serves = _serves_scan(rows)
    ss = check(scan, serves)["server_sides"]
    assert ss["start_end"] == {"SINNER": "near", "ALCARAZ": "far"}  # exactly 2/3, and the other player is inferred
    assert ss["agreement"] == 2 / 3 and ss["consistent"] is False
    scan, serves = _serves_scan(rows[:1] + rows[2:])  # 1 : 1
    assert check(scan, serves)["server_sides"]["start_end"] is None
    scan, serves = _serves_scan([("SINNER", (0, 0), "near")] * 2 + [("SINNER", (0, 0), "far")] * 2)
    assert check(scan, serves)["server_sides"]["start_end"] is None  # 2 of 4 is not 2/3


def test_a_name_outside_the_two_players_voids_the_assignment() -> None:
    scores = [_score(0.0, server=0, names=("SINNER", "ZVEREV")), _score(10.0, server=1, names=("SINNER", "ZVEREV"))]
    scan = BroadcastScan(15, 2, scores=scores, names=NAMES)
    out = check(scan, [Serve(0, 1.0, "near", None), Serve(1, 11.0, "far", None)])["server_sides"]
    assert out["start_end"] is None  # ZVEREV is not ALCARAZ: the two names of the match decide, nothing is guessed


def test_server_names_are_matched_to_the_players_by_containment() -> None:
    def server(read: str) -> str:
        scores = [_score(0.0, server=1, names=("SINNER", read))]
        scan = BroadcastScan(15, 1, scores=scores, names=NAMES)
        return check(scan, [Serve(0, 1.0, "near", None)])["server_sides"]["serves"][0]["server"]

    assert server("CARLOS ALCARAZ") == "ALCARAZ"  # the player's name is inside what was read
    assert server("CARAZ") == "ALCARAZ"  # what was read is inside the name (4+ letters)
    assert server("RAZ") == "RAZ"  # too short to trust
    assert server("ALCARAZ") == "ALCARAZ" and server("XYZ NOBODY") == "XYZ NOBODY"
    assert server("SINNER ALCARAZ") == "SINNER ALCARAZ"  # fits both players: left alone


def test_without_names_the_server_name_is_kept_as_read_and_no_names_are_assigned() -> None:
    scan, serves = _serves_scan([("SINNER", (0, 0), "near"), ("ALCARAZ", (0, 0), "far")], names=None)
    out = check(scan, serves)
    assert out["names"] is None and out["server_sides"]["start_end"] == {"SINNER": "near", "ALCARAZ": "far"}


def test_one_unanimous_player_names_the_other_by_the_flipped_end() -> None:
    scan, serves = _serves_scan([("ALCARAZ", (1, 0), "near")])
    assert check(scan, serves)["server_sides"]["start_end"] == {"ALCARAZ": "far", "SINNER": "near"}
    scan, serves = _serves_scan([("ALCARAZ", (1, 0), "near")], names=None)
    assert check(scan, serves)["server_sides"]["start_end"] == {"ALCARAZ": "far"}  # nobody to name the other


def test_the_report_carries_the_scan_header_and_changes() -> None:
    a, b = _score(0.0, (2,), (4,)), _score(1.0, (2,), (4,))
    c = ScoreReading(60, 2.0, (ScoreRow("SINNER", (2,), "15", True), ScoreRow("ALCARAZ", (4,), "0", False)))
    d = ScoreReading(75, 2.5, c.rows)
    scan = BroadcastScan(15, 99, scores=[a, b, c, d], speeds=_speeds([90] * 2), speed_unit="mph", names=NAMES)
    out = check(scan, [])
    assert (out["ocr"], out["frames_read"], out["step_frames"], out["names"]) == ("apple-vision", 99, 15, list(NAMES))
    assert out["score_readings"] == 4 and out["speed_unit"] == "mph"
    assert out["score_changes"] == [{"t": [1.0, 2.0], "before": [[[2], "0"], [[4], "0"]],
                                     "after": [[[2], "15"], [[4], "0"]], "winner": "SINNER", "n_points": 1}]
    assert out["speed_updates"] == [{"t": [0.0, 0.0], "value": 90, "previous": None}]


# =========================================================================================================
# render/hud.py

# ---- lean_in ---------------------------------------------------------------------------------------------

def test_lean_in_is_geometry_unless_the_receiver_played_on() -> None:
    assert lean_in({"p_in": 0.5}) is True  # exactly half leans in
    assert lean_in({"p_in": 0.4999}) is False and lean_in({"p_in": 0.0}) is False
    assert lean_in({"p_in": 0.9, "play_continued": False}) is True
    assert lean_in({"p_in": 0.1, "play_continued": True}) is True  # play on beats an outside-looking fit
    assert lean_in({"p_in": 0.1, "play_continued": None}) is False
    assert lean_in({"p_in": 0.1, "play_continued": 0}) is False


def test_lean_in_returns_a_real_bool_and_does_not_need_p_in_when_play_continued() -> None:
    assert lean_in({"p_in": 0.1, "play_continued": "yes"}) is True
    assert lean_in({"play_continued": True}) is True  # short-circuit: p_in is not read
    with pytest.raises(KeyError):
        lean_in({})


# ---- panel -----------------------------------------------------------------------------------------------

def _grey(h: int = 20, w: int = 30, v: int = 200) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


def test_panel_blends_in_place_only_inside_the_rectangle_in_bgr_order() -> None:
    f = _grey()
    assert panel(f, 5, 5, 10, 10, 0.5, (0, 0, 100)) is None
    assert f.shape == (20, 30, 3) and f.dtype == np.uint8
    assert f[5:10, 5:10].tolist() == [[[100, 100, 150]] * 5] * 5  # B, G, R each blended with its own colour value
    mask = np.ones(f.shape[:2], bool)
    mask[5:10, 5:10] = False
    assert (f[mask] == 200).all()  # nothing outside (the edges x1, y1 are exclusive)


def test_panel_blend_truncates_toward_zero_and_the_default_colour_is_the_panel_grey() -> None:
    f = _grey(v=255)
    panel(f, 0, 0, 30, 20, 0.72)
    assert f[0, 0].tolist() == [91, 88, 87]  # 255 * 0.28 + (28, 24, 22) * 0.72 = 91.56, 88.68, 87.24 cut, not rounded
    assert hud.PANEL == (28, 24, 22)


def test_panel_alpha_zero_leaves_the_frame_and_alpha_one_paints_the_colour() -> None:
    f = _grey()
    panel(f, 0, 0, 30, 20, 0.0, (1, 2, 3))
    assert (f == 200).all()
    panel(f, 0, 0, 30, 20, 1.0, (1, 2, 3))
    assert (f == np.array([1, 2, 3])).all()


def test_panel_is_clipped_to_the_frame_on_every_side() -> None:
    f = _grey(v=101)
    panel(f, -5, -5, 100, 100, 1.0, (1, 2, 3))
    assert (f == np.array([1, 2, 3])).all()
    g = _grey(v=101)
    panel(g, 25, 15, 100, 100, 1.0, (9, 9, 9))  # runs over the right and bottom edges
    assert (g == 9).all(axis=2).sum() == 5 * 5 and g.shape == (20, 30, 3)
    assert (g[:15] == 101).all() and (g[:, :25] == 101).all()


def test_panel_with_an_empty_or_inverted_or_far_away_rectangle_changes_nothing() -> None:
    for rect in ((10, 10, 10, 20), (10, 10, 5, 5), (40, 40, 60, 60)):
        f = _grey(v=101)
        panel(f, *rect, 1.0, (9, 9, 9))
        assert (f == 101).all(), rect


def test_panel_with_a_rectangle_left_of_or_above_the_frame_paints_nothing() -> None:
    # A negative end used to become a slice from the end and paint all but the last 10 rows and columns.
    for rect in ((-30, -30, -10, -10), (-30, 5, -10, 15), (5, -30, 15, -10)):
        f = _grey(v=101)
        panel(f, *rect, 1.0, (9, 9, 9))
        assert (f == 101).all(), rect
    g = _grey(v=101)
    panel(g, -30, -30, 0, 0, 1.0, (9, 9, 9))  # ending exactly at 0 is empty, as it should be
    assert (g == 101).all()


# ---- draw_texts ------------------------------------------------------------------------------------------

def _black(h: int = 100, w: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def _ink(img: np.ndarray) -> np.ndarray:
    return np.argwhere(img.any(axis=2))


def test_draw_texts_without_items_hands_back_the_same_frame() -> None:
    f = _black()
    assert draw_texts(f, []) is f


def test_draw_texts_returns_a_new_frame_of_the_same_shape_and_dtype_and_leaves_the_input_alone() -> None:
    f = _black()
    out = draw_texts(f, [TextItem((20, 30), "Merhaba", 30)])
    assert out is not f and out.shape == f.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert not f.any() and out.any()


def test_draw_texts_ink_stays_near_the_text_position() -> None:
    out = draw_texts(_black(), [TextItem((100, 30), "Merhaba", 30)])
    ys, xs = _ink(out)[:, 0], _ink(out)[:, 1]
    assert xs.min() >= 98 and xs.max() <= 100 + 7 * 30  # to the right of the anchor, within the string's width
    assert ys.min() >= 28 and ys.max() <= 30 + 2 * 30  # below the top of the anchor, within about two em


def test_draw_texts_takes_the_colour_in_bgr_and_paints_it_in_the_right_channels() -> None:
    red = draw_texts(_black(), [TextItem((20, 30), "Merhaba", 30, (0, 0, 255))])
    assert red[..., 2].max() == 255 and red[..., 0].max() == 0 and red[..., 1].max() == 0
    blue = draw_texts(_black(), [TextItem((20, 30), "Merhaba", 30, (255, 0, 0))])
    assert blue[..., 0].max() == 255 and blue[..., 1].max() == 0 and blue[..., 2].max() == 0
    default = draw_texts(_black(), [TextItem((20, 30), "Merhaba", 30)])  # WHITE
    assert default[..., 0].max() == default[..., 1].max() == default[..., 2].max() == 245


def test_draw_texts_bigger_sizes_cover_more_pixels_and_longer_text_is_wider() -> None:
    n = [len(_ink(draw_texts(_black(), [TextItem((0, 0), "Merhaba", s)]))) for s in (10, 20, 40)]
    assert n[0] < n[1] < n[2]
    short = _ink(draw_texts(_black(), [TextItem((0, 0), "Me", 30)]))[:, 1].max()
    long_ = _ink(draw_texts(_black(), [TextItem((0, 0), "Merhaba dünya", 30)]))[:, 1].max()
    assert long_ > 2 * short


def test_draw_texts_renders_turkish_letters_the_dotted_capital_and_the_hud_symbols_without_error() -> None:
    for text in ("İÇERİDE", "DIŞARIDA?", "ışık ğüöç ŞĞ", "SKOR · YAYINDAN OKUNDU", "± 12 cm (1σ)", "—", "İNCELEME"):
        out = draw_texts(_black(), [TextItem((20, 30), text, 30)])
        assert out.shape == (100, 300, 3) and out.any(), text


def test_draw_texts_with_empty_or_fully_outside_text_changes_nothing() -> None:
    f = _black()
    for item in (TextItem((20, 30), "", 30), TextItem((-500, 30), "Merhaba", 30), TextItem((500, 30), "Merhaba", 30),
                 TextItem((20, 500), "Merhaba", 30), TextItem((20, -500), "Merhaba", 30)):
        assert not draw_texts(f, [item]).any(), item


def test_draw_texts_that_does_not_fit_is_cut_at_the_frame_edge_without_growing_the_frame() -> None:
    f = _black(60, 120)
    out = draw_texts(f, [TextItem((60, 20), "Bu yazı sığmıyor, çok uzun bir cümle", 30)])
    assert out.shape == (60, 120, 3) and out.any()
    assert out[:, :55].sum() == 0  # nothing to the left of where the text starts
    out2 = draw_texts(f, [TextItem((-40, -10), "Merhaba", 30)])  # starts off-frame top-left: the rest shows
    assert out2.shape == (60, 120, 3) and out2.any()


def test_draw_texts_paints_every_item() -> None:
    out = draw_texts(_black(), [TextItem((10, 5), "Aa", 20, (0, 0, 255)), TextItem((10, 60), "Bb", 20, (255, 0, 0))])
    top, bottom = out[:50], out[50:]
    assert top[..., 2].max() > 0 and top[..., 0].max() == 0
    assert bottom[..., 0].max() > 0 and bottom[..., 2].max() == 0


# ---- fonts -----------------------------------------------------------------------------------------------

def test_font_is_cached_per_size_and_grows_with_size() -> None:
    assert hud._font(20) is hud._font(20) and hud._font(20) is not hud._font(21)
    assert hud._font(20).getlength("Merhaba") < hud._font(40).getlength("Merhaba")


def test_font_falls_back_when_no_candidate_loads(monkeypatch) -> None:
    hud._font.cache_clear()
    try:
        monkeypatch.setattr(hud, "_FONT_CANDIDATES", [("/no/such/font.ttc", 0)])
        missing = hud._font(24)
        assert missing.getlength("Merhaba") > 0
        hud._font.cache_clear()

        real = hud.ImageFont.truetype

        def broken(font, *a, **k):
            if isinstance(font, str):
                raise OSError("cannot open resource")
            return real(font, *a, **k)  # the default font loads its own bytes through the same call

        monkeypatch.setattr(hud, "_FONT_CANDIDATES", [(__file__, 0)])  # exists, but is not a font
        monkeypatch.setattr(hud.ImageFont, "truetype", broken)
        assert hud._font(24).getlength("Merhaba") > 0  # the OSError falls through to the default font
        out = draw_texts(_black(), [TextItem((10, 10), "İÇERİDE", 24)])
        assert out.any()
    finally:
        hud._font.cache_clear()


# ---- ground_ellipse_px -----------------------------------------------------------------------------------

@pytest.fixture
def broadcast_camera():
    """tests/conftest.py's broadcast camera (rebuilt here so the file also runs from outside tests/)."""
    try:
        from conftest import look_at_camera
    except ImportError:  # pragma: no cover - conftest not on the path
        from courtvision.calib.camera import Camera

        def look_at_camera(center, target, f, size):
            fwd = (target - center) / np.linalg.norm(target - center)
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            right = right / np.linalg.norm(right)
            R = np.stack([right, np.cross(fwd, right), fwd])
            K = np.array([[f, 0.0, size[0] / 2], [0.0, f, size[1] / 2], [0.0, 0.0, 1.0]])
            return Camera(K=K, R=R, t=-R @ center)

    return look_at_camera(np.array([0.0, -28.0, 11.0]), np.array([0.0, 1.5, 0.0]), 1750.0, (1920, 1080))


def _unit(deg: float) -> np.ndarray:
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])


def _mahalanobis_sq(H: np.ndarray, ring: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    world = apply_h(np.linalg.inv(H), ring) - mean
    return np.einsum("ij,jk,ik->i", world, np.linalg.inv(cov), world)


def test_ground_ellipse_is_a_closed_ring_of_48_image_points(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    ring = ground_ellipse_px(H, np.array([1.0, 10.0]), np.diag([0.04, 0.09]))
    assert ring.shape == (48, 2) and np.isfinite(ring).all()
    assert np.allclose(ring[0], ring[-1])  # the angle runs 0 .. 2 pi inclusive


def test_ground_ellipse_points_lie_at_the_95_percent_distance_back_on_the_ground(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    mean = np.array([1.0, 10.0])
    k = 2.4477
    tilted = [4.0 * np.outer(_unit(a), _unit(a)) + 0.25 * np.outer(_unit(a + 90), _unit(a + 90))
              for a in (20, 45, 70, 110, 135, 160)]  # eigenvector matrices of both handedness, whatever LAPACK picks
    for cov in (np.diag([0.04, 0.09]), np.array([[0.30, 0.12], [0.12, 0.10]]), np.diag([1.0, 0.01]), *tilted):
        ring = ground_ellipse_px(H, mean, cov)
        assert np.allclose(_mahalanobis_sq(H, ring, mean, cov), k * k, rtol=1e-6), cov
        ring1 = ground_ellipse_px(H, mean, cov, k=1.0)
        assert np.allclose(_mahalanobis_sq(H, ring1, mean, cov), 1.0, rtol=1e-6), cov


def test_ground_ellipse_follows_the_axis_lengths_and_the_tilt_of_the_covariance(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    mean = np.array([0.0, 8.0])
    k = 2.4477
    world = apply_h(np.linalg.inv(H), ground_ellipse_px(H, mean, np.diag([4.0, 0.25]))) - mean
    assert world[:, 0].max() == pytest.approx(2 * k, rel=1e-3) and world[:, 1].max() == pytest.approx(0.5 * k, rel=1e-3)
    for deg in (30, 60, 120, 150):
        u, v = _unit(deg), _unit(deg + 90)
        cov = 4.0 * np.outer(u, u) + 0.25 * np.outer(v, v)
        world = apply_h(np.linalg.inv(H), ground_ellipse_px(H, mean, cov)) - mean
        # the ring has 48 samples: its maximum along an axis falls short by at most 1 - cos(3.9 deg) = 0.2 %
        assert (world @ u).max() == pytest.approx(2 * k, rel=5e-3), deg
        assert (world @ v).max() == pytest.approx(0.5 * k, rel=5e-3), deg


def test_ground_ellipse_matches_the_cameras_own_projection_of_the_ground_point(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    mean = np.array([-2.0, 6.0])
    ring = ground_ellipse_px(H, mean, np.diag([0.25, 0.25]))
    centre = broadcast_camera.project(np.array([mean[0], mean[1], 0.0]))
    poly = ring.astype(np.float32).reshape(-1, 1, 2)
    assert cv2.pointPolygonTest(poly, (float(centre[0]), float(centre[1])), False) > 0
    # the same circle is smaller in the image when it lies further from the camera
    far = ground_ellipse_px(H, np.array([-2.0, 20.0]), np.diag([0.25, 0.25]))
    assert np.ptp(far[:, 0]) < np.ptp(ring[:, 0])


def test_ground_ellipse_of_a_zero_or_slightly_negative_covariance_collapses_without_nan(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    mean = np.array([1.0, 10.0])
    centre = broadcast_camera.project(np.array([1.0, 10.0, 0.0]))
    point = ground_ellipse_px(H, mean, np.zeros((2, 2)))
    assert np.allclose(point, centre)  # a point: every ring vertex on the projected mean
    line = ground_ellipse_px(H, mean, np.diag([1.0, -1e-9]))  # a rounding-negative eigenvalue is clipped to 0
    assert np.isfinite(line).all()
    world = apply_h(np.linalg.inv(H), line)
    assert np.allclose(world[:, 1], 10.0, atol=1e-6) and np.ptp(world[:, 0]) > 4.0  # flat along x


# ---- Renderer --------------------------------------------------------------------------------------------

def _state(**kw) -> OverlayState:
    base = dict(frame_idx=0, t=12.34, calib_ok=False, H=None, calib_note="", ball_px=None, ball_trail=[], players={},
                last_shot_speed=None, bounces=[], latest_call=None)
    base.update(kw)
    return OverlayState(**base)


def test_renderer_units_scale_with_the_frame_height_and_never_drop_below_one() -> None:
    assert Renderer(1920, 1080, CourtModel())._u(24) == 24
    small = Renderer(960, 540, CourtModel())
    assert small._u(24) == 12 and small._u(1) == 1 and small._u(0.1) == 1
    assert Renderer(3840, 2160, CourtModel())._u(24) == 48


def test_the_overlays_keep_shape_and_dtype_leave_the_input_alone_and_spare_the_middle_of_the_frame() -> None:
    frame = np.full((1080, 1920, 3), 80, np.uint8)
    keep = frame.copy()
    r = Renderer(1920, 1080, CourtModel())
    for fn in (r.classic, r.pro):
        out = fn(frame, _state())  # no calibration, no ball, no players, no call
        assert out.shape == frame.shape and out.dtype == np.uint8 and out is not frame
        assert (frame == keep).all()
        assert (out[100:980, :1200] == 80).all()  # left of the mini court and its badge, above the bottom bar
        assert (out[560:988] == 80).all()  # below the mini court, above the bar


def test_without_a_calibration_the_mini_court_and_the_bottom_bar_are_the_only_things_drawn() -> None:
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    out = Renderer(1920, 1080, CourtModel()).classic(frame, _state())
    above = (out[:988] != 200).any(axis=2)
    rows, cols = np.nonzero(above.any(axis=1))[0], np.nonzero(above.any(axis=0))[0]
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (54, 506, 1633, 1881)  # the mini court (top right)
    assert (out[987] == 200).all()  # the bar starts at row 1080 - 92
    assert out[988, 5].tolist() == [53, 50, 48] and out[1079, 5].tolist() == [53, 50, 48]  # 200 * 0.15 + PANEL * 0.85
    pro = Renderer(1920, 1080, CourtModel()).pro(frame, _state())
    above = (pro[:988] != 200).any(axis=2)
    badge = np.nonzero(above.any(axis=0))[0].min()  # pro adds the "no court lock" badge left of the mini court
    assert 1000 < badge < 1633  # its width depends on the font (1209 with the macOS font, 1132 with Pillow's)
    assert np.nonzero(above.any(axis=1))[0].min() == 54


def test_overlays_with_every_turkish_label_present_render_without_error_on_a_calibrated_frame(broadcast_camera) -> None:
    H = broadcast_camera.ground_homography()
    frame = np.full((1080, 1920, 3), 60, np.uint8)
    players = {"near": {"box": np.array([800.0, 500.0, 900.0, 700.0]), "world": np.array([0.0, -10.0]),
                        "speed_kmh": 12.34, "dist_m": 120.5, "peak_kmh": 25.0, "track_id": 1},
               "far": {"box": None, "world": None, "speed_kmh": None, "dist_m": None, "peak_kmh": None,
                       "track_id": 2}}
    bounce = {"xy": np.array([2.0, 11.0]), "cov": np.diag([0.04, 0.09]), "verdict": "REVIEW", "p_in": 0.4,
              "margin_m": -0.02, "sigma_m": 0.04, "line": "far baseline", "age": 3}
    call = {"verdict": "REVIEW", "p_in": 0.4, "margin_m": -0.02, "sigma_m": 0.04, "line": "far baseline",
            "speed_kmh": 187.0, "net_clearance_m": 0.31, "net_clearance_sd_m": 0.05, "play_continued": True}
    score = [{"name": "ŞİMŞEK", "games": [6, 3], "points": "15", "serving": True},
             {"name": "ÇAĞLAYAN ÖZGÜR UZUN", "games": [4, 2], "points": None, "serving": False}]
    st = _state(calib_ok=True, H=H, calib_note="not 1,2 px", ball_px=np.array([900.0, 400.0]),
                ball_trail=[np.array([880.0, 420.0]), np.array([890.0, 410.0]), np.array([900.0, 400.0])],
                players=players, last_shot_speed=187.4, bounces=[bounce], latest_call=call, score=score,
                names={"near": "ışıl", "far": "İpek"})
    r = Renderer(1920, 1080, CourtModel())
    for fn in (r.classic, r.pro):
        out = fn(frame, st)
        assert out.shape == frame.shape and out.dtype == np.uint8 and (out != 60).any()
    # the bounce marker's cross is drawn on the projected bounce point, nothing is drawn far from the court
    pro = r.pro(frame, st)
    cx, cy = (int(round(v)) for v in apply_h(H, bounce["xy"]))
    assert (pro[cy - 3:cy + 4, cx - 3:cx + 4] != 60).any()
    assert (pro[300:330, 1500:1560] == 60).all()
