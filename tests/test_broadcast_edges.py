"""Score graphic and speed board at their edges (broadcast/scoreboard.py): point steps across deuce, games and
sets, score changes across gaps and glitches, three sets, the serving marker, the speed board's plausible range,
flickers and name ties. Values checked by running the code (no OCR needed)."""

from __future__ import annotations

import numpy as np

from courtvision.broadcast.ocr import TextBox
from courtvision.broadcast.scoreboard import (
    ScoreReading,
    ScoreRow,
    SpeedReading,
    _drop_flickers,
    _point_step,
    majority_names,
    parse_scoreboard,
    parse_speed,
    score_changes,
)


def _reading(t: float, g0: int, p0: str | None, g1: int, p1: str | None) -> ScoreReading:
    return ScoreReading(int(t * 30), t, (ScoreRow("A", (g0,), p0), ScoreRow("B", (g1,), p1)))


def board(top, bot, mk0: str = "", mk1: str = "") -> list[TextBox]:
    """Text boxes of a two-row graphic laid out like the US Open's (names, then one column per value)."""
    b = [TextBox(mk0 + "MEDVEDEV", 1, .06, .833, .1, .028), TextBox(mk1 + "DJOKOVIC", 1, .06, .872, .1, .028)]
    for i, (a, c) in enumerate(zip(top, bot, strict=True)):
        x = .26 + .03 * i
        b += [TextBox(str(a), 1, x, .836, .011, .022), TextBox(str(c), 1, x, .875, .011, .025)]
    return b


def test_point_steps_across_deuce_games_and_sets() -> None:
    assert _point_step(_reading(0, 2, "AD", 4, "40"), _reading(1, 3, "0", 4, "0")) == (0, 1)  # advantage -> game
    assert _point_step(_reading(0, 2, "40", 4, "40"), _reading(1, 2, "AD", 4, "40")) == (0, 1)  # deuce -> advantage
    assert _point_step(_reading(0, 2, "40", 4, "40"), _reading(1, 3, "0", 4, "0")) == (None, None)  # two points
    assert _point_step(_reading(0, 2, "30", 4, "0"), _reading(1, 2, "15", 4, "0")) == (None, None)  # went back
    sa = ScoreReading(0, 0., (ScoreRow("A", (6, 5), "40"), ScoreRow("B", (4, 5), "15")))
    sb = ScoreReading(1, 1., (ScoreRow("A", (6, 5, 0), "0"), ScoreRow("B", (4, 5, 0), "0")))
    assert _point_step(sa, sb) == (None, None)  # a set ended: the layout changed


def test_score_changes_across_gaps_and_glitches() -> None:
    a, b = (2, "15", 4, "30"), (2, "30", 4, "30")
    seq = [_reading(0, *a), _reading(.5, *a), _reading(5, *b), _reading(5.5, *b)]
    assert [(c.t_before, c.t_after, c.winner) for c in score_changes(seq)] == [(0.5, 5, 0)]  # 4.5 s apart: paired
    seq = [_reading(0, *a), _reading(2, *a), _reading(2.5, *a), _reading(3, *b), _reading(3.5, *b)]
    assert [(c.t_before, c.t_after) for c in score_changes(seq)] == [(2.5, 3)]  # a gap over max_gap splits a run
    z, y = (1, "0", 1, "0"), (1, "15", 1, "0")
    seq = [_reading(0, *z), _reading(.5, *z), _reading(1, *y), _reading(1.5, *z), _reading(2, *z)]
    assert score_changes(seq) == []  # one glitch reading between equal states


def test_three_set_columns_with_and_without_points() -> None:
    r = parse_scoreboard(board([6, 3, 4, "15"], [4, 6, 4, "40"]), None, 0, 0.)
    assert [x.games for x in r.rows] == [(6, 3, 4), (4, 6, 4)] and [x.points for x in r.rows] == ["15", "40"]
    r = parse_scoreboard(board([6, 3, 4], [4, 6, 5]), None, 0, 0.)
    assert r.rows[0].points is None and r.rows[1].games == (4, 6, 5)


def test_the_serving_marker() -> None:
    def serving(boxes, img=None):
        return [x.serving for x in parse_scoreboard(boxes, img, 0, 0.).rows]

    assert serving(board([2, "0"], [4, "0"], "●")) == [True, False]
    assert serving(board([2, "0"], [4, "0"])) == [None, None]  # no marker
    assert serving(board([2, "0"], [4, "0"], "●", "●")) == [None, None]  # both marked: unknown
    img = np.zeros((720, 1280, 3), np.uint8)
    img[630:640, 58:70] = 255  # a bright blob left of the bottom name
    assert serving(board([2, "0"], [4, "0"]), img) == [False, True]


def test_the_speed_boards_plausible_range() -> None:
    def read(s):
        return parse_speed([TextBox(s, 1, .3, .1, .05, .03)], 0, 0.)

    assert read("165MPH")[0].value == 165 and read("166MPH") == (None, "mph")
    assert read("50 MPH")[0].value == 50 and read("49MPH") == (None, "mph")
    assert read("265 KM/H")[0].value == 265 and read("79KMH") == (None, "km/h")
    lab = [TextBox("SERVE SPEED", 1, .3, .14, .03, .014), TextBox("190", 1, .3, .125, .02, .02)]
    assert parse_speed(lab, 0, 0.)[0].value == 190 and parse_speed(lab, 0, 0.)[1] is None  # no unit: widest range
    assert parse_speed(lab + [TextBox("MPH", 1, .33, .13, .01, .01)], 0, 0.) == (None, "mph")  # no 190 mph serve
    far = [TextBox("SERVE SPEED", 1, .3, .14, .03, .014), TextBox("120", 1, .8, .8, .02, .02)]
    assert parse_speed(far, 0, 0.) == (None, None)  # a number away from the label is not its value


def test_flickers_on_the_board() -> None:
    def s(v):
        return SpeedReading(0, 0, v)

    assert len(_drop_flickers([(0, 1, s(1)), (2, 3, s(2)), (6.0, 7, s(1))], 4.0)) == 3  # back after 4.0 s: kept
    assert len(_drop_flickers([(0, 1, s(1)), (2, 3, s(2)), (3.9, 5, s(1))], 4.0)) == 1  # back within 4 s: a flicker
    chain = _drop_flickers([(0, 1, s(1)), (2, 3, s(2)), (4, 5, s(1)), (5, 6, s(2)), (6, 7, s(1))], 4.0)
    assert [(a, b, c.value) for a, b, c in chain] == [(0, 7, 1)]
    assert _drop_flickers([], 4.0) == [] and len(_drop_flickers([(0, 1, s(1)), (2, 3, s(2))], 4.0)) == 2


def test_majority_names_on_a_tie_and_with_none() -> None:
    def mk(a, b):
        return ScoreReading(0, 0., (ScoreRow(a, (0,), None), ScoreRow(b, (0,), None)))

    assert majority_names([]) is None
    assert majority_names([mk("A", "X"), mk("B", "Y")]) == ("A", "X")  # a tie: the first seen
    assert majority_names([mk("B", "Y"), mk("A", "X")]) == ("B", "Y")
