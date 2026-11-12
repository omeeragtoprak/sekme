"""Score graphic and speed board parsing, on text boxes taken from real Vision output (no OCR needed)."""

from courtvision.broadcast.ocr import TextBox
from courtvision.broadcast.scoreboard import (
    ScoreReading,
    ScoreRow,
    SpeedReading,
    _point_step,
    parse_scoreboard,
    parse_speed,
    score_changes,
    speed_updates,
)

# US Open 2023 final, frame 1602 (Vision boxes, normalised, origin top-left)
US_OPEN = [
    TextBox("SET POINT #3", 1.0, 0.069, 0.797, 0.089, 0.025),
    TextBox("3 MEDVEDEV", 1.0, 0.061, 0.833, 0.102, 0.028),
    TextBox("2 DJOKOVIC SRB", 1.0, 0.059, 0.872, 0.12, 0.028),
    TextBox("3", 1.0, 0.259, 0.836, 0.011, 0.022),
    TextBox("5", 1.0, 0.259, 0.875, 0.011, 0.025),
    TextBox("15", 1.0, 0.278, 0.833, 0.019, 0.025),
    TextBox("40", 1.0, 0.277, 0.875, 0.02, 0.025),
    TextBox("SERVE SPEED", 1.0, 0.314, 0.142, 0.027, 0.014),
    TextBox("86", 1.0, 0.316, 0.125, 0.016, 0.019),
    TextBox("Emirates Airline", 1.0, 0.205, 0.061, 0.069, 0.019),
]


def test_score_rows_are_parsed_and_the_country_code_is_dropped() -> None:
    r = parse_scoreboard(US_OPEN, None, 1602, 53.45)
    assert r is not None
    top, bot = r.rows
    assert (top.name, top.games, top.points) == ("MEDVEDEV", (3,), "15")
    assert (bot.name, bot.games, bot.points) == ("DJOKOVIC", (5,), "40")


def test_a_marker_glyph_marks_the_server_and_advantage_implies_forty() -> None:
    boxes = [
        TextBox("Djokovic", 1.0, 0.25, 0.70, 0.06, 0.025),
        TextBox("▸ Sonego", 1.0, 0.245, 0.735, 0.065, 0.025),
        TextBox("2", 1.0, 0.33, 0.70, 0.01, 0.025), TextBox("1", 1.0, 0.345, 0.70, 0.01, 0.025),
        TextBox("6", 1.0, 0.33, 0.735, 0.01, 0.025), TextBox("4", 1.0, 0.345, 0.735, 0.01, 0.025),
        TextBox("Ad", 1.0, 0.36, 0.735, 0.015, 0.025),
    ]
    r = parse_scoreboard(boxes, None, 0, 0.0)
    assert r is not None
    assert [x.serving for x in r.rows] == [False, True]
    assert r.rows[0].points == "40" and r.rows[1].points == "AD" and r.rows[1].games == (6, 4)


def test_a_reading_missing_a_set_column_in_one_row_is_rejected() -> None:
    boxes = [b for b in US_OPEN if b.text != "5"]  # Djokovic's games digit not recognised
    assert parse_scoreboard(boxes, None, 0, 0.0) is None


def _reading(t: float, g0: int, p0: str | None, g1: int, p1: str | None) -> ScoreReading:
    return ScoreReading(int(t * 30), t, (ScoreRow("A", (g0,), p0), ScoreRow("B", (g1,), p1)))


def test_point_winner_logic_including_deuce_and_games() -> None:
    assert _point_step(_reading(0, 2, "15", 4, "30"), _reading(1, 2, "30", 4, "30")) == (0, 1)
    assert _point_step(_reading(0, 2, "AD", 4, "40"), _reading(1, 2, "40", 4, "40")) == (1, 1)
    assert _point_step(_reading(0, 2, "40", 4, "15"), _reading(1, 3, "0", 4, "0")) == (0, 1)
    assert _point_step(_reading(0, 2, "15", 4, "0"), _reading(1, 2, "40", 4, "15")) == (None, 3)
    assert _point_step(_reading(0, 2, "30", 4, "40"), _reading(1, 3, "15", 5, "40")) == (None, None)


def test_changes_need_two_matching_readings() -> None:
    seq = [_reading(0.0, 2, "15", 4, "30"), _reading(0.5, 2, "15", 4, "30"), _reading(1.0, 9, "40", 9, "40"),
           _reading(1.5, 2, "30", 4, "30"), _reading(2.0, 2, "30", 4, "30")]
    ch = score_changes(seq)
    assert len(ch) == 1 and ch[0].winner == 0 and ch[0].t_before == 0.5 and ch[0].t_after == 1.5


def test_speed_board_value_unit_and_updates() -> None:
    sp, unit = parse_speed(US_OPEN + [TextBox("MPH", 1.0, 0.33, 0.13, 0.01, 0.01)], 1602, 53.45)
    assert sp is not None and sp.value == 86 and unit == "mph"
    seq = [SpeedReading(i, i * 0.5, v) for i, v in enumerate([125, 125, 93, 125, 93, 93, 93])]
    ups = speed_updates(seq)
    assert [(u.previous, u.value) for u in ups] == [(None, 125), (125, 93)]
    assert ups[1].t_after == 2.0


def test_an_unmatched_board_update_is_a_miss_only_inside_a_measurable_span() -> None:
    from courtvision.broadcast.scoreboard import BroadcastScan
    from courtvision.eval.broadcast_check import Serve, check

    seq = [SpeedReading(i, i * 0.5, v) for i, v in enumerate([90] * 4 + [120] * 20 + [95] * 10)]
    scan = BroadcastScan(step_frames=15, frames_read=len(seq), speeds=seq, speed_unit="mph")
    # 90 -> 120 at 1.5-2.0 s: window [-5, 1] crosses the cut at 0.5 s; 120 -> 95 at 11.5-12 s: inside.
    out = check(scan, [], measurable=[(0.5, 20.0)])
    assert [u["reason"] for u in out["serve_speed"]["unmatched_updates"]] == ["cut", "missed"]
    out = check(scan, [Serve(7, 0.8, "far", 160.0)], measurable=[(0.5, 20.0)])
    assert out["serve_speed"]["pairs"][0]["flight_id"] == 7 and out["serve_speed"]["pairs"][0]["board_kmh"] == 193.1


def test_speed_with_the_unit_glued_on_and_no_label_is_read():
    from courtvision.broadcast.scoreboard import parse_speed

    # US Open 2025 final: Vision returns "116MРL" (Cyrillic Р, H read as L) and misses "SERVE SPEED".
    sp, unit = parse_speed([TextBox("116MРL", 1.0, 0.31, 0.1, 0.04, 0.03), TextBox("0:29", 1.0, 0.6, 0.1, 0.02, 0.02)],
                           3612, 60.2)
    assert sp is not None and sp.value == 116 and unit == "mph"
    sp, unit = parse_speed([TextBox("201 KM/H", 1.0, 0.3, 0.1, 0.05, 0.03)], 1, 0.0)
    assert sp is not None and sp.value == 201 and unit == "km/h"
    # Implausible glued values (an advert's "20 KM") are not serve speeds.
    assert parse_speed([TextBox("20 KM", 1.0, 0.3, 0.1, 0.05, 0.03)], 1, 0.0)[0] is None
    # A bare number (a score, a clock) is not a speed.
    assert parse_speed([TextBox("30", 1.0, 0.3, 0.8, 0.02, 0.03)], 1, 0.0) == (None, None)


def test_a_board_update_with_an_unknown_time_is_not_paired():
    from courtvision.broadcast.scoreboard import BroadcastScan
    from courtvision.eval.broadcast_check import Serve, check

    # 90 mph seen until 2 s, then nothing legible until 120 mph at 20 s: the update could follow any serve.
    seq = [SpeedReading(i, i * 0.5, 90) for i in range(5)] + [SpeedReading(40 + i, 20 + i * 0.5, 120) for i in range(4)]
    scan = BroadcastScan(step_frames=15, frames_read=len(seq), speeds=seq, speed_unit="mph")
    out = check(scan, [Serve(3, 16.5, "far", 190.0)])
    assert out["serve_speed"]["pairs"] == []
    assert [u["reason"] for u in out["serve_speed"]["unmatched_updates"]] == ["timing"]
