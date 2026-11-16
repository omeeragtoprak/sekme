"""Pure helpers that no test named: the report JSON and the names/strokes attached to flights (pipeline.py), the
scoreboard's text and pixel helpers (broadcast/scoreboard.py) and the flight model's integrator, seed and projection
(physics/flight.py). Every expected value was run against the code first (nothing needs video, a model or a GPU).
Self-contained: the camera helper is a copy of conftest's."""

from __future__ import annotations

import copy
import json
import types

import cv2
import numpy as np
import pytest

from courtvision import pipeline
from courtvision.broadcast.ocr import TextBox
from courtvision.broadcast.scoreboard import (
    BroadcastScan,
    ScoreReading,
    ScoreRow,
    _is_speed_label,
    _marker_pixels,
    _name_parts,
    _split_tokens,
    _stable,
)
from courtvision.calib.camera import Camera
from courtvision.events.flights import Flight
from courtvision.events.strokes import Stroke
from courtvision.geometry.court import BALL_RADIUS, Half
from courtvision.officiating.line_call import LineCall
from courtvision.physics.flight import (
    DRAG_K,
    SPIN_PRIOR_SIGMA,
    Anchor,
    FlightFit,
    G,
    _project,
    _residuals_batch,
    flight_states,
    initial_guess,
    integrate,
)
from courtvision.sim.simulate import simulate_shot

# ------------------------------------------------------------------------------------------------
# pipeline._f, _flight_json


P_FIT = np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.1234567, -0.0004999])
COV = np.diag([0.04, 0.09] + [0.01] * 9)


def _fit(params=P_FIT, cov=COV, rms=1.23456, success=True, anchors=(0.1, 0.9)):
    return FlightFit(params, cov, rms, 17, np.arange(3) / 30, np.zeros((3, 3)), success,
                     anchor_times=np.array(anchors))


def _flight(fit="default", **kw):
    fit = _fit() if isinstance(fit, str) else fit
    return Flight(start_frame=3, end_frame=20, hitter="near", receiver=Half.FAR, fit=fit, **kw)


BOUNCE_2D = (15.2345, np.array([1.23456, -2.34567]), np.array([[0.01, 0.0], [0.0, 0.0004]]))


def test_f_rounds_and_turns_none_nan_and_infinity_into_none() -> None:
    assert pipeline._f(1.23456) == 1.235
    assert pipeline._f(1.23456, 1) == 1.2
    assert pipeline._f(1.23456, 2) == 1.23
    assert pipeline._f(0) == 0.0 and pipeline._f(3, 1) == 3.0
    assert pipeline._f(-0.0004) == -0.0
    for bad in (None, float("nan"), float("inf"), -float("inf"), np.nan):
        assert pipeline._f(bad) is None


def test_f_returns_a_plain_python_float_even_for_numpy_input() -> None:
    for x in (np.float32(1.23456), np.float64(1.5), np.int64(2)):
        out = pipeline._f(x)
        assert type(out) is float
    assert pipeline._f(np.float32(1.23456)) == 1.235
    json.dumps(pipeline._f(np.float64(1.5)))


def test_f_rounds_halves_to_even_like_python_round() -> None:
    # pinned (pipeline.py:268): round() is banker's rounding, so 2.5 -> 2.0 and 0.5 -> 0.0 at nd=0 (no caller
    # uses nd=0 today).
    assert pipeline._f(2.5, 0) == 2.0 and pipeline._f(3.5, 0) == 4.0 and pipeline._f(0.5, 0) == 0.0


def test_flight_json_keeps_the_ids_frames_names_and_rounds_the_speeds_to_one_decimal() -> None:
    fl = _flight(kind="serve", bounce_frame=15, hit_speed_kmh=123.456, bounce_speed_kmh=float("nan"),
                 net_clearance_m=0.123456, net_clearance_sd_m=None, hit_height_m=2.34567, contact_time=0.1,
                 play_continued=True, notes=["a", "b"])
    d = pipeline._flight_json(7, fl, 30.0)
    assert d["id"] == 7 and d["start_frame"] == 3 and d["end_frame"] == 20
    assert d["hitter"] == "near" and d["receiver"] == "far" and type(d["receiver"]) is str
    assert d["kind"] == "serve" and d["bounce_frame"] == 15
    assert d["hit_speed_kmh"] == 123.5 and d["bounce_speed_kmh"] is None
    assert d["net_clearance_m"] == 0.123 and d["net_clearance_sd_m"] is None and d["hit_height_m"] == 2.346
    assert d["play_continued"] is True and d["rejected"] is None and d["notes"] == ["a", "b"]
    assert d["call"] is None
    json.dumps(d)  # the whole report goes to JSON


def test_flight_json_without_a_fit_or_a_bounce_has_every_key_but_empty_fields() -> None:
    d = pipeline._flight_json(0, Flight(1, 2, "far", Half.NEAR), 25.0)
    assert d["receiver"] == "near" and d["kind"] == "rally"
    assert d["bounce"] is None and d["trajectory"] is None and d["fit"] is None and d["call"] is None
    assert d["contact_frame"] is None and d["hit_speed_kmh"] is None and d["notes"] == []
    assert list(d) == ["id", "start_frame", "end_frame", "hitter", "receiver", "kind", "bounce_frame",
                       "hit_speed_kmh", "bounce_speed_kmh", "net_clearance_m", "net_clearance_sd_m", "hit_height_m",
                       "contact_frame", "play_continued", "rejected", "notes", "bounce", "call", "trajectory", "fit"]


def test_flight_json_contact_frame_is_the_contact_time_in_frames_and_zero_is_not_missing() -> None:
    assert pipeline._flight_json(0, _flight(contact_time=0.1234), 30.0)["contact_frame"] == 3.7
    assert pipeline._flight_json(0, _flight(contact_time=0.1234), 25.0)["contact_frame"] == 3.08
    assert pipeline._flight_json(0, _flight(contact_time=0.0), 30.0)["contact_frame"] == 0.0


def test_flight_json_3d_bounce_block_in_seconds_frames_metres_and_centimetres() -> None:
    d = pipeline._flight_json(0, _flight(contact_time=0.1), 30.0)
    assert d["bounce"] == {"t": 0.5, "frame": 15.0, "x": 1.0, "y": 5.0, "cov": [[0.04, 0.0], [0.0, 0.09]],
                           "sd_x_cm": 20.0, "sd_y_cm": 30.0, "method": "3d"}
    assert pipeline._flight_json(0, _flight(contact_time=0.1), 25.0)["bounce"]["frame"] == 12.5
    assert all(type(v) is float for row in d["bounce"]["cov"] for v in row)


def test_flight_json_fit_block_rounds_and_keeps_a_negative_zero_spin() -> None:
    d = pipeline._flight_json(0, _flight(), 30.0)
    assert d["fit"] == {"rms_px": 1.23, "n_obs": 17, "spin_in": 0.123, "spin_out": -0.0}


def test_flight_json_fit_block_needs_a_successful_fit_with_a_finite_rms() -> None:
    assert pipeline._flight_json(0, _flight(_fit(success=False)), 30.0)["fit"] is None
    assert pipeline._flight_json(0, _flight(_fit(rms=float("nan"))), 30.0)["fit"] is None
    # a nan rms drops the fit block only; the bounce still comes from the same fit
    assert pipeline._flight_json(0, _flight(_fit(rms=float("nan"))), 30.0)["bounce"]["method"] == "3d"
    # a failed fit has no bounce either, even with finite numbers
    failed = pipeline._flight_json(0, _flight(_fit(success=False)), 30.0)
    assert failed["bounce"] is None and failed["trajectory"] is None


def test_flight_json_rejected_fit_keeps_its_fit_block_but_makes_no_3d_bounce_or_trajectory() -> None:
    d = pipeline._flight_json(0, _flight(rejected="speed"), 30.0)
    assert d["rejected"] == "speed"
    assert d["fit"]["rms_px"] == 1.23
    assert d["bounce"] is None and d["trajectory"] is None


def test_flight_json_uses_the_image_kink_when_the_fit_is_rejected_or_has_no_finite_covariance() -> None:
    d = pipeline._flight_json(0, _flight(rejected="speed", bounce_2d=BOUNCE_2D), 30.0)
    assert d["bounce"] == {"t": 0.5078, "frame": 15.23, "x": 1.2346, "y": -2.3457,
                           "cov": [[0.01, 0.0], [0.0, 0.0004]], "method": "2d", "sd_x_cm": 10.0, "sd_y_cm": 2.0}
    assert d["trajectory"] is None
    assert pipeline._flight_json(0, Flight(1, 2, "far", Half.NEAR, bounce_2d=BOUNCE_2D), 25.0)["bounce"]["t"] == 0.6094
    cov = COV.copy()
    cov[0, 0] = np.nan
    d = pipeline._flight_json(0, _flight(_fit(cov=cov), bounce_2d=BOUNCE_2D), 30.0)
    assert d["bounce"]["method"] == "2d" and d["fit"]["rms_px"] == 1.23 and d["trajectory"] is None
    assert pipeline._flight_json(0, _flight(_fit(cov=cov)), 30.0)["bounce"] is None  # no kink either: no bounce


def test_flight_json_a_good_fit_wins_over_the_image_kink() -> None:
    d = pipeline._flight_json(0, _flight(bounce_2d=BOUNCE_2D), 30.0)
    assert d["bounce"]["method"] == "3d" and d["bounce"]["x"] == 1.0


def test_flight_json_only_the_bounce_block_of_the_covariance_has_to_be_finite() -> None:
    cov = COV.copy()
    cov[5, 5] = np.nan  # a velocity variance: not used by the report
    assert pipeline._flight_json(0, _flight(_fit(cov=cov)), 30.0)["bounce"]["method"] == "3d"


def test_flight_json_trajectory_has_60_samples_from_the_contact_to_the_last_anchor() -> None:
    d = pipeline._flight_json(0, _flight(contact_time=0.1), 30.0)
    tr = d["trajectory"]
    assert len(tr) == 60 and all(len(r) == 4 for r in tr)
    assert tr[0] == [0.1, 0.572, 0.722, 2.482]
    assert tr[-1] == [0.9, 1.384, 8.459, 1.575]
    assert tr[1][0] == pytest.approx(0.1 + 0.8 / 59, abs=1e-4)
    # the samples are the fitted flight's own states
    pos, _ = flight_states(P_FIT, np.array([0.1, 0.9]))
    assert tr[0][1:] == pytest.approx(pos[0], abs=1e-3) and tr[-1][1:] == pytest.approx(pos[1], abs=1e-3)


def test_flight_json_trajectory_ends_at_the_end_frame_when_no_anchor_follows_the_bounce() -> None:
    for anchors in ((), (0.3,), (0.5,)):  # none, one before the bounce, one exactly at the bounce
        d = pipeline._flight_json(0, _flight(_fit(anchors=anchors)), 30.0)
        assert d["trajectory"][0][0] == 0.1  # no contact time: starts at start_frame / fps
        assert d["trajectory"][-1][0] == 0.6667  # end_frame / fps
    d = pipeline._flight_json(0, _flight(_fit(anchors=(0.3,)), contact_time=0.0), 30.0)
    assert d["trajectory"][0] == [0.0, 0.457, -0.432, 2.846]  # contact time 0 is a time, not missing


def test_flight_json_call_is_the_line_calls_dict() -> None:
    lc = LineCall("IN", 0.91234, 0.1, 0.02, "baseline", "rally", 1.0, 2.0, [[0.1, 0.0], [0.0, 0.1]])
    d = pipeline._flight_json(0, _flight(call=lc), 30.0)
    assert d["call"] == {"verdict": "IN", "p_in": 0.91234, "margin_m": 0.1, "sigma_m": 0.02, "line": "baseline",
                         "zone": "rally", "x": 1.0, "y": 2.0, "cov": [[0.1, 0.0], [0.0, 0.1]], "conflict": None}


# ------------------------------------------------------------------------------------------------
# pipeline._score_rows


def _read(t, p0, p1, g0=(1,), g1=(2,), s0=True, s1=None):
    return ScoreReading(int(t * 30), t, (ScoreRow("A", g0, p0, s0), ScoreRow("B", g1, p1, s1)))


def _scan(*reads):
    return BroadcastScan(15, 10, scores=list(reads))


def test_score_rows_without_a_scan_or_any_reading_in_range_is_none() -> None:
    assert pipeline._score_rows(None, 5.0) is None
    assert pipeline._score_rows(BroadcastScan(15, 0), 5.0) is None
    scan = _scan(_read(1.0, "15", "0"))
    assert pipeline._score_rows(scan, 0.5) is None  # nothing read yet at that time


def test_score_rows_are_name_games_points_and_server_of_the_last_reading_at_or_before_t() -> None:
    scan = _scan(_read(1.0, "15", "0"), _read(5.0, "30", "15", s0=False, s1=True))
    assert pipeline._score_rows(scan, 1.0) == [
        {"name": "A", "games": [1], "points": "15", "serving": True},
        {"name": "B", "games": [2], "points": "0", "serving": None}]
    rows = pipeline._score_rows(scan, 5.0)  # a reading at t itself counts
    assert rows == [{"name": "A", "games": [1], "points": "30", "serving": False},
                    {"name": "B", "games": [2], "points": "15", "serving": True}]
    assert pipeline._score_rows(scan, 4.99)[0]["points"] == "15"
    assert isinstance(rows[0]["games"], list)  # a list, not the row's tuple (JSON)


def test_score_rows_go_stale_after_max_age_inclusive() -> None:
    scan = _scan(_read(1.0, "15", "0"))
    assert pipeline._score_rows(scan, 21.0) is not None  # exactly 20 s old
    assert pipeline._score_rows(scan, 21.5) is None
    assert pipeline._score_rows(scan, 3.0, max_age=2.0) is not None
    assert pipeline._score_rows(scan, 3.5, max_age=2.0) is None
    assert pipeline._score_rows(scan, 1.0, max_age=0.0) is not None


def test_score_rows_a_stale_latest_reading_does_not_fall_back_to_an_older_one() -> None:
    scan = _scan(_read(1.0, "15", "0"), _read(10.0, "30", "0"))
    assert pipeline._score_rows(scan, 30.0)[0]["points"] == "30"  # 20 s after the second
    assert pipeline._score_rows(scan, 31.0) is None


def test_score_rows_fill_a_missing_zero_only_when_the_other_side_shows_points() -> None:
    scan = _scan(_read(1.0, None, "40"), _read(3.0, "15", None), _read(5.0, None, None))
    assert [r["points"] for r in pipeline._score_rows(scan, 2.0)] == ["0", "40"]
    assert [r["points"] for r in pipeline._score_rows(scan, 4.0)] == ["15", "0"]
    assert [r["points"] for r in pipeline._score_rows(scan, 6.0)] == [None, None]  # nothing shown: nothing invented


def test_score_rows_take_the_last_reading_in_list_order() -> None:
    # pinned (pipeline.py:681): `seen[-1]` is the last in list order, not the latest in time; scan_video appends in
    # time order so it holds today, but an unsorted list gives the older reading.
    scan = _scan(_read(3.0, "30", "0"), _read(1.0, "15", "0"))
    assert pipeline._score_rows(scan, 4.0)[0]["points"] == "15"


# ------------------------------------------------------------------------------------------------
# pipeline._annotate_from_broadcast


def _fl(i, kind, hitter, start):
    return {"id": i, "kind": kind, "hitter": hitter, "receiver": "far" if hitter == "near" else "near",
            "start_frame": start}


def _bc(pairs=(), serves=(), start_end=None, changes=()):
    return {"serve_speed": {"pairs": [{"flight_id": i, "board_kmh": v} for i, v in pairs]},
            "server_sides": {"serves": [{"flight_id": i, "server": s} for i, s in serves],
                             "start_end": start_end or {}},
            "score_changes": [{"t": [a, b]} for a, b in changes]}


def _games(t, g0, g1):
    return ScoreReading(int(t * 30), t, (ScoreRow("ALCARAZ", (g0,), "0"), ScoreRow("SINNER", (g1,), "0")))


def _graphic(g0, g1, names=("ALCARAZ", "SINNER")):
    """A scan whose graphic shows the same games at 1 s and 10 s (flights up to 18 s are within its 8 s window)."""
    return BroadcastScan(15, 10, scores=[_games(1, g0, g1), _games(10, g0, g1)], names=names)


NAMES = ("ALCARAZ", "SINNER")
ENDS = {"ALCARAZ": "near", "SINNER": "far"}
FLIGHTS = [_fl(0, "serve", "near", 30), _fl(1, "rally", "far", 60), _fl(2, "rally", "near", 90),
           _fl(3, "serve", "far", 300), _fl(4, "rally", "near", 330)]


def _run(bc, scan, fps=30.0, cuts=(), flights=FLIGHTS):
    f = copy.deepcopy(flights)
    pipeline._annotate_from_broadcast(f, bc, scan, fps, list(cuts))
    return f


def _names(f):
    return [(x["hitter_name"], x["receiver_name"]) for x in f]


def test_annotate_from_broadcast_without_a_scan_leaves_the_flights_untouched() -> None:
    f = copy.deepcopy(FLIGHTS)
    assert pipeline._annotate_from_broadcast(f, _bc(pairs=[(0, 200.0)]), None, 30.0, []) is None
    assert f == FLIGHTS and "board_speed_kmh" not in f[0]


def test_annotate_from_broadcast_pairs_board_speeds_and_server_names_by_flight_id() -> None:
    scan = _graphic(0, 0)
    f = _run(_bc(pairs=[(0, 201.5), (3, 150.0)], serves=[(0, "ALCARAZ"), (3, "SINNER")], start_end=ENDS), scan)
    assert [x["board_speed_kmh"] for x in f] == [201.5, None, None, 150.0, None]
    assert [x["server_name"] for x in f] == ["ALCARAZ", None, None, "SINNER", None]
    assert [x["id"] for x in f] == [0, 1, 2, 3, 4]  # nothing else is renamed


def test_annotate_from_broadcast_a_marked_server_names_both_ends_for_the_whole_point() -> None:
    scan = _graphic(0, 0)
    f = _run(_bc(serves=[(0, "ALCARAZ"), (3, "SINNER")], start_end=ENDS), scan)
    assert _names(f) == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER"),
                         ("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER")]


def test_annotate_from_broadcast_the_serve_anchor_beats_the_graphics_end_changes() -> None:
    # the graphic's start ends say the opposite of the marked server: the serve marker is what names flights 0-2
    scan = _graphic(0, 0)
    swapped = {"ALCARAZ": "far", "SINNER": "near"}
    f = _run(_bc(serves=[(0, "ALCARAZ")], start_end=swapped), scan)
    assert _names(f)[:3] == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER")]
    # a flight the anchor does not reach (a new serve, no marked server) falls back to the graphic: near = SINNER
    assert _names(f)[3:] == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ")]


def test_annotate_from_broadcast_a_score_change_ends_the_anchor_at_the_rounded_frame() -> None:
    scan = _graphic(0, 0)
    swapped = {"ALCARAZ": "far", "SINNER": "near"}
    # the new score is first seen at 3.0 s = frame 90: flight 2 starts there, the point is over for it
    f = _run(_bc(serves=[(0, "ALCARAZ")], start_end=swapped, changes=[(2.0, 3.0)]), scan)
    assert _names(f)[:3] == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ"), ("SINNER", "ALCARAZ")]
    # 3.02 s * 30 = 90.6 -> frame 91: flight 2 (frame 90) is still inside the point (rounded, not truncated)
    f = _run(_bc(serves=[(0, "ALCARAZ")], start_end=swapped, changes=[(2.0, 3.02)]), scan)
    assert _names(f)[2] == ("ALCARAZ", "SINNER")
    # the end of the change (index 1) counts, not its start: 2.0 s = frame 60 would cut flight 1
    f = _run(_bc(serves=[(0, "ALCARAZ")], start_end=swapped, changes=[(2.0, 10.0)]), scan)
    assert _names(f)[1] == ("SINNER", "ALCARAZ") and _names(f)[2] == ("ALCARAZ", "SINNER")


def test_annotate_from_broadcast_a_camera_cut_ends_the_anchor_and_without_start_ends_nobody_is_named() -> None:
    scan = _graphic(0, 0)
    f = _run(_bc(serves=[(0, "ALCARAZ")]), scan, cuts=[75])
    assert _names(f) == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ"), (None, None), (None, None), (None, None)]


def test_annotate_from_broadcast_without_a_marked_server_the_start_ends_and_game_count_name_the_players() -> None:
    f = _run(_bc(start_end=ENDS), _graphic(0, 0))
    assert [x["server_name"] for x in f] == [None] * 5
    assert _names(f) == [("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER"),
                         ("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER")]
    # one game played: the players have changed ends once, so the near end now belongs to the other player
    f = _run(_bc(start_end=ENDS), _graphic(1, 0))
    assert _names(f)[:3] == [("SINNER", "ALCARAZ"), ("ALCARAZ", "SINNER"), ("SINNER", "ALCARAZ")]


def test_annotate_from_broadcast_graphic_lookup_uses_the_start_frame_over_fps_and_an_8_s_window() -> None:
    scan = BroadcastScan(15, 10, scores=[_games(1, 0, 0)], names=NAMES)  # one reading, at 1 s
    fl = [_fl(0, "rally", "near", 270)]  # 9.0 s at 30 fps: 8 s after the reading
    assert _names(_run(_bc(start_end=ENDS), scan, fps=30.0, flights=fl)) == [("ALCARAZ", "SINNER")]
    assert _names(_run(_bc(start_end=ENDS), scan, fps=15.0, flights=fl)) == [(None, None)]  # 18 s
    assert _names(_run(_bc(start_end=ENDS), scan, fps=300.0, flights=fl)) == [("ALCARAZ", "SINNER")]  # 0.9 s


def test_annotate_from_broadcast_serves_are_not_anchored_when_the_scan_has_no_names() -> None:
    scan = _graphic(0, 0, names=None)
    f = _run(_bc(serves=[(0, "ALCARAZ")], start_end=ENDS), scan)
    assert f[0]["server_name"] == "ALCARAZ"  # the server column still comes from the check
    assert _names(f)[0] == ("ALCARAZ", "SINNER")  # named through the graphic instead


# ------------------------------------------------------------------------------------------------
# pipeline._annotate_strokes (pose network and classifier replaced by recording stubs)

N_FRAMES = 300


class _Rig:
    """Players, kinematics, ball track and stubs for the pose network and the stroke classifier."""

    def __init__(self, tmp_path, poses: dict[int, str | None], ball_finite: set[int] | None = None,
                 y_nan: set[int] = frozenset()):
        f = np.arange(N_FRAMES, dtype=float)
        off = {"near": 0.0, "far": 1000.0}
        self.players = {s: types.SimpleNamespace(boxes=np.column_stack([f, f + 1, f + 2, f + 3]) + off[s])
                        for s in ("near", "far")}
        y = f * 0.01
        y[list(y_nan)] = np.nan
        self.kin = {s: types.SimpleNamespace(position=np.column_stack([f, y])) for s in ("near", "far")}
        xy = np.full((N_FRAMES, 2), np.nan)
        for j in (range(N_FRAMES) if ball_finite is None else ball_finite):
            xy[j] = (j + 0.5, j + 0.25)
        self.ball = types.SimpleNamespace(xy=xy)
        self.poses = poses
        self.out = tmp_path
        (tmp_path / "pose.pt").write_bytes(b"not weights, just bytes to hash")
        self.cfg = pipeline.PipelineConfig(models={"pose": str(tmp_path / "pose.pt")})
        self.device = types.SimpleNamespace(type="cpu")
        self.pose_calls: list = []
        self.classify_calls: list = []

    def install(self, monkeypatch) -> None:
        def poses_at(video, reqs, est):
            self.pose_calls.append((video, [(fr, np.array(b)) for fr, b in reqs], est))
            return [self.poses.get(fr) for fr, _ in reqs]

        def classify(is_serve, pose, ball_px, box, side, hand, bounced, court_y):
            self.classify_calls.append((is_serve, pose, ball_px, box, side, hand, bounced, court_y))
            return Stroke(f"k{len(self.classify_calls)}", "forehand", len(self.classify_calls) % 2 == 0, f"cue{side}")

        monkeypatch.setattr(pipeline, "poses_at", poses_at)
        monkeypatch.setattr(pipeline, "pose_estimator", lambda w, d: f"net:{d}")
        monkeypatch.setattr(pipeline, "serve_hand", lambda pose: pose if pose in ("right", "left") else None)
        monkeypatch.setattr(pipeline, "classify", classify)

    def run(self, fjson, flights, video="clip.mp4", sha="abc"):
        return pipeline._annotate_strokes(video, fjson, flights, self.ball, self.players, self.kin, self.out, sha,
                                          self.cfg, self.device)


def _sf(start, hitter, kind="rally", contact=None, bounce=None):
    other = Half.FAR if hitter == "near" else Half.NEAR
    return Flight(start, start + 40, hitter, other, kind=kind, contact_time=contact, bounce_frame=bounce)


def _fj(*names):
    return [{"id": i, "hitter_name": n} for i, n in enumerate(names)]


def _with_hitters(fjson, flights):
    for f, fl in zip(fjson, flights, strict=True):
        f["hitter"] = fl.hitter
    return fjson


# flights: 0 serve near | 1 rally far | 2 rally near, no contact | 3 serve far | 4 rally far (same hitter as 3)
#          5 rally near | 6 rally far
STROKE_FLIGHTS = [_sf(10, "near", "serve", bounce=30), _sf(50, "far", contact=1.6), _sf(90, "near"),
                  _sf(130, "far", "serve", bounce=150), _sf(170, "far", contact=5.7),
                  _sf(210, "near", contact=7.0, bounce=230), _sf(250, "far", contact=8.3)]
STROKE_POSES = {10: "right", 50: "p50", 130: "left", 170: None, 210: "p210", 250: "p250"}


def _strokes_run(tmp_path, monkeypatch, **kw):
    rig = _Rig(tmp_path, STROKE_POSES, **kw)
    rig.install(monkeypatch)
    fjson = _with_hitters(_fj("ALCARAZ", "SINNER", "ALCARAZ", None, None, "ALCARAZ", "SINNER"), STROKE_FLIGHTS)
    info = rig.run(fjson, STROKE_FLIGHTS)
    return rig, fjson, info


def test_strokes_request_a_pose_only_for_flights_with_a_contact_or_a_serve_at_the_hitters_box(
        tmp_path, monkeypatch) -> None:
    rig, fjson, _ = _strokes_run(tmp_path, monkeypatch)
    video, reqs, est = rig.pose_calls[0]
    assert len(rig.pose_calls) == 1 and video == "clip.mp4" and est == "net:cpu"
    assert [fr for fr, _ in reqs] == [10, 50, 130, 170, 210, 250]  # flight 2 has no contact and is no serve
    for fr, box in reqs:
        side = next(fl.hitter for fl in STROKE_FLIGHTS if fl.start_frame == fr)
        base = np.array([fr, fr + 1, fr + 2, fr + 3.0]) + (1000.0 if side == "far" else 0.0)
        assert box == pytest.approx(base)


def test_strokes_unrequested_flights_are_marked_not_visible_in_the_clip(tmp_path, monkeypatch) -> None:
    _, fjson, _ = _strokes_run(tmp_path, monkeypatch)
    assert fjson[2] == {"id": 2, "hitter_name": "ALCARAZ", "hitter": "near", "stroke": None, "stroke_wing": None,
                        "stroke_confident": False, "stroke_cue": "vuruş anı klipte görülmüyor", "hand": None,
                        "hand_source": None}


def test_strokes_summary_counts_poses_requests_and_the_hands_voted_from_serves(tmp_path, monkeypatch) -> None:
    _, _, info = _strokes_run(tmp_path, monkeypatch)
    # six requests, one pose missing (flight 4); serve votes: ALCARAZ right, the unnamed far side left
    assert info == {"model": "yolo26s-pose", "poses": 5, "requested": 6,
                    "hands": {"ALCARAZ": "right", "far": "left"}}


def test_strokes_hand_comes_from_the_players_serve_votes_else_defaults_to_right(tmp_path, monkeypatch) -> None:
    _, fjson, _ = _strokes_run(tmp_path, monkeypatch)
    assert [(f["hand"], f["hand_source"]) for f in (fjson[i] for i in (0, 1, 3, 4, 5, 6))] == [
        ("right", "servis"), ("right", "varsayılan"), ("left", "servis"), ("left", "servis"),
        ("right", "servis"), ("right", "varsayılan")]


def test_strokes_the_classifier_gets_serve_flag_pose_box_side_hand_bounce_and_court_depth(
        tmp_path, monkeypatch) -> None:
    rig, fjson, _ = _strokes_run(tmp_path, monkeypatch)
    calls = rig.classify_calls
    assert [c[0] for c in calls] == [True, False, True, False, False, False]
    assert [c[4] for c in calls] == ["near", "far", "far", "far", "near", "far"]
    assert [c[5] for c in calls] == ["right", "right", "left", "left", "right", "right"]
    assert [c[1] for c in calls] == ["right", "p50", "left", None, "p210", "p250"]
    # the box at the start frame, of that hitter
    assert calls[1][3] == pytest.approx([1050, 1051, 1052, 1053])
    assert calls[4][3] == pytest.approx([210, 211, 212, 213])
    # did the previous flight (it ends where this one starts, other hitter) bounce: serve/same hitter -> None
    assert [c[6] for c in calls] == [None, True, False, None, False, True]
    # court depth is the kinematics position[1] at the start frame, as a float
    assert calls[1][7] == pytest.approx(0.5) and type(calls[1][7]) is float
    assert calls[4][7] == pytest.approx(2.1)


def test_strokes_the_stroke_fields_are_copied_from_the_classifiers_answer(tmp_path, monkeypatch) -> None:
    _, fjson, _ = _strokes_run(tmp_path, monkeypatch)
    assert (fjson[0]["stroke"], fjson[0]["stroke_wing"], fjson[0]["stroke_confident"], fjson[0]["stroke_cue"]) == (
        "k1", "forehand", False, "cuenear")
    assert (fjson[1]["stroke"], fjson[1]["stroke_confident"], fjson[1]["stroke_cue"]) == ("k2", True, "cuefar")
    assert fjson[4]["stroke"] == "k4"  # a flight whose pose is missing is still classified


def test_strokes_a_missing_court_depth_and_a_ball_that_is_never_found_are_passed_as_none(
        tmp_path, monkeypatch) -> None:
    rig, _, _ = _strokes_run(tmp_path, monkeypatch, ball_finite=set(range(0, 245)), y_nan={250})
    calls = rig.classify_calls
    assert calls[5][2] is None and calls[5][7] is None  # flight 6 at frame 250: no ball in 248..251, depth nan
    assert calls[4][2] == pytest.approx([210.5, 210.25]) and calls[4][7] == pytest.approx(2.1)


def test_strokes_the_ball_position_is_looked_up_at_s_then_s_minus_1_then_s_plus_1_then_s_minus_2(
        tmp_path, monkeypatch) -> None:
    flights = [_sf(100, "near", contact=3.0), _sf(150, "near", contact=5.0), _sf(200, "near", contact=6.5),
               _sf(250, "near", contact=8.0)]
    fjson = _with_hitters(_fj(None, None, None, None), flights)
    rig = _Rig(tmp_path, {100: "p", 150: "p", 200: "p", 250: "p"},
               ball_finite={100, 99, 101, 98,   # 100: all four finite -> s
                            149, 151, 148,  # 150: s missing -> s - 1
                            201, 198,  # 200: s, s-1 missing -> s + 1
                            248})  # 250: only s - 2
    rig.install(monkeypatch)
    rig.run(fjson, flights)
    assert [tuple(c[2]) for c in rig.classify_calls] == [(100.5, 100.25), (149.5, 149.25), (201.5, 201.25),
                                                         (248.5, 248.25)]


def test_strokes_the_ball_lookup_never_wraps_to_the_end_of_the_track_or_past_it(tmp_path, monkeypatch) -> None:
    flights = [_sf(0, "near", contact=0.0), _sf(N_FRAMES - 1, "far", contact=9.9)]
    fjson = _with_hitters(_fj(None, None), flights)
    rig = _Rig(tmp_path, {0: "p", N_FRAMES - 1: "p"}, ball_finite={1, N_FRAMES - 2, N_FRAMES - 3})
    rig.install(monkeypatch)
    rig.ball.xy[-1] = (7.0, 7.0)  # a finite last frame that frame 0's lookup must not reach as index -1
    rig.run(fjson, flights)
    assert tuple(rig.classify_calls[0][2]) == (1.5, 1.25)  # frame 0: s-1 is skipped, not index -1
    assert tuple(rig.classify_calls[1][2]) == (7.0, 7.0)  # the last frame itself is finite


def test_strokes_unnamed_players_vote_per_side_and_a_tie_gives_no_hand(tmp_path, monkeypatch) -> None:
    flights = [_sf(10, "near", "serve"), _sf(50, "far", "serve"), _sf(90, "near", "serve"), _sf(130, "near", "serve")]
    fjson = _with_hitters(_fj(None, None, None, None), flights)
    rig = _Rig(tmp_path, {10: "right", 50: "left", 90: "left", 130: "right"})
    rig.install(monkeypatch)
    info = rig.run(fjson, flights)
    # near: right, left, right -> right; far: left -> left (the two unnamed sides do not share one vote)
    assert info["hands"] == {"near": "right", "far": "left"}


def test_strokes_a_tied_vote_gives_no_hand_and_the_default_is_marked_as_such(tmp_path, monkeypatch) -> None:
    flights = [_sf(10, "near", "serve"), _sf(90, "near", "serve"), _sf(130, "near", contact=4.0)]
    fjson = _with_hitters(_fj("X", "X", "X"), flights)
    rig = _Rig(tmp_path, {10: "right", 90: "left", 130: "p"})
    rig.install(monkeypatch)
    info = rig.run(fjson, flights)
    assert info["hands"] == {}
    assert (fjson[2]["hand"], fjson[2]["hand_source"]) == ("right", "varsayılan")
    assert rig.classify_calls[2][5] == "right"


def test_strokes_poses_are_cached_by_the_requests_and_reused_only_when_asked(tmp_path, monkeypatch) -> None:
    rig, _, info = _strokes_run(tmp_path, monkeypatch)
    assert (tmp_path / "pose.pkl").is_file() and len(rig.pose_calls) == 1
    rig.cfg.reuse_perception = True
    fjson = _with_hitters(_fj("ALCARAZ", "SINNER", "ALCARAZ", None, None, "ALCARAZ", "SINNER"), STROKE_FLIGHTS)
    again = rig.run(fjson, STROKE_FLIGHTS)
    assert len(rig.pose_calls) == 1 and again == info  # same video, model and boxes: the cache is read
    rig.run(copy.deepcopy(fjson), STROKE_FLIGHTS, sha="other video")
    assert len(rig.pose_calls) == 2  # another video: the poses are computed again
    rig.players["near"].boxes[10] += 5.0  # another player selection at a requested frame
    rig.run(copy.deepcopy(fjson), STROKE_FLIGHTS, sha="other video")
    assert len(rig.pose_calls) == 3
    rig.cfg.reuse_perception = False
    rig.run(copy.deepcopy(fjson), STROKE_FLIGHTS, sha="other video")
    assert len(rig.pose_calls) == 4  # not asked for: always computed


# ------------------------------------------------------------------------------------------------
# pipeline._own_arch


class _Foreign:
    pass


def test_own_arch_reads_the_arch_of_a_dict_checkpoint(tmp_path) -> None:
    import torch
    path = tmp_path / "m.pt"
    torch.save({"arch": "sekme-test", "w": torch.zeros(2)}, path)
    assert pipeline._own_arch(str(path)) == "sekme-test"


def test_own_arch_is_none_for_everything_that_is_not_one_of_our_checkpoints(tmp_path) -> None:
    import torch
    torch.save({"w": torch.zeros(2)}, tmp_path / "noarch.pt")
    torch.save([1, 2, 3], tmp_path / "list.pt")
    torch.save(_Foreign(), tmp_path / "foreign.pt")  # a third-party class cannot be unpickled with weights_only
    (tmp_path / "junk.pt").write_bytes(b"this is not a checkpoint")
    for name in ("noarch.pt", "list.pt", "foreign.pt", "junk.pt", "missing.pt"):
        assert pipeline._own_arch(str(tmp_path / name)) is None


# ------------------------------------------------------------------------------------------------
# broadcast/scoreboard.py: _split_tokens, _name_parts, _is_speed_label, _stable, _marker_pixels


def test_split_tokens_gives_each_token_a_box_proportional_to_its_place_in_the_text() -> None:
    out = _split_tokens(TextBox("5 40", 0.9, 0.2, 0.5, 0.1, 0.02))
    assert [b.text for b in out] == ["5", "40"]
    assert out[0].x == pytest.approx(0.2) and out[0].w == pytest.approx(0.025)
    assert out[1].x == pytest.approx(0.25) and out[1].w == pytest.approx(0.05)
    assert all(b.conf == 0.9 and b.y == 0.5 and b.h == 0.02 for b in out)


def test_split_tokens_repeated_tokens_and_wide_gaps_keep_their_own_positions() -> None:
    out = _split_tokens(TextBox("5 5 5", 1.0, 0.0, 0.0, 0.5, 1.0))
    assert [b.x for b in out] == pytest.approx([0.0, 0.2, 0.4]) and [b.w for b in out] == pytest.approx([0.1] * 3)
    out = _split_tokens(TextBox("1  22 3", 1.0, 0.0, 0.0, 0.7, 1.0))
    assert [b.text for b in out] == ["1", "22", "3"]
    assert [b.x for b in out] == pytest.approx([0.0, 0.3, 0.6]) and [b.w for b in out] == pytest.approx([0.1, 0.2, 0.1])


def test_split_tokens_a_single_token_or_blank_text_comes_back_as_the_same_box() -> None:
    one, blank = TextBox("5", 1.0, 0.1, 0.2, 0.3, 0.4), TextBox("   ", 1.0, 0.1, 0.2, 0.3, 0.4)
    assert _split_tokens(one) == [one] and _split_tokens(blank) == [blank]


@pytest.mark.parametrize("text, expected", [
    ("MEDVEDEV", (False, "MEDVEDEV")),
    ("3 MEDVEDEV", (False, "MEDVEDEV")),  # games counter in front
    ("2 DJOKOVIC SRB", (False, "DJOKOVIC")),
    ("  zverev ger ", (False, "ZVEREV GER")),  # a lower-case code is not recognised as one: it stays
    ("Auger-Aliassime CAN", (False, "AUGER-ALIASSIME")),
    ("O'CONNELL AUS", (False, "O'CONNELL")),
    ("DE MINAUR", (False, "DE MINAUR")),  # a two-word surname is not a country code
    ("ZHENG QINWEN CHN", (False, "ZHENG QINWEN")),
    ("JAN LENNARD STRUFF GER", (False, "JAN LENNARD STRUFF")),
    ("LI NA", (False, "LI NA")),
    ("ZHENG Qin", (False, "ZHENG QIN")),  # a three-letter last word that is not upper case is kept
    ("SRB", (False, "SRB")),  # one word: nothing to strip
    ("ABC", (False, "ABC")),
    ("Ñandi", (False, "ÑANDI")),
])
def test_name_parts_cleans_a_name_box(text, expected) -> None:
    assert _name_parts(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("▸DJOKOVIC", (True, "DJOKOVIC")),
    ("DJOKOVIC ●", (True, "DJOKOVIC")),
    ("▶ DJOKOVIC SRB ◀", (True, "DJOKOVIC")),
    ("DJOKOVIC SRB ●", (True, "DJOKOVIC")),
    ("  ▶ ", (True, "")),
    ("▸", (True, "")),
])
def test_name_parts_reports_a_marker_glyph_at_either_end(text, expected) -> None:
    assert _name_parts(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "12", "1 2", "ab1", "AB", "X Y", "a b", "ALCARAZ G.", "J.-W. TSONGA"])
def test_name_parts_rejects_what_is_not_a_name(text) -> None:
    # pinned (scoreboard.py:35, _NAME): a name with a dotted initial ("ALCARAZ G.", "J.-W. TSONGA") is not read as a
    # name, so a graphic that prints initials loses that row; two letters or fewer ("AB", "X Y") are dropped on purpose.
    assert _name_parts(text)[1] == ""


@pytest.mark.parametrize("text", ["SERVE SPEED", "Serve Speed", "serve", "SPEED", "MPH", "km/h", "KMH", "K M H",
                                  "SERVE SPEEO", "SERVESPEED", "SERVE SPEED MPH", "mph:", "SPE", "SPED"])
def test_speed_labels_are_found_case_blind_and_through_ocr_noise(text) -> None:
    assert _is_speed_label(text)


@pytest.mark.parametrize("text", ["", "1 2 3", "AB", "KM", "MP", "PH", "FORMULA", "VESP", "SERVESPXXXX", "Emirates",
                                  "40", "12:30"])
def test_texts_that_are_not_speed_labels(text) -> None:
    assert not _is_speed_label(text)


def test_speed_label_similarity_threshold_is_inclusive_at_0_7_and_needs_three_letters() -> None:
    assert _is_speed_label("SERVESPXXX")  # ratio to SERVESPEED is exactly 0.7
    assert not _is_speed_label("SERVESPXXXX")  # 0.667
    assert not _is_speed_label("KM") and not _is_speed_label("MP")  # 0.8 to KMH / MPH but only two letters
    assert _is_speed_label("12 KMH 34")  # digits and spaces are ignored
    # pinned (scoreboard.py:218): fuzzy matching accepts look-alike words, so a caption "SERVICE" or
    # "SPEEDY" counts as a label (parse_speed then still needs a plausible number beside it).
    assert _is_speed_label("SERVICE") and _is_speed_label("SPEEDY")


class _R:
    def __init__(self, t, k):
        self.t, self.k = t, k


def _runs(seq, gap):
    return [(a, b, c.k, c.t) for a, b, c in _stable(seq, lambda r: r.k, gap)]


def test_stable_keeps_runs_of_two_or_more_equal_readings_with_their_first_and_last_time() -> None:
    seq = [_R(0, 1), _R(0.5, 1), _R(1, 2), _R(1.5, 3), _R(2, 3), _R(2.5, 3), _R(9, 3)]
    assert _runs(seq, 1.0) == [(0, 0.5, 1, 0), (1.5, 2.5, 3, 1.5)]  # the item is the run's first reading
    assert _stable([], lambda r: r.k, 1.0) == []
    assert _runs([_R(0, 1)], 1.0) == []  # one reading is noise


def test_stable_a_gap_longer_than_max_gap_splits_a_run_and_exactly_max_gap_does_not() -> None:
    seq = [_R(0, 3), _R(1, 3), _R(2, 3)]
    assert _runs(seq, 1.0) == [(0, 2, 3, 0)]  # gaps of exactly 1.0 join
    assert _runs(seq, 0.99) == []
    assert _runs([_R(0, 3), _R(0.5, 3), _R(5, 3), _R(5.5, 3), _R(9, 3)], 1.0) == [(0, 0.5, 3, 0), (5, 5.5, 3, 5)]


def test_stable_a_reading_of_another_value_in_between_breaks_the_run() -> None:
    assert _runs([_R(0, 1), _R(0.5, 2), _R(1, 1), _R(1.5, 1)], 1.0) == [(1, 1.5, 1, 1)]
    assert _runs([_R(0, 1), _R(0, 1), _R(0, 1)], 1.0) == [(0, 0, 1, 0)]  # equal times still count as readings


HH, WW = 100, 200
YELLOW = (0, 255, 255)
ROW = TextBox("NAME", 1.0, 0.2, 0.4, 0.2, 0.1)  # rows 40..50; left region columns 24..39, right region 80..gap


def _blank():
    return np.zeros((HH, WW, 3), np.uint8)


def _paint(img, x0, x1, y0, y1, bgr=YELLOW):
    img[y0:y1, x0:x1] = bgr
    return img


def _hsv(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0])


def test_marker_pixels_is_zero_without_an_image() -> None:
    assert _marker_pixels(None, ROW, 0.6, [ROW]) == 0
    assert _marker_pixels(_blank(), ROW, 0.6, [ROW]) == 0


def test_marker_pixels_counts_salient_pixels_right_and_left_of_the_name_and_adds_both_sides() -> None:
    assert _marker_pixels(_paint(_blank(), 100, 108, 42, 48), ROW, 0.6, [ROW]) == 48  # right of the name
    assert _marker_pixels(_paint(_blank(), 28, 34, 42, 48), ROW, 0.6, [ROW]) == 36  # left of the name
    both = _paint(_paint(_blank(), 100, 108, 42, 48), 28, 34, 42, 48)
    assert _marker_pixels(both, ROW, 0.6, [ROW]) == 84


def test_marker_pixels_the_row_band_is_the_boxes_rows_plus_one() -> None:
    def n(y):
        return _marker_pixels(_paint(_blank(), 28, 34, y, y + 1), ROW, 0.6, [ROW])
    assert [n(y) for y in (38, 39, 40, 50, 51, 52)] == [0, 0, 6, 6, 0, 0]


def test_marker_pixels_the_left_region_is_1_6_row_heights_wide_in_x_units_of_the_frame() -> None:
    # row height 0.1 of a 100 px high, 200 px wide frame: 1.6 * 0.1 * 100 / 200 = 0.08 of the width = 16 px
    def n(x):
        return _marker_pixels(_paint(_blank(), x, x + 1, 42, 48), ROW, 0.6, [ROW])
    assert [n(x) for x in (4, 10, 20, 23, 24, 30, 38, 39)] == [0, 0, 0, 0, 6, 6, 6, 0]  # 39: the name's own margin


def test_marker_pixels_the_right_region_ends_at_the_given_gap() -> None:
    def n(x, gap):
        return _marker_pixels(_paint(_blank(), x, x + 1, 42, 48), ROW, gap, [ROW])
    assert [n(x, 0.6) for x in (79, 100, 119, 120, 150)] == [0, 6, 6, 0, 0]  # the name's own box ends the left side
    assert n(150, 0.8) == 6  # a wider gap reaches further


def test_marker_pixels_only_yellow_and_white_count() -> None:
    def n(h, s, v):
        return _marker_pixels(_paint(_blank(), 100, 102, 42, 43, _hsv(h, s, v)), ROW, 0.6, [ROW])
    assert n(30, 200, 200) == 2  # yellow ball logo
    assert n(100, 10, 230) == 2  # white arrow (any hue: it is unsaturated and bright)
    assert n(10, 200, 200) == 0  # orange
    assert n(60, 200, 200) == 0  # green
    assert n(30, 60, 255) == 0  # pale yellow: not saturated enough, too saturated for white
    assert n(30, 200, 100) == 0  # dark yellow
    assert n(100, 10, 180) == 0  # grey, not bright enough for white
    assert n(100, 100, 230) == 0  # coloured and bright


def test_marker_pixels_recognised_text_boxes_are_not_markers_with_a_one_two_pixel_margin() -> None:
    digit = TextBox("40", 1.0, 0.5, 0.4, 0.02, 0.1)  # columns 100..104, rows 40..50
    def lost(c):
        return _marker_pixels(_paint(_blank(), c, c + 1, 42, 48), ROW, 0.6, [ROW, digit]) == 0
    assert [c for c in range(95, 112) if lost(c)] == [99, 100, 101, 102, 103, 104, 105]
    assert _marker_pixels(_paint(_blank(), 95, 112, 42, 48), ROW, 0.6, [ROW]) == 17 * 6  # without the box: all count
    assert _marker_pixels(_paint(_blank(), 95, 112, 42, 48), ROW, 0.6, [ROW, digit]) == 10 * 6
    lower = TextBox("40", 1.0, 0.5, 0.45, 0.02, 0.1)  # rows 45..55: excludes 44..(ours end at 50)
    def kept(r):
        return _marker_pixels(_paint(_blank(), 100, 101, r, r + 1), ROW, 0.6, [ROW, lower]) == 1
    assert [r for r in range(40, 51) if kept(r)] == [40, 41, 42, 43]


def test_marker_pixels_a_text_box_at_the_left_edge_masks_from_column_zero() -> None:
    row = TextBox("NAME", 1.0, 0.05, 0.4, 0.2, 0.1)  # its left region is columns 0..9
    edge = TextBox("1", 1.0, 0.0, 0.4, 0.015, 0.1)
    img = _paint(_blank(), 0, 10, 42, 48)
    assert _marker_pixels(img, row, 0.6, [row]) == 54  # the row's own box masks column 9 and beyond
    assert _marker_pixels(img, row, 0.6, [row, edge]) == 24  # columns 0..4 are masked by the box at x = 0


# ------------------------------------------------------------------------------------------------
# physics/flight.py: integrate, initial_guess, _project, _residuals_batch


def _look_at(center, target, f=1750.0, size=(1920, 1080)) -> Camera:
    """Copy of conftest.look_at_camera (world z up, image y down)."""
    center, target = np.asarray(center, float), np.asarray(target, float)
    fwd = target - center
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[f, 0.0, size[0] / 2], [0.0, f, size[1] / 2], [0.0, 0.0, 1.0]])
    return Camera(K=K, R=R, t=-R @ center)


def _jittered_cameras(n):
    """One slightly different camera per frame, as a per-frame calibration gives."""
    return [_look_at([0.05 * np.sin(i), -28.0 + 0.1 * i, 11.0], [0.0, 1.5, 0.0]) for i in range(n)]


def test_units_gravity_in_metres_per_second_squared_and_drag_in_per_metre() -> None:
    assert G == 9.81
    assert pytest.approx(0.0202, abs=2e-4) == DRAG_K  # 1/m: 0.5 rho Cd A / m for a tennis ball
    assert np.sqrt(G / DRAG_K) == pytest.approx(22.06, abs=0.1)  # terminal speed, m/s (about 79 km/h)


def test_integrate_forward_without_drag_is_the_closed_form_parabola_per_batch_member() -> None:
    p0 = np.array([[0.0, 0.0, 1.0], [2.0, -1.0, 0.5]])
    v0 = np.array([[3.0, 4.0, 10.0], [1.0, 0.0, 0.0]])
    rel = np.array([[0.1, 0.5, 1.0], [0.2, 0.3, 0.4]])  # each row has its own times
    pos, vel = integrate(p0, v0, rel, k=0.0)
    assert pos.shape == (2, 3, 3) and vel.shape == (2, 3, 3)
    for b in range(2):
        t = rel[b][:, None]
        assert pos[b] == pytest.approx(p0[b] + v0[b] * t - 0.5 * G * t**2 * np.array([0, 0, 1.0]), abs=1e-9)
        assert vel[b] == pytest.approx(v0[b] - G * t * np.array([0, 0, 1.0]), abs=1e-9)


def test_integrate_backward_without_drag_runs_the_parabola_in_negative_time() -> None:
    p0 = np.array([[0.0, 0.0, 1.0]])
    v0 = np.array([[3.0, 4.0, 10.0]])
    rel = np.array([[-0.1, -0.5]])  # ordered by increasing |t|
    pos, vel = integrate(p0, v0, rel, k=0.0)
    t = rel[0][:, None]
    assert pos[0] == pytest.approx(p0[0] + v0[0] * t - 0.5 * G * t**2 * np.array([0, 0, 1.0]), abs=1e-9)
    assert pos[0, 1, 2] == pytest.approx(1.0 - 5.0 - 0.5 * G * 0.25, abs=1e-9)
    assert vel[0, 1] == pytest.approx([3.0, 4.0, 10.0 + G * 0.5], abs=1e-9)  # faster upward earlier in time


def test_integrate_each_target_continues_from_the_previous_one_and_a_zero_target_is_the_start() -> None:
    p0, v0 = np.array([[0.0, 0.0, 1.0]]), np.array([[3.0, 4.0, 10.0]])
    pos, vel = integrate(p0, v0, np.array([[0.0, 0.25, 0.25, 1.0]]), k=0.0)
    assert pos[0, 0] == pytest.approx(p0[0]) and vel[0, 0] == pytest.approx(v0[0])
    assert pos[0, 1] == pytest.approx(pos[0, 2], abs=1e-12)  # a repeated target stays put
    one, _ = integrate(p0, v0, np.array([[1.0]]), k=0.0)
    assert pos[0, 3] == pytest.approx(one[0, 0], abs=1e-9)  # the way there does not matter without drag


def test_integrate_does_not_modify_its_inputs_and_accepts_integer_arrays() -> None:
    p0, v0 = np.array([[0, 0, 1]]), np.array([[3, 4, 10]])
    pos, vel = integrate(p0, v0, np.array([[0.5]]))
    assert p0.tolist() == [[0, 0, 1]] and v0.tolist() == [[3, 4, 10]]
    assert pos.dtype == float and vel.dtype == float


def test_integrate_a_ball_dropped_with_drag_follows_the_hyperbolic_fall_to_terminal_speed() -> None:
    k = 1.0  # a large k makes drag matter within a few seconds
    vt = np.sqrt(G / k)
    t = np.array([2.0, 5.0])
    pos, vel = integrate(np.array([[0.0, 0.0, 100.0]]), np.zeros((1, 3)), t[None], k=k)
    assert pos[0, :, 2] == pytest.approx(100 - (vt**2 / G) * np.log(np.cosh(G * t / vt)), abs=1e-6)
    assert vel[0, :, 2] == pytest.approx(-vt * np.tanh(G * t / vt), abs=1e-6)
    assert vel[0, 1, 2] == pytest.approx(-vt, abs=1e-6)  # terminal speed, m/s


def test_integrate_a_ball_thrown_up_with_the_real_drag_peaks_lower_and_earlier_than_in_vacuum() -> None:
    vt, v0 = np.sqrt(G / DRAG_K), 30.0
    t_apex = (vt / G) * np.arctan(v0 / vt)
    apex = (vt**2 / (2 * G)) * np.log(1 + v0**2 / vt**2)
    pos, vel = integrate(np.zeros((1, 3)), np.array([[0.0, 0.0, v0]]), np.array([[t_apex]]))
    assert pos[0, 0, 2] == pytest.approx(apex, abs=1e-6) and vel[0, 0, 2] == pytest.approx(0.0, abs=1e-6)
    assert t_apex < v0 / G and apex < v0**2 / (2 * G)  # 2.11 s and 25.97 m against 3.06 s and 45.9 m in vacuum


def test_integrate_drag_slows_the_horizontal_speed_and_the_default_is_the_balls_drag() -> None:
    p0, v0 = np.zeros((1, 3)), np.array([[30.0, 0.0, 0.0]])
    rel = np.array([[0.5, 1.0]])
    _, drag = integrate(p0, v0, rel)
    _, same = integrate(p0, v0, rel, k=DRAG_K)
    _, vacuum = integrate(p0, v0, rel, k=0.0)
    assert np.array_equal(drag, same) and vacuum[0, :, 0] == pytest.approx([30.0, 30.0])
    assert np.all(np.diff(drag[0, :, 0]) < 0) and drag[0, 1, 0] < 30.0 * np.exp(-DRAG_K * 30.0 * 0.5)


def test_integrate_substeps_trade_accuracy_for_time() -> None:
    k, vt, t = 1.0, 1.0 / np.sqrt(1.0 / G), 5.0

    def err(substeps):
        pos, _ = integrate(np.array([[0.0, 0.0, 100.0]]), np.zeros((1, 3)), np.array([[t]]), k=k, substeps=substeps)
        return abs(pos[0, 0, 2] - (100 - (vt**2 / G) * np.log(np.cosh(G * t / vt))))

    assert err(1) > 5e-7 > err(2)  # the default is two sub-steps per 1/30 s
    assert err(8) < 1e-8


def test_integrate_keeps_wild_trials_finite_by_clipping_speed_and_position() -> None:
    pos, vel = integrate(np.array([[9999.0, 0.0, 0.0]]), np.array([[800.0, 0.0, 0.0]]), np.array([[2.0]]), k=0.0)
    assert pos[0, 0, 0] == 1e4 and vel[0, 0, 0] == 500.0
    assert vel[0, 0, 2] == pytest.approx(-2 * G, abs=1e-9)  # what is inside the limits is untouched
    pos, vel = integrate(np.zeros((1, 3)), np.array([[0.0, -900.0, 0.0]]), np.array([[0.5]]), k=0.0)
    assert vel[0, 0, 1] == -500.0


def test_integrate_positive_spin_dips_the_ball_and_negative_spin_lifts_it() -> None:
    p0, v0 = np.zeros((1, 3)), np.array([[10.0, 0.0, 0.0]])
    rel = np.array([[0.2]])

    def z(spin):
        return integrate(p0, v0, rel, k=0.0, spin=np.array([spin]))[0][0, 0, 2]

    assert z(0.0) == pytest.approx(-0.5 * G * 0.04, abs=1e-12)
    assert z(0.5) < z(0.0) < z(-0.5)
    assert z(0.5) == pytest.approx(-0.5 * (G + 5.0) * 0.04, rel=0.02)  # a_z = -g - spin * v_h at the start


def test_initial_guess_on_a_frame_grid_puts_the_bounce_exactly_where_the_ball_touches_down() -> None:
    n = 30
    t = np.arange(n) / 30  # t = 0.5 s is frame 15
    truth = np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.0, 0.0])
    pos, _ = flight_states(truth, t)
    cams = _jittered_cameras(n)
    obs = np.array([c.project(p) for c, p in zip(cams, pos, strict=True)])
    g = initial_guess(t, obs, cams, 0.5)
    assert g.shape == (11,)
    assert g[0] == pytest.approx(1.0, abs=1e-6) and g[1] == pytest.approx(5.0, abs=1e-6)
    assert g[2] == 0.5 and g[9] == 0.0 and g[10] == 0.0
    first = cams[0].backproject_to_height(obs[0], 1.0)  # the horizontal speeds come from 1 m back-projections
    last = cams[-1].backproject_to_height(obs[-1], 1.0)
    pb = np.array([g[0], g[1], BALL_RADIUS])
    assert g[3:5] == pytest.approx(((pb - first) / 0.5)[:2], abs=1e-6)
    assert g[6:8] == pytest.approx(((last - pb) / (29 / 30 - 0.5))[:2], abs=1e-6)
    assert g[5] == pytest.approx(-(1.0 + 0.5 * G * 0.25) / 0.5)  # from dropping 1 m in the time before the bounce
    dt_out = 29 / 30 - 0.5
    assert g[8] == pytest.approx((1.0 + 0.5 * G * dt_out**2) / dt_out)


def test_initial_guess_on_a_clean_simulated_shot_lands_near_the_truth() -> None:
    cam = _look_at([0.0, -28.0, 11.0], [0.0, 1.5, 0.0])
    shot = simulate_shot(cam, np.array([-1.0, -11.5, 1.0]), np.array([2.0, 9.0]), 120.0, pixel_noise=0.0,
                         rng=np.random.default_rng(0), phase=0.0)
    g = initial_guess(shot.times, shot.pixels, [cam] * len(shot.times), shot.bounce_time)
    assert np.linalg.norm(g[:2] - shot.bounce_xy) < 0.15  # measured 0.05 m
    assert g[2] == shot.bounce_time
    true_in = (shot.positions[20] - shot.positions[19]) * 30  # horizontal velocity shortly before the bounce
    true_out = (shot.positions[26] - shot.positions[25]) * 30
    for guess, true in ((g[3:5], true_in[:2]), (g[6:8], true_out[:2])):
        assert float(guess @ true / (np.linalg.norm(guess) * np.linalg.norm(true))) > 0.99  # same direction
        assert 0.5 < np.linalg.norm(guess) / np.linalg.norm(true) < 1.6  # a seed, within a factor of two
    assert g[5] < 0 < g[8] and g[9] == 0.0 and g[10] == 0.0


def test_initial_guess_takes_the_bounce_from_the_observation_nearest_the_guessed_time() -> None:
    n = 30
    t = np.arange(n) / 30
    truth = np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.0, 0.0])
    pos, _ = flight_states(truth, t)
    cams = _jittered_cameras(n)
    obs = np.array([c.project(p) for c, p in zip(cams, pos, strict=True)])
    exact = initial_guess(t, obs, cams, 0.5)
    near = initial_guess(t, obs, cams, 0.5 + 0.004)  # still closest to frame 15
    assert near[:2] == pytest.approx(exact[:2]) and near[2] == 0.504
    off = initial_guess(t, obs, cams, 0.5 + 1 / 30)  # now frame 16's ball is taken as the touch-down
    assert np.linalg.norm(off[:2] - exact[:2]) > 0.3


def test_initial_guess_floors_the_flight_times_at_a_tenth_of_a_second() -> None:
    n = 30
    t = np.arange(n) / 30
    pos, _ = flight_states(np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.0, 0.0]), t)
    cams = _jittered_cameras(n)
    obs = np.array([c.project(p) for c, p in zip(cams, pos, strict=True)])
    early = initial_guess(t, obs, cams, 0.0)  # bounce at the first observation: no time before it
    assert early[5] == pytest.approx(-(1.0 + 0.5 * G * 0.1**2) / 0.1)
    late = initial_guess(t, obs, cams, t[-1])
    assert late[8] == pytest.approx((1.0 + 0.5 * G * 0.1**2) / 0.1)


def test_project_matches_the_cameras_own_projection_for_every_batch_member_and_observation() -> None:
    cams = _jittered_cameras(5)
    rng = np.random.default_rng(3)
    pos = np.column_stack([rng.uniform(-5, 5, 15), rng.uniform(-10, 10, 15), rng.uniform(0, 4, 15)]).reshape(3, 5, 3)
    KR = np.stack([c.K @ c.R for c in cams])
    Kt = np.stack([c.K @ c.t for c in cams])
    out = _project(KR, Kt, pos)
    assert out.shape == (3, 5, 2)
    for b in range(3):
        for i in range(5):
            assert out[b, i] == pytest.approx(cams[i].project(pos[b, i]), abs=1e-8)
    assert _project(KR, Kt, pos[:1]) == pytest.approx(out[:1])


def test_project_a_point_on_the_optical_axis_lands_on_the_principal_point() -> None:
    cam = _look_at([0.0, -28.0, 11.0], [0.0, 1.5, 0.0])
    KR, Kt = (cam.K @ cam.R)[None], (cam.K @ cam.t)[None]
    assert _project(KR, Kt, np.array([[[0.0, 1.5, 0.0]]]))[0, 0] == pytest.approx([960.0, 540.0], abs=1e-6)
    right = _project(KR, Kt, np.array([[[1.0, 1.5, 0.0]]]))[0, 0]
    assert right[0] > 960.0 and right[1] == pytest.approx(540.0, abs=1e-6)  # +x is image right
    assert _project(KR, Kt, np.array([[[0.0, 1.5, 2.0]]]))[0, 0, 1] < 540.0  # up is image up (y falls)


TRUTH = np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.2, -0.1])


def _residual_setup(n=8):
    times = np.arange(n) / 30 + 0.3
    cams = _jittered_cameras(n)
    pos, _ = flight_states(TRUTH, times)
    obs = np.array([c.project(p) for c, p in zip(cams, pos, strict=True)])
    KR = np.stack([c.K @ c.R for c in cams])
    Kt = np.stack([c.K @ c.t for c in cams])
    return times, cams, obs, KR, Kt


def test_residuals_pixel_part_is_projection_minus_observation_interleaved_u_v_then_the_spin_priors() -> None:
    times, _, obs, KR, Kt = _residual_setup()
    shifted = obs + np.array([2.0, -3.0])  # observed 2 px right and 3 px up of the model
    res = _residuals_batch(TRUTH[None], times, shifted, KR, Kt, (), 1.5)
    assert res.shape == (1, 2 * 8 + 2)
    assert res[0, :16] == pytest.approx(np.tile([-2.0, 3.0], 8), abs=1e-6)  # model - observation
    assert res[0, 16:] == pytest.approx([0.2 / SPIN_PRIOR_SIGMA * 1.5, -0.1 / SPIN_PRIOR_SIGMA * 1.5])


def test_residuals_priors_are_scaled_by_the_pixel_sigma_and_the_pixel_part_is_not() -> None:
    times, _, obs, KR, Kt = _residual_setup()
    a = _residuals_batch(TRUTH[None], times, obs + 1.0, KR, Kt, (), 1.5)[0]
    b = _residuals_batch(TRUTH[None], times, obs + 1.0, KR, Kt, (), 3.0)[0]
    assert b[:16] == pytest.approx(a[:16]) and b[16:] == pytest.approx(2 * a[16:])
    assert a[16] != a[17]  # the two phases have their own spin


def test_residuals_anchor_rows_are_whitened_time_x_y_and_height_band_violations_in_that_order() -> None:
    times, _, obs, KR, Kt = _residual_setup()
    anchors = (Anchor(t=0.25, xy=np.array([0.5, 2.0])),  # inside the height band: its last row is zero
               Anchor(t=0.8, xy=np.array([0.0, 0.0]), t_sigma=0.1, sigma_x=2.0, sigma_y=4.0, z_min=0.0, z_max=1.0,
                      z_scale=0.5),  # above the band
               Anchor(t=0.9, xy=np.array([0.0, 0.0]), z_min=50.0, z_max=60.0, z_scale=0.1))  # below the band
    P = np.concatenate([TRUTH, [0.2, 0.85, 0.95]])[None]
    res = _residuals_batch(P, times, obs, KR, Kt, anchors, 2.0)[0]
    assert res.shape == (2 * 8 + 3 * 4 + 2,)
    pos, _ = flight_states(TRUTH, np.array([0.2, 0.85, 0.95]))
    expect = []
    for (px, ta), a in zip(zip(pos, (0.2, 0.85, 0.95), strict=True), anchors, strict=True):
        band = max(a.z_min - px[2], 0.0) + max(px[2] - a.z_max, 0.0)
        expect += [(ta - a.t) / a.t_sigma, (px[0] - a.xy[0]) / a.sigma_x, (px[1] - a.xy[1]) / a.sigma_y,
                   band / a.z_scale]
    expect += [0.2 / SPIN_PRIOR_SIGMA, -0.1 / SPIN_PRIOR_SIGMA]
    assert res[16:] == pytest.approx(np.array(expect) * 2.0, abs=1e-6)
    assert res[16 + 3] == 0.0 and res[16 + 7] > 0 and res[16 + 11] > 0
    assert res[16] == pytest.approx((0.2 - 0.25) / (2 / 30) * 2.0)


def test_residuals_batch_rows_are_independent_and_equal_to_one_row_at_a_time() -> None:
    times, _, obs, KR, Kt = _residual_setup()
    anchors = (Anchor(t=0.25, xy=np.array([0.5, 2.0])),)
    other = TRUTH.copy()
    other[[0, 1, 9]] += [0.1, -0.2, 0.05]
    P = np.vstack([np.append(TRUTH, 0.2), np.append(other, 0.22)])
    both = _residuals_batch(P, times, obs, KR, Kt, anchors, 1.5)
    for b in range(2):
        assert both[b] == pytest.approx(_residuals_batch(P[b:b + 1], times, obs, KR, Kt, anchors, 1.5)[0], abs=1e-9)
    assert not np.allclose(both[0], both[1])
