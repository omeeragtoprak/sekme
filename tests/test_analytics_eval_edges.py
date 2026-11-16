"""Player speed and distance (analytics/kinematics.py), Review Room labels turned into measured errors
(eval/labels.py) and the sealing of evaluations to the code and files that made them (eval/provenance.py), at their
edges. Values checked by running the code (gaps found by a test review); behaviour pinned as it is but listed in
docs/ARCHITECTURE.md ("Known issues") is marked there."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import courtvision.eval.provenance as pv
from courtvision.analytics.kinematics import _fill_short_gaps, _segments, hampel, player_kinematics
from courtvision.eval.labels import (
    FOOTPRINT_HALF,
    evaluate,
    px_scale_cm,
    serve_state,
    serve_summary,
    summarize,
    world_from_px,
    zone_from_name,
)

# --- 1. kinematics -------------------------------------------------------------------------------------------------


def _walk(n: int, fps: float, v: float = 5.0) -> np.ndarray:
    """A player walking along x at ``v`` m/s."""
    return np.c_[v * np.arange(n) / fps, np.zeros(n)]


def test_a_player_walking_5_ms_is_18_kmh_with_the_right_distance_at_any_frame_rate() -> None:
    for fps in (10, 25, 30, 60):
        n = int(3 * fps)
        k = player_kinematics(_walk(n, fps), fps)
        assert k.peak_kmh == pytest.approx(18.0, abs=1e-6)  # 5 m/s * 3.6
        assert k.mean_kmh == pytest.approx(18.0, abs=1e-6)
        assert k.distance_m[-1] == pytest.approx(5.0 * (n - 1) / fps, abs=1e-6)  # frames-1 steps, not frames
        assert k.outliers_removed == 0
        assert np.isfinite(k.speed_kmh).all()


def test_speed_is_a_vector_norm_in_both_ground_axes_and_distance_is_in_metres() -> None:
    w = _walk(90, 30)
    diag = np.c_[w[:, 0] * 0.6, w[:, 0] * 0.8]  # same 5 m/s, along a 3-4-5 direction
    k = player_kinematics(diag, 30)
    assert k.peak_kmh == pytest.approx(18.0, abs=1e-6)
    assert k.distance_m[-1] == pytest.approx(5.0 * 89 / 30, abs=1e-6)
    assert k.position[-1] == pytest.approx(diag[-1], abs=1e-9)  # a straight line is not bent by the smoothing


def test_the_speed_follows_a_constant_acceleration_instead_of_lagging_behind_it() -> None:
    t = np.arange(90) / 30
    k = player_kinematics(np.c_[t**2, 0 * t], 30)  # x = t^2 -> v = 2t m/s
    for i in (10, 45, 80):
        assert k.speed_kmh[i] == pytest.approx(2 * t[i] * 3.6, abs=1e-6)  # the quadratic window is exact on it


def test_the_peak_is_the_98th_percentile_of_the_speeds_not_their_maximum_and_the_mean_is_their_mean() -> None:
    t = np.arange(150) / 30
    k = player_kinematics(np.c_[t**2, 0 * t], 30)
    s = k.speed_kmh[np.isfinite(k.speed_kmh)]
    assert k.peak_kmh == pytest.approx(float(np.percentile(s, 98)), abs=1e-9)
    assert k.peak_kmh < float(s.max()) - 0.5  # a ramp has its maximum at the last frame
    assert k.peak_kmh > float(s.mean())
    assert k.mean_kmh == pytest.approx(float(s.mean()), abs=1e-9)


def test_one_glitch_frame_does_not_set_the_peak_or_the_distance() -> None:
    w = _walk(90, 30)
    w[45] += [3.0, 3.0]  # box glitch of 4 m
    k = player_kinematics(w, 30)
    assert k.outliers_removed == 1
    assert k.peak_kmh == pytest.approx(18.0, abs=1e-6)
    assert k.distance_m[-1] == pytest.approx(5.0 * 89 / 30, abs=1e-6)  # the hole is interpolated back onto the line
    assert not np.isnan(k.position).any()


def test_a_glitch_of_up_to_four_frames_is_removed_but_five_in_a_row_win_the_median() -> None:
    for b, removed in ((2, 2), (3, 3), (4, 4)):
        w = _walk(90, 30)
        w[40:40 + b] += [0.0, 10.0]  # a brief identity swap onto another player
        k = player_kinematics(w, 30)
        assert k.outliers_removed == removed
        assert k.peak_kmh == pytest.approx(18.0, abs=1e-6)
    w = _walk(90, 30)
    w[40:45] += [0.0, 10.0]
    k = player_kinematics(w, 30)
    # Known issue: the Hampel window (half 4) is nine frames, so a swap of five frames is the majority and is kept:
    # the report then shows a 61.6 km/h peak and 26 m of distance for a player who walked 14.8 m.
    assert k.outliers_removed == 0
    assert k.peak_kmh > 60 and k.distance_m[-1] > 25


def test_hampel_flags_a_stationary_jump_only_above_the_35_cm_floor() -> None:
    for d, removed in ((0.30, 0), (0.34, 0), (0.36, 1), (0.40, 1)):
        w = np.zeros((20, 2))
        w[10, 0] = d
        out, n = hampel(w)
        assert n == removed
        assert np.isnan(out[10]).all() == bool(removed)
        assert np.isfinite(np.delete(out, 10, axis=0)).all()  # only the offender is blanked
        assert np.isfinite(w).all()  # the input is not modified


def test_on_a_walking_player_the_mad_term_not_the_floor_sets_the_hampel_threshold() -> None:
    w = _walk(90, 30)
    w[45] += [0.0, 0.5]  # well above the 0.35 m floor, but the walking spread of the window is about 0.5 m
    k = player_kinematics(w, 30)
    assert k.outliers_removed == 0
    assert k.peak_kmh == pytest.approx(18.0, abs=0.1)  # kept, but the smoothing absorbs it


def test_hampel_needs_three_finite_points_in_the_window_and_skips_unknown_frames() -> None:
    two = np.full((3, 2), np.nan)
    two[0], two[1] = (0.0, 0.0), (5.0, 0.0)
    assert hampel(two)[1] == 0  # two points: no median to judge against
    three = np.zeros((3, 2))
    three[1] = (5.0, 0.0)
    assert hampel(three)[1] == 1  # three are enough
    gap = _walk(90, 30)
    gap[10, 0] = np.nan  # a half-known point is unknown, not an outlier
    out, n = hampel(gap)
    assert n == 0 and np.isnan(out[10, 0]) and np.isfinite(out[10, 1])


def test_segments_are_half_open_runs_of_valid_frames() -> None:
    assert _segments(np.array([1, 1, 0, 1, 0, 0, 1], bool)) == [(0, 2), (3, 4), (6, 7)]
    assert _segments(np.array([], bool)) == []
    assert _segments(np.array([0, 0], bool)) == []
    assert _segments(np.array([1, 1, 1], bool)) == [(0, 3)]


def test_gaps_up_to_max_gap_frames_are_filled_linearly_and_longer_ones_are_left() -> None:
    w = np.full((10, 2), np.nan)
    w[0], w[5] = (0.0, 10.0), (5.0, 0.0)
    out = _fill_short_gaps(w, max_gap=4)  # four missing frames between the two known ones
    assert out[1:5] == pytest.approx(np.array([[1, 8], [2, 6], [3, 4], [4, 2]], float))
    assert np.isnan(out[6:]).all()  # nothing is extrapolated after the last known frame
    assert np.isnan(_fill_short_gaps(w, max_gap=3)[1:5]).all()  # a gap of four is too long for max_gap 3
    assert np.isnan(w[1]).all()  # input untouched
    lead = np.full((6, 2), np.nan)
    lead[3:] = 1.0
    assert np.isnan(_fill_short_gaps(lead, max_gap=9)[:3]).all()  # nor before the first


def test_the_gap_filled_is_a_fifth_of_a_second_with_a_floor_of_two_frames() -> None:
    def known(fps: int, gap: int) -> int:
        w = _walk(fps * 6, fps)
        w[10:10 + gap] = np.nan
        return int(np.isfinite(player_kinematics(w, fps).position[:, 0]).sum())

    assert [known(30, g) for g in (6, 7)] == [180, 173]  # int(0.2 * 30) = 6 frames
    assert [known(60, g) for g in (12, 13)] == [360, 347]  # 12 frames at 60 fps
    assert [known(10, g) for g in (2, 3)] == [60, 57]  # 0.2 s is 2 frames
    assert [known(5, g) for g in (2, 3)] == [30, 27]  # 0.2 s is 1 frame: the floor of 2 applies


def test_a_filled_gap_keeps_the_distance_and_a_long_gap_is_left_out_of_it() -> None:
    full = player_kinematics(_walk(90, 30), 30).distance_m[-1]
    short = _walk(90, 30)
    short[40:46] = np.nan  # 6 frames: filled
    assert player_kinematics(short, 30).distance_m[-1] == pytest.approx(full, abs=1e-6)
    long = _walk(90, 30)
    long[40:47] = np.nan  # 7 frames: the player's movement across the gap is not counted
    k = player_kinematics(long, 30)
    assert k.distance_m[-1] == pytest.approx(13.5, abs=1e-6)
    assert k.distance_m[46] == k.distance_m[39]  # flat across the gap
    assert np.isnan(k.position[40:47]).all()


def test_the_smoothing_window_is_0_6_s_made_odd_and_a_segment_shorter_than_it_has_no_speed() -> None:
    # 30 fps: round(18) | 1 = 19 frames
    assert [int(np.isfinite(player_kinematics(_walk(n, 30), 30).speed_kmh).sum()) for n in (17, 18, 19)] == [0, 0, 19]
    # 60 fps: 37 frames
    assert [int(np.isfinite(player_kinematics(_walk(n, 60), 60).speed_kmh).sum()) for n in (36, 37)] == [0, 37]
    # 5 fps: 0.6 s is 3 frames, the minimum of 5 applies
    assert [int(np.isfinite(player_kinematics(_walk(n, 5), 5).speed_kmh).sum()) for n in (4, 5)] == [0, 5]


def test_a_short_segment_keeps_its_position_and_distance_but_claims_no_speed() -> None:
    k = player_kinematics(_walk(18, 30), 30)
    assert np.isnan(k.speed_kmh).all()
    assert k.peak_kmh == 0.0 and k.mean_kmh == 0.0
    # Known issue: the distance of an unsmoothed short segment is still summed from raw positions, so a clip made
    # only of such segments reports 2.8 m run at a speed of 0 (and raw jitter would count in full).
    assert k.distance_m[-1] == pytest.approx(5.0 * 17 / 30, abs=1e-9)
    assert k.position == pytest.approx(_walk(18, 30), abs=1e-12)


def test_the_mean_speed_averages_only_the_frames_that_have_a_speed() -> None:
    w = np.vstack([_walk(60, 30), np.full((10, 2), np.nan), _walk(10, 30) + [20.0, 0.0]])
    k = player_kinematics(w, 30)
    assert int(np.isfinite(k.speed_kmh).sum()) == 60  # the 10-frame tail is too short
    assert k.mean_kmh == pytest.approx(18.0, abs=1e-6) and k.peak_kmh == pytest.approx(18.0, abs=1e-6)
    assert k.distance_m[59] == k.distance_m[69] == pytest.approx(5.0 * 59 / 30, abs=1e-6)  # nothing over the gap
    assert k.distance_m[-1] == pytest.approx(5.0 * 59 / 30 + 5.0 * 9 / 30, abs=1e-6)  # the tail adds its own steps


def test_smoothing_removes_jitter_from_the_distance_of_a_standing_player() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.05, (90, 2))  # 5 cm foot-point noise
    raw = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
    k = player_kinematics(w, 30)
    assert raw > 5.0
    assert k.distance_m[-1] < 1.0 and k.peak_kmh < 3.0
    still = player_kinematics(np.tile([1.0, 2.0], (60, 1)), 30)
    assert still.distance_m[-1] == pytest.approx(0.0, abs=1e-9) and still.peak_kmh == pytest.approx(0.0, abs=1e-9)


def test_unknown_leading_frames_stay_unknown_and_add_no_distance() -> None:
    w = _walk(90, 30)
    w[:5] = np.nan
    k = player_kinematics(w, 30)
    assert np.isnan(k.position[:5]).all() and np.isnan(k.speed_kmh[:5]).all()
    assert (k.distance_m[:6] == 0).all()
    assert k.distance_m[-1] == pytest.approx(5.0 * 84 / 30, abs=1e-6)


def test_no_data_at_all_gives_zeros_not_errors() -> None:
    k = player_kinematics(np.empty((0, 2)), 30)
    assert len(k.speed_kmh) == 0 and k.peak_kmh == 0.0 and k.mean_kmh == 0.0
    k = player_kinematics(np.full((10, 2), np.nan), 30)
    assert k.peak_kmh == 0.0 and k.mean_kmh == 0.0 and (k.distance_m == 0).all()


# --- 2. labels -----------------------------------------------------------------------------------------------------

H = [50.0, 0.0, 640.0, 0.0, -50.0, 700.0, 0.0, 0.0, 1.0]  # 2 cm per pixel, world (0, 12) at pixel (640, 100)


def _call(**kw) -> dict:
    return {"x": 0.0, "y": 5.0, "cov": [[0.01, 0.0], [0.0, 0.04]], "zone": "far-singles", "verdict": "IN", **kw}


def test_a_pixel_maps_through_the_inverse_homography_and_a_flat_list_equals_a_matrix() -> None:
    assert world_from_px(H, (640, 100)) == pytest.approx([0.0, 12.0], abs=1e-9)
    assert world_from_px(np.array(H).reshape(3, 3), (690, 200)) == pytest.approx([1.0, 10.0], abs=1e-9)
    persp = [*H[:6], 0.0, 0.001, 1.0]  # the divide by w matters once the bottom row is not (0, 0, 1)
    assert world_from_px(persp, (640, 100)) == pytest.approx([0.15329341, 11.9760479], abs=1e-6)


def test_px_scale_is_in_centimetres_and_takes_the_coarser_image_direction() -> None:
    assert px_scale_cm(H, (640, 100)) == pytest.approx(2.0, abs=1e-9)
    assert px_scale_cm([50.0, 0, 640, 0, -25.0, 700, 0, 0, 1.0], (640, 100)) == pytest.approx(4.0, abs=1e-6)
    assert px_scale_cm([25.0, 0, 640, 0, -50.0, 700, 0, 0, 1.0], (640, 100)) == pytest.approx(4.0, abs=1e-6)
    assert px_scale_cm([*H[:6], 0.0, 0.001, 1.0], (640, 100)) == pytest.approx(2.0239520958, abs=1e-6)


def test_zone_names_round_trip_and_unknown_names_are_rejected() -> None:
    for name in ("far-singles", "far-doubles", "near-singles", "near-doubles", "service-near-left",
                 "service-near-right", "service-far-left", "service-far-right"):
        assert zone_from_name(name).name == name
    right = zone_from_name("service-far-right")
    assert right.x1 == pytest.approx(4.115) and right.x0 == pytest.approx(-0.025)  # the centre line is in both boxes
    assert zone_from_name("near-doubles").x1 == pytest.approx(5.485)
    for bad in ("", "x", "near", "a-b-c", "service-near", "far-bogus", "service-near-up"):
        with pytest.raises(ValueError):
            zone_from_name(bad)


def test_the_error_is_in_centimetres_and_the_mahalanobis_distance_uses_the_calls_covariance() -> None:
    r = evaluate([0.1, 5.0], _call())
    assert r["error_cm"] == 10.0 and r["mahalanobis2"] == 1.0  # 0.1 m along an axis with variance 0.01
    assert evaluate([0.0, 5.2], _call())["mahalanobis2"] == 1.0  # 0.2 m along the axis with variance 0.04
    assert evaluate([0.2, 5.0], _call())["mahalanobis2"] == 4.0
    assert evaluate([0.123456, 5.0], _call())["error_cm"] == 12.3  # rounded to a millimetre
    assert r["system"] == "IN" and r["truth"] == "IN" and r["agrees"] is True


def test_the_95_percent_ellipse_is_chi_square_5_991_and_a_singular_covariance_does_not_crash() -> None:
    cov = np.eye(2) * 0.01
    inside = evaluate([0.24, 5.0], _call(cov=cov))  # d2 = 5.76
    outside = evaluate([0.25, 5.0], _call(cov=cov))  # d2 = 6.25
    assert inside["inside_95"] is True and outside["inside_95"] is False
    zero = _call(cov=np.zeros((2, 2)))
    assert evaluate([0.001, 5.0], zero)["mahalanobis2"] == 10000.0  # the 1e-10 ridge keeps it finite
    assert evaluate([0.0, 5.0], zero)["mahalanobis2"] == 0.0


def test_agreement_is_three_valued_and_a_review_call_is_neither_right_nor_wrong() -> None:
    assert evaluate([0.0, 5.0], _call(verdict="REVIEW"))["agrees"] is None
    assert evaluate([0.0, 5.0], _call(verdict="IN"))["agrees"] is True
    assert evaluate([0.0, 5.0], _call(verdict="OUT"))["agrees"] is False
    out = evaluate([0.0, 12.0], _call(verdict="IN"))  # 12.0 m is past the baseline (11.885) plus the 2 cm footprint
    assert out["truth"] == "OUT" and out["agrees"] is False and out["truth_line"] == "far baseline"


def test_the_contact_footprint_is_2_cm_and_the_edge_of_it_is_still_in() -> None:
    assert FOOTPRINT_HALF == 0.02
    base = 11.885
    on_line = evaluate([0.0, base], _call())
    assert on_line["truth"] == "IN" and on_line["truth_margin_cm"] == 2.0 and on_line["truth_line"] == "far baseline"
    edge = evaluate([0.0, base + FOOTPRINT_HALF], _call())
    assert edge["truth"] == "IN" and edge["truth_margin_cm"] == 0.0  # inclusive at the grown edge
    assert evaluate([0.0, base + FOOTPRINT_HALF + 1e-6], _call())["truth"] == "OUT"
    side = evaluate([4.115 + FOOTPRINT_HALF + 0.001, 5.0], _call())
    assert side["truth"] == "OUT" and side["truth_margin_cm"] == -0.1 and side["truth_line"] == "right singles sideline"
    left = evaluate([-4.115 - FOOTPRINT_HALF - 0.001, 5.0], _call())
    assert left["truth"] == "OUT" and left["truth_line"] == "left singles sideline"


def test_a_bounce_on_the_wrong_side_of_the_net_is_out_with_a_margin_of_its_distance_to_the_net() -> None:
    r = evaluate([0.0, -0.7], _call())
    assert r["truth"] == "OUT" and r["truth_line"] == "net" and r["truth_margin_cm"] == -70.0
    near = _call(zone="near-singles", y=-5.0)
    r = evaluate([0.0, 0.5], near)
    assert r["truth"] == "OUT" and r["truth_line"] == "net" and r["truth_margin_cm"] == -50.0
    assert evaluate([0.0, 0.0], near)["truth"] == "IN"  # exactly on the net line counts for either half
    assert evaluate([0.0, 0.0], _call())["truth"] == "IN"
    assert evaluate([0.0, -0.5], near)["truth"] == "IN"


def test_summarize_keeps_the_latest_label_per_shot_whatever_the_file_order() -> None:
    def rec(fid: int, created: str, err: float, agrees: bool | None, truth: str = "IN", in95: bool = True) -> dict:
        return {"flight_id": fid, "created": created,
                "eval": {"error_cm": err, "agrees": agrees, "truth": truth, "inside_95": in95}}

    records = [rec(1, "2026-01-03", 5.0, True),
               rec(1, "2026-01-01", 99.0, False),  # older, listed after: must lose
               rec(2, "2026-01-01", 10.0, False, in95=False),
               rec(3, "2026-01-01", 20.0, None, truth="OUT"),
               rec(4, "2026-01-01", 30.0, None, truth="IN"),
               rec(5, "2026-01-01", 40.0, True)]
    s = summarize(records)
    assert s["n"] == 5
    assert s["error_cm_median"] == 20.0 and s["error_cm_p90"] == 36.0  # percentile of [5, 10, 20, 30, 40]
    assert s["inside_95_share"] == 0.8
    assert (s["decisive"], s["agreements"], s["reviews"]) == (3, 2, 2)
    assert s["review_truth"] == {"IN": 1, "OUT": 1}
    assert summarize([]) == {"n": 0}
    same = summarize([rec(1, "t", 1.0, True), rec(1, "t", 2.0, True)])  # equal stamps: the later line wins
    assert same["error_cm_median"] == 2.0


def test_serve_state_replays_the_log_in_order() -> None:
    log = [{"type": "serve", "id": "a", "frame": 10}, {"type": "serve", "id": "b", "frame": 20},
           {"type": "serve", "id": "a", "frame": 11},  # same id: replaced
           {"type": "remove", "target": "zzz"},  # unknown target: ignored
           {"type": "remove", "target": "b"},
           {"type": "serve", "id": "b", "frame": 30},  # re-added after removal
           {"type": "other", "id": "q"},
           {"type": "complete", "complete": True}]
    serves, complete = serve_state(log)
    assert [(s["id"], s["frame"]) for s in serves] == [("a", 11), ("b", 30)] and complete is True
    assert serve_state([]) == ([], False)
    gone, _ = serve_state([{"type": "serve", "id": "a", "frame": 1}, {"type": "remove", "target": "a"}])
    assert gone == []  # removed for good unless it is added again
    serves, complete = serve_state([{"type": "complete", "complete": True}, {"type": "complete", "complete": False}])
    assert serves == [] and complete is False
    first = serve_state([{"type": "remove", "target": "a"}, {"type": "serve", "id": "a", "frame": 1}])[0]
    assert first[0]["frame"] == 1  # a removal before the serve exists does not block it


def _s(i: str, f: int) -> dict:
    return {"id": i, "type": "serve", "frame": f}


def test_the_serve_tolerance_is_in_seconds_so_it_scales_with_the_frame_rate() -> None:
    assert serve_summary([_s("a", 100)], [120.0], fps=60)["matched"] == 1  # 0.5 s = 30 frames
    assert serve_summary([_s("a", 100)], [120.0], fps=30)["matched"] == 0  # 0.5 s = 15 frames
    assert serve_summary([_s("a", 100)], [130.0], fps=60)["matched"] == 1
    assert serve_summary([_s("a", 100)], [131.0], fps=60)["matched"] == 0
    assert serve_summary([_s("a", 100)], [120.0], fps=30, tol_s=1.0)["matched"] == 1
    assert serve_summary([_s("a", 100)], [80.0], fps=30)["matched"] == 0  # before the label counts the same way
    assert serve_summary([_s("a", 100)], [85.0], fps=30)["matched"] == 1
    assert serve_summary([_s("a", 100)], [85.0], fps=30)["tol_s"] == 0.5


def test_pairing_takes_the_smallest_gaps_first_across_labels_not_label_order() -> None:
    s = serve_summary([_s("a", 100), _s("b", 110)], [108.0, 95.0], fps=30)
    rows = {r["id"]: r for r in s["rows"]}
    assert s["matched"] == 2 and s["unmatched_system"] == []
    assert rows["b"]["matched_system_frame"] == 108.0 and rows["b"]["dt_frames"] == -2.0  # system minus label
    assert rows["a"]["matched_system_frame"] == 95.0 and rows["a"]["dt_frames"] == -5.0
    tie = serve_summary([_s("a", 100), _s("b", 110)], [105.0], fps=30)  # equidistant: the earlier label wins
    assert [r["matched_system_frame"] for r in tie["rows"]] == [105.0, None]


def test_recall_and_precision_are_ratios_rounded_to_three_places_and_withheld_when_undefined() -> None:
    recs = [_s("a", 100), _s("b", 200), _s("c", 300), {"type": "complete", "complete": True}]
    s = serve_summary(recs, [100.0, 200.0, 900.0], fps=30)
    assert s["recall"] == 0.667 and s["precision"] == 0.667 and s["complete"] is True
    assert s["unmatched_system"] == [900.0] and [r["id"] for r in s["rows"]] == ["a", "b", "c"]
    assert serve_summary(recs, [], fps=30)["precision"] is None  # complete, but the system found nothing
    assert serve_summary(recs, [], fps=30)["recall"] == 0.0
    assert serve_summary([_s("a", 1)], [1.0], fps=30)["precision"] is None  # not declared complete
    assert serve_summary([{"type": "complete", "complete": True}], [5.0], fps=30)["recall"] is None
    assert serve_summary([_s("a", 100)], [101.234], fps=30)["rows"][0]["dt_frames"] == 1.23  # rounded to 0.01 frame


# --- 3. provenance -------------------------------------------------------------------------------------------------


def test_sha256_matches_hashlib_for_empty_small_and_multi_chunk_files(tmp_path: Path) -> None:
    for name, data in (("empty", b""), ("abc", b"abc"), ("big", os.urandom(1 << 20) * 2 + b"tail")):
        p = tmp_path / name
        p.write_bytes(data)
        assert pv.sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert pv.sha256_file(tmp_path / "abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (tmp_path / "abc").write_bytes(b"abd")  # one byte differs
    assert pv.sha256_file(tmp_path / "abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(FileNotFoundError):
        pv.sha256_file(tmp_path / "missing")


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(root), *args], capture_output=True, text=True,
                       check=True)
    return r.stdout.strip()


def _commit(root: Path, msg: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny repository standing in for the project: cli -> pipeline -> perception.ball, events.serves."""
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.com"), ("GIT_COMMITTER_NAME", "t"),
                 ("GIT_COMMITTER_EMAIL", "t@example.com")):
        monkeypatch.setenv(k, v)
    root = tmp_path / "repo"
    pkg = root / "src" / "courtvision"
    files = {
        "__init__.py": "",
        "cli.py": "import json\n",
        "pipeline.py": "from courtvision.perception import ball\n\n\ndef f():\n    import courtvision.io.video\n",
        "perception/__init__.py": "",
        "perception/ball.py": "from courtvision.geometry.court import Half\n",
        "geometry/__init__.py": "",
        "geometry/court.py": "",
        "io/__init__.py": "",
        "io/video.py": "",
        "events/__init__.py": "",
        "events/serves.py": "import numpy\n",
        "api/__init__.py": "",
        "api/app.py": "x = 1\n",
    }
    for rel, text in files.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    (root / "notes.txt").write_text("a\n")
    _git(root, "init", "-q")
    _commit(root, "first")
    monkeypatch.setattr(pv, "ROOT", root)
    return root


def test_analysis_code_is_the_import_closure_of_the_analyze_command(repo: Path) -> None:
    got = pv.analysis_code()
    assert got == sorted(f"src/courtvision/{f}" for f in (
        "__init__.py", "cli.py", "pipeline.py", "perception/__init__.py", "perception/ball.py",
        "geometry/__init__.py", "geometry/court.py", "io/__init__.py", "io/video.py", "events/__init__.py",
        "events/serves.py"))  # a function-level import and "from pkg import module" are followed; api/ is not
    assert "src/courtvision/api/app.py" not in got


def test_a_relative_import_in_the_analysis_code_is_refused(repo: Path) -> None:
    (repo / "src/courtvision/pipeline.py").write_text("from . import io\n")
    with pytest.raises(SystemExit):
        pv.analysis_code()


def test_a_run_is_the_same_code_only_when_analysis_files_are_unchanged_since_its_marker(
    repo: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert pv.same_code(run) is False  # no code_sha.txt: unknown, so not the same
    sha = _git(repo, "rev-parse", "HEAD")
    (run / "code_sha.txt").write_text(sha + "\n")  # the trailing newline is stripped
    assert pv.same_code(run) is True
    _commit_change = repo / "src/courtvision/api/app.py"
    _commit_change.write_text("x = 2\n")
    (repo / "notes.txt").write_text("b\n")
    _commit(repo, "api and notes only")
    assert pv.same_code(run) is True  # not analysis code: the run is not stale
    (repo / "src/courtvision/events/serves.py").write_text("import numpy\nx = 1\n")
    assert pv.same_code(run) is True  # uncommitted edits are not in HEAD, which is what is compared
    _commit(repo, "serves")
    assert pv.same_code(run) is False
    (run / "code_sha.txt").write_text(_git(repo, "rev-parse", "HEAD"))
    assert pv.same_code(run) is True  # re-marked at the new HEAD
    (run / "code_sha.txt").write_text("0" * 40)
    assert pv.same_code(run) is False  # a marker git does not know


def test_a_change_to_an_indirectly_imported_module_makes_a_run_stale(repo: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "code_sha.txt").write_text(_git(repo, "rev-parse", "HEAD"))
    (repo / "src/courtvision/geometry/court.py").write_text("HALF = 1\n")  # cli -> pipeline -> ball -> court
    _commit(repo, "court")
    assert pv.same_code(run) is False


def test_clean_sees_staged_unstaged_and_untracked_changes_only_under_the_given_paths(repo: Path) -> None:
    assert pv.clean(["src"]) is True
    (repo / "src/courtvision/new.py").write_text("")
    assert pv.clean(["src"]) is False  # untracked
    assert pv.clean(["notes.txt"]) is True  # elsewhere
    (repo / "src/courtvision/new.py").unlink()
    (repo / "src/courtvision/api/app.py").write_text("x = 3\n")
    assert pv.clean(["src"]) is False  # unstaged
    _git(repo, "add", "-A")
    assert pv.clean(["src"]) is False  # staged
    assert pv.clean(["notes.txt"]) is True


def test_head_sha_is_the_full_commit_hash(repo: Path) -> None:
    assert pv.head_sha() == _git(repo, "rev-parse", "HEAD") and len(pv.head_sha()) == 40


def test_files_are_tracked_and_clean_only_when_committed_and_unmodified(repo: Path) -> None:
    f = repo / "src/courtvision/cli.py"
    assert pv.tracked_and_clean([]) is False  # nothing to vouch for
    assert pv.tracked_and_clean([f]) is True
    extra = repo / "extra.bin"
    extra.write_bytes(b"1")
    assert pv.tracked_and_clean([f, extra]) is False  # untracked
    _commit(repo, "extra")
    assert pv.tracked_and_clean([f, extra]) is True
    extra.write_bytes(b"2")
    assert pv.tracked_and_clean([f, extra]) is False  # modified since its commit
    assert pv.tracked_and_clean([f]) is True


def test_code_is_unchanged_since_a_sha_only_when_clean_and_equal_to_it(repo: Path) -> None:
    code = ["src/courtvision/cli.py", "src/courtvision/pipeline.py"]
    sha = pv.head_sha()
    assert pv.code_unchanged_since(sha, code) is True
    (repo / "src/courtvision/cli.py").write_text("import json\nx = 1\n")
    assert pv.code_unchanged_since(sha, code) is False  # edited, not committed
    assert pv.code_unchanged_since(sha, ["src/courtvision/pipeline.py"]) is True  # the edit is outside these paths
    _commit(repo, "cli")
    assert pv.code_unchanged_since(sha, code) is False  # committed since the run started
    assert pv.code_unchanged_since(pv.head_sha(), code) is True
    assert pv.code_unchanged_since("0" * 40, code) is False  # unknown sha
