from __future__ import annotations

from courtvision.eval.spot import (
    Event,
    first_bounce_flags,
    label_events,
    match_events,
    match_name,
    per_match,
    score_clip,
    summary,
)


def _clip(video: str = "wimbledon_2019_final_100_400") -> dict:
    ev = [("far_court_serve", 10), ("near_court_bounce", 19), ("near_court_bounce", 22),  # 2nd bounce
          ("near_court_swing", 30), ("far_court_bounce", 45), ("far_court_swing", 60)]
    return {"video": video, "fps": 25.0, "num_frames": 300,
            "events": [{"label": lab, "frame": f, "comment": ""} for lab, f in ev]}


def _report() -> dict:
    return {"hits": [{"frame": 11, "side": "far", "serve": True}, {"frame": 29, "side": "near", "serve": False},
                     {"frame": 90, "side": "far", "serve": False}],
            "calibration": {"valid_frames": 270},
            "flights": [{"kind": "serve", "hitter": "far", "receiver": "near", "contact_frame": 10.4,
                         "start_frame": 11, "bounce": {"frame": 20.0, "y": -5.0}},
                        # receiver says far, but the bounce landed on the near half (y < 0): side comes from y
                        {"kind": "rally", "hitter": "near", "receiver": "far", "contact_frame": None,
                         "start_frame": 29, "bounce": {"frame": 47.0, "y": -2.0}}]}


def test_labels_split_into_kinds_and_sides() -> None:
    ev = label_events(_clip())
    assert sorted(e.kind for e in ev if e.frame == 10) == ["serve", "serve_toss", "stroke"]
    assert Event("bounce", "near", 19.0) in ev and Event("stroke", "far", 60.0) in ev
    assert first_bounce_flags(ev) == [True, False, True]  # 22 is the second bounce after the serve


def test_matching_is_one_to_one_and_closest_first() -> None:
    ref = [Event("bounce", "near", 10.0), Event("bounce", "near", 12.0)]
    sys = [Event("bounce", "near", 11.5)]
    assert match_events(ref, sys, 2.5) == [(1, 0)]  # gap 0.5 beats gap 1.5; the system event is used once
    assert match_events(ref, [Event("bounce", "near", 15.0)], 2.5) == []


def test_score_clip_and_summary() -> None:
    row = score_clip(_clip(), _report())
    b = row["bounce"]  # tolerance 0.1 s = 2.5 frames at 25 fps
    assert (b["labelled"], b["system"], b["matched"]) == (3, 2, 2)  # 20~19 and 47~45
    assert b["same_side"] == 1  # 47 landed near (y < 0) but the label says far: side disagrees
    assert (b["first_labelled"], b["first_matched"]) == (2, 2) and b["dt"] == [1.0, 2.0]
    assert b["sweep"] == {"0.05": 1, "0.1": 2, "0.2": 2}
    st = row["stroke"]
    assert (st["labelled"], st["system"], st["matched"], st["same_side"]) == (3, 3, 2, 2)
    assert row["serve"]["matched"] == 1 and abs(row["serve"]["dt"][0] - 0.4) < 1e-9
    assert row["serve_toss"]["matched"] == 1 and row["calibrated_share"] == 0.9
    s = summary([row])
    assert s["bounce"]["recall"] == 2 / 3 and s["bounce"]["precision"] == 1.0
    assert s["bounce"]["first_recall"] == 1.0 and s["stroke"]["same_side"] == 1.0
    assert s["bounce"]["recall_sweep"]["0.05"] == 1 / 3 and s["events_per_clip"] == 6.0


def test_failed_clip_counts_as_all_missed_and_matches_are_kept_apart() -> None:
    assert match_name("usopen_2019_mens_final_medvedev_nadal_100940_101296") == "usopen_2019_mens_final_medvedev_nadal"
    rows = [score_clip(_clip("a_1_2"), _report()), score_clip(_clip("b_1_2"), None)]
    s = summary(rows)
    assert s["failed"] == 1 and s["bounce"]["recall"] == 2 / 6 and s["calibrated_share"] == 0.9
    pm = per_match(rows)
    assert pm["a"]["bounce"]["recall"] == 2 / 3 and pm["b"]["bounce"]["recall"] == 0.0


def test_a_system_serve_before_a_labelled_serve_of_the_same_side_is_set_apart_as_a_likely_fault() -> None:
    clip = {"video": "m_0_1000", "fps": 25.0, "num_frames": 1000,
            "events": [{"label": "far_court_serve", "frame": 400, "comment": ""}]}
    rep = {"hits": [], "calibration": {"valid_frames": 1000},
           "flights": [{"kind": "serve", "hitter": "far", "contact_frame": 100.0, "start_frame": 100},  # 12 s before
                       {"kind": "serve", "hitter": "far", "contact_frame": 401.0, "start_frame": 401},
                       {"kind": "serve", "hitter": "near", "contact_frame": 300.0, "start_frame": 300}]}  # other side
    row = score_clip(clip, rep)
    assert row["serve"]["matched"] == 1 and row["serve"]["before_label"] == 1
    s = summary([row])["serve"]
    assert s["precision"] == 1 / 3 and s["precision_excl_faults"] == 1 / 2 and s["before_label"] == 1
