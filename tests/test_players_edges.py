"""Near and far player selection at its edges (perception/players.py): the sides' sign, the play area, id switches
and re-seeding, a second person or a ball kid on one half, uncalibrated frames, the speed and size gates, and the
IoU tracker. Values checked by running the code (gaps found by a test review)."""

from __future__ import annotations

import numpy as np

from courtvision.perception.court import FrameCalibration
from courtvision.perception.players import (
    IouTracker,
    PersonDet,
    _plausible,
    expected_height_px,
    select_players,
)


def _calib(cam, n, invalid=()):
    H = cam.ground_homography()
    Hi = np.linalg.inv(H)
    return [FrameCalibration(False) if i in invalid else FrameCalibration(True, H, Hi, cam, 1.0, 0.3)
            for i in range(n)]


def _det(cam, tid, x, y, frac=1.0, conf=0.9):
    foot = cam.project(np.array([[x, y, 0.0]]))[0]
    h = frac * expected_height_px(cam, np.array([x, y]))
    return PersonDet(tid, np.array([foot[0] - 20, foot[1] - h, foot[0] + 20, foot[1]]), conf)


def test_sides_are_not_swapped_and_each_half_has_its_own_player(broadcast_camera):
    cam = broadcast_camera
    dets = [[_det(cam, 1, 1.0, -12.0), _det(cam, 2, -1.0, 12.0)] for _ in range(5)]
    r = select_players(dets, _calib(cam, 5))
    assert np.allclose(r["near"].world, [1.0, -12.0], atol=1e-6) and r["near"].track_ids == [1]
    assert np.allclose(r["far"].world, [-1.0, 12.0], atol=1e-6) and r["far"].track_ids == [2]


def test_people_outside_the_play_area_are_never_picked_even_when_bigger_and_alone(broadcast_camera):
    cam = broadcast_camera
    # a line judge 8 m to the side and a spectator 19 m behind the baseline, both near-side and tall
    for x, y in ((8.0, -12.0), (-8.0, -12.0), (0.0, -19.0), (0.0, -0.5)):
        r = select_players([[_det(cam, 5, x, y, conf=0.99)] for _ in range(4)], _calib(cam, 4))
        assert np.isnan(r["near"].world).all() and r["near"].track_ids == []
        assert np.isnan(r["far"].world).all()


def test_play_area_edges_are_inclusive(broadcast_camera):
    cam = broadcast_camera
    for (x, y), ok in (((7.4, -12.0), True), ((7.6, -12.0), False), ((0.0, -0.9), True), ((0.0, -17.4), True),
                       ((0.0, -17.6), False)):
        r = select_players([[_det(cam, 1, x, y)]] * 3, _calib(cam, 3))
        assert bool(np.isfinite(r["near"].world).all()) is ok, (x, y)


def test_main_track_beats_a_closer_second_person_in_the_same_half(broadcast_camera):
    cam = broadcast_camera
    # player (id 1, conf 0.9) at 1.0, -12; a second full-size person (id 2, lower conf) beside him, nearer
    # the last position in frames where the player is momentarily missed, but the main track wins when seen
    n = 10
    dets = [[_det(cam, 1, 1.0, -12.0), _det(cam, 2, 1.5, -12.0, conf=0.5)] for _ in range(n)]
    r = select_players(dets, _calib(cam, n))
    assert np.allclose(r["near"].world, [1.0, -12.0], atol=1e-6) and r["near"].track_ids == [1]


def test_a_ball_kid_in_the_half_is_ignored_and_a_crossing_kid_does_not_steal_the_track(broadcast_camera):
    cam = broadcast_camera
    n = 12
    dets = []
    for i in range(n):
        row = [_det(cam, 1, 1.0, -12.0)]
        row.append(_det(cam, 9, 4.0 - 0.3 * i, -12.0, frac=0.4, conf=0.95))  # crouching kid walking past
        dets.append(row)
    r = select_players(dets, _calib(cam, n))
    assert np.allclose(r["near"].world, [1.0, -12.0], atol=1e-6)
    assert r["near"].track_ids == [1]


def test_uncalibrated_frame_gives_nan_for_that_frame_only_and_tracking_resumes(broadcast_camera):
    cam = broadcast_camera
    n = 8
    dets = [[_det(cam, 1, 1.0, -12.0)] for _ in range(n)]
    r = select_players(dets, _calib(cam, n, invalid={3}))
    ok = np.isfinite(r["near"].world).all(axis=1)
    assert ok.tolist() == [True, True, True, False, True, True, True, True]
    assert np.isnan(r["near"].boxes[3]).all()


def test_untracked_detections_with_negative_ids_cannot_be_main_but_are_picked_by_the_reseed(broadcast_camera):
    cam = broadcast_camera
    dets = [[_det(cam, -1, 1.0, -12.0)] for _ in range(4)]
    r = select_players(dets, _calib(cam, 4))
    assert np.isfinite(r["near"].world).all()
    assert r["near"].track_ids == [-1]


def test_identity_switch_is_bridged_by_continuity_and_both_ids_are_reported(broadcast_camera):
    cam = broadcast_camera
    n = 10
    dets = [[_det(cam, 1 if i < 5 else 2, 1.0 + 0.05 * i, -12.0)] for i in range(n)]
    r = select_players(dets, _calib(cam, n))
    assert np.isfinite(r["near"].world).all()
    assert r["near"].track_ids == [1, 2]


def test_a_teleporting_id_switch_to_another_person_is_rejected_then_reseeded_after_a_second(broadcast_camera):
    cam = broadcast_camera
    n = 80
    # id 1 present frames 0-9 at x=-5; from frame 10 only id 2 appears at x=+5 (10 m away): not reachable
    dets = [[_det(cam, 1, -5.0, -12.0)] if i < 10 else [_det(cam, 2, 5.0, -12.0)] for i in range(n)]
    r = select_players(dets, _calib(cam, n))
    ok = np.isfinite(r["near"].world).all(axis=1)
    # the first frame after a 10 m jump is accepted: 10 <= 11 * k / 30 + 0.35 needs k = 27 frames since frame 9
    # (frame 36); the re-seed at k > 30 would come later
    assert not ok[10:36].any() and ok[36:].all()


def test_gap_longer_than_fps_reseeds_exactly_one_frame_after_the_second(broadcast_camera):
    cam = broadcast_camera
    # player at -6 for frame 0, reappears at +6 (12 m) after k frames of nothing
    def run(k):
        dets = [[_det(cam, 1, -6.0, -12.0)]] + [[] for _ in range(k - 1)] + [[_det(cam, 1, 6.0, -12.0)]]
        return np.isfinite(select_players(dets, _calib(cam, k + 1))["near"].world[k]).all()
        assert not run(30)  # 12 m in 1.0 s is a teleport and 30 frames is not yet "over a second"
    assert run(31)  # lost for more than a second: re-seeded


def test_speed_gate_boundary_is_inclusive():
    p = PersonDet(1, np.array([0.0, 0.0, 40.0, 100.0]), 0.9)
    last = np.array([0.0, -12.0])
    limit = 11.0 * 3 / 30.0 + 0.35
    assert _plausible(np.array([limit - 1e-6, -12.0]), p, last, 3, 30.0, 100.0)
    assert not _plausible(np.array([limit + 1e-3, -12.0]), p, last, 3, 30.0, 100.0)


def test_size_gate_boundaries_with_clip_height_and_with_camera_height():
    def d(h):
        return PersonDet(1, np.array([0.0, 0.0, 40.0, float(h)]), 0.9)
    w = np.array([0.0, -12.0])
    assert _plausible(w, d(60), None, 0, 30.0, 100.0) and not _plausible(w, d(59), None, 0, 30.0, 100.0)
    assert _plausible(w, d(160), None, 0, 30.0, 100.0) and not _plausible(w, d(161), None, 0, 30.0, 100.0)
    assert _plausible(w, d(50), None, 0, 30.0, 999.0, 100.0) and not _plausible(w, d(49), None, 0, 30.0, 999.0, 100.0)
    assert _plausible(w, d(160), None, 0, 30.0, 999.0, 100.0) and not _plausible(w, d(161), None, 0, 30.0, 999.0, 100.0)
    assert _plausible(w, d(1), None, 0, 30.0, None)  # no reference at all: any size passes


def test_expected_height_needs_camera_and_point_and_shrinks_with_distance(broadcast_camera):
    assert expected_height_px(None, np.array([0.0, -12.0])) is None
    assert expected_height_px(broadcast_camera, None) is None
    near = expected_height_px(broadcast_camera, np.array([0.0, -12.0]))
    far = expected_height_px(broadcast_camera, np.array([0.0, 12.0]))
    assert near > far > 0


def test_size_from_camera_without_calibration_falls_back_to_no_size_gate(broadcast_camera):
    cam = broadcast_camera
    dets = [[_det(cam, 1, 1.0, -12.0)] for _ in range(3)]
    r = select_players(dets, _calib(cam, 3, invalid={1}), size_from_camera=True)
    assert np.isfinite(r["near"].world[[0, 2]]).all() and np.isnan(r["near"].world[1]).all()


def test_iou_tracker_drops_a_track_after_max_age_and_respects_min_iou():
    t = IouTracker(min_iou=0.3, max_age=2)
    a = np.array([0.0, 0.0, 10.0, 20.0])
    first = t.update([a])
    t.update([])
    t.update([])  # two frames unseen: still alive
    assert t.update([a]) == first
    t.update([])
    t.update([])
    t.update([])  # three unseen: expired
    assert t.update([a]) != first
    t2 = IouTracker(min_iou=0.3, max_age=2)
    i0 = t2.update([a])
    assert t2.update([a + np.array([8.0, 0, 8.0, 0])]) != i0  # IoU 2/18 < 0.3: a new id


def test_a_ball_kid_is_not_taken_as_the_player_when_the_player_is_missed(broadcast_camera):
    cam = broadcast_camera
    n = 10
    dets = [[_det(cam, 1, 1.0, -12.0)] if i not in (4, 5) else [_det(cam, 9, 1.3, -12.0, frac=0.4)]
            for i in range(n)]
    r = select_players(dets, _calib(cam, n))
    assert np.isnan(r["near"].world[[4, 5]]).all() and np.isfinite(r["near"].world[[3, 6]]).all()
    assert r["near"].track_ids == [1]
