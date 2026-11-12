from __future__ import annotations

import numpy as np
import pytest

from courtvision.calib.camera import apply_h, camera_from_homography, estimate_focal, fit_homography
from courtvision.geometry.court import (
    HALF_LENGTH,
    SINGLES_HALF_WIDTH,
    CourtModel,
    Half,
    half_of,
    rally_zone,
    service_box,
)


def test_keypoints_follow_itf_dimensions_on_line_centres() -> None:
    kp = CourtModel().keypoints()
    assert kp.shape == (14, 2)
    # Far baseline doubles corners are 10.97 m apart minus one sideline width (centre to centre).
    assert kp[1, 0] - kp[0, 0] == pytest.approx(10.97 - 0.05)
    # Baseline centre is half a 10 cm baseline inside the 11.885 m outer edge.
    assert kp[0, 1] == pytest.approx(HALF_LENGTH - 0.05)
    assert kp[12, 1] == pytest.approx(6.40 - 0.025)
    assert half_of(kp[0, 1]) == Half.FAR and half_of(kp[2, 1]) == Half.NEAR


def test_world_frame_is_right_handed() -> None:
    kp = CourtModel().keypoints()
    x_axis = kp[1] - kp[0]  # far-left -> far-right
    y_axis = kp[0] - kp[2]  # near -> far
    z = np.cross(np.append(x_axis, 0), np.append(y_axis, 0))
    assert z[2] > 0


def test_rally_zone_margins_use_outer_line_edges() -> None:
    z = rally_zone(Half.FAR)
    m, line = z.signed_margin(np.array([SINGLES_HALF_WIDTH - 0.01, 5.0]))
    assert m == pytest.approx(0.01) and line == "right singles sideline"
    m, line = z.signed_margin(np.array([0.0, HALF_LENGTH + 0.03]))
    assert m == pytest.approx(-0.03) and line == "far baseline"
    # The net side is open: a point near the net is not "close to a line".
    m, _ = z.signed_margin(np.array([0.0, 0.05]))
    assert m > 4.0


def test_service_box_is_cross_court_and_shares_centre_line() -> None:
    box = service_box(Half.FAR, server_x=+2.0)  # near server on the right -> far left box
    assert box.x0 == pytest.approx(-SINGLES_HALF_WIDTH) and box.x1 == pytest.approx(0.025)
    assert box.contains(np.array([0.02, 3.0]))  # touching the centre service line is in
    assert not box.contains(np.array([0.0, 6.5]))


def test_homography_and_camera_recovery(broadcast_camera) -> None:
    rng = np.random.default_rng(0)
    world = CourtModel().keypoints()
    px = broadcast_camera.project(np.column_stack([world, np.zeros(14)]))
    noisy = px + rng.normal(0, 0.7, px.shape)
    noisy[3] = [5000.0, 5000.0]  # gross outlier must be rejected
    noisy[7] = np.nan  # missing keypoint
    fit = fit_homography(noisy, world)
    assert fit is not None
    assert not fit.inliers[3] and not fit.inliers[7]
    assert fit.rmse_px < 2.0

    f = estimate_focal(fit.H, 960, 540)
    assert f == pytest.approx(broadcast_camera.focal, rel=0.05)

    cam = camera_from_homography(fit.H, (1920, 1080))
    assert cam is not None
    assert np.linalg.norm(cam.center - broadcast_camera.center) < 1.0

    # A ball centre 3.35 cm up, back-projected at the right height, lands where it started.
    ball = np.array([3.9, 9.0, 0.0335])
    pix = broadcast_camera.project(ball)
    est = cam.backproject_to_height(pix, 0.0335)
    assert np.linalg.norm(est[:2] - ball[:2]) < 0.05
    # Treating it as a ground point (plain homography) is biased away from the camera.
    naive = apply_h(np.linalg.inv(fit.H), pix)
    assert naive[1] - ball[1] > 0.03


def test_height_above_a_ground_point(broadcast_camera) -> None:
    # Serve contacts at both baselines: the ray above the server's feet recovers the contact height.
    for feet, z in (((-0.8, 12.0), 2.8), ((1.0, -11.7), 3.05)):
        pix = broadcast_camera.project(np.array([*feet, z]))
        assert broadcast_camera.height_above(pix, np.array(feet)) == pytest.approx(z, abs=1e-6)
