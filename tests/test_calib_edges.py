"""Camera geometry at its edges (calib/camera.py, geometry/court.py): exact recovery of a known camera, which side
it is on and its height's sign, rejected cameras, degenerate point sets, apply_h edge cases, back-projection and the
court's ITF dimensions. Values checked by running the code (gaps found by a test review)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import look_at_camera
from courtvision.calib.camera import apply_h, camera_from_homography, fit_homography
from courtvision.geometry import court as court_geom
from courtvision.geometry.court import CourtModel

SIZE = (1920, 1080)  # the shared broadcast_camera fixture's image size


def _fit(cam):
    w = CourtModel().keypoints()
    px = cam.project(np.column_stack([w, np.zeros(len(w))]))
    return w, px, fit_homography(px, w)


def test_noise_free_recovery_is_exact(broadcast_camera) -> None:
    w, px, fit = _fit(broadcast_camera)
    assert fit.n_inliers == 14 and fit.rmse_px < 1e-3
    cam = camera_from_homography(fit.H, SIZE)
    assert cam.focal == pytest.approx(1750.0, rel=1e-4)
    assert np.allclose(cam.center, broadcast_camera.center, atol=1e-3)
    assert np.allclose(cam.R, broadcast_camera.R, atol=1e-5)
    assert np.allclose(cam.t, broadcast_camera.t, atol=1e-3)
    assert np.allclose(broadcast_camera.ground_homography(), fit.H, atol=1e-3)


def test_camera_side_and_height_sign(broadcast_camera) -> None:
    _, _, fit = _fit(broadcast_camera)
    cam = camera_from_homography(fit.H, SIZE)
    assert cam.center[1] < -20 and cam.center[2] > 0  # near side, above the ground
    assert cam.depth(np.zeros(3)) > 0
    # H is defined up to sign: -H must give the same camera
    neg = camera_from_homography(-fit.H, SIZE)
    assert np.allclose(neg.center, cam.center, atol=1e-6)
    # far-side camera in a different image size is recovered on its own side
    far = look_at_camera(np.array([3.0, 28.0, 9.0]), np.array([0, -1.5, 0.0]), 1500.0, (1280, 720))
    got = camera_from_homography(far.ground_homography(), (1280, 720))
    assert got.focal == pytest.approx(1500.0, rel=1e-6)
    assert np.allclose(got.center, [3.0, 28.0, 9.0], atol=1e-6)


def test_camera_below_plane_or_bad_focal_is_rejected(broadcast_camera) -> None:
    below = look_at_camera(np.array([0.0, -28.0, -11.0]), np.array([0, 1.5, 0.0]), 1750.0, SIZE)
    assert camera_from_homography(below.ground_homography(), SIZE) is None
    _, _, fit = _fit(broadcast_camera)
    assert camera_from_homography(fit.H, SIZE, focal=100.0) is None  # < 0.2 * width
    assert camera_from_homography(fit.H, SIZE, focal=1e5) is None  # > 20 * width
    assert camera_from_homography(fit.H, SIZE, focal=1750.0).focal == 1750.0
    fronto = np.array([[100.0, 0, 960], [0, 100, 540], [0, 0, 1]])
    assert camera_from_homography(fronto, SIZE) is None


def test_fit_homography_degenerate_inputs(broadcast_camera) -> None:
    w, px, _ = _fit(broadcast_camera)
    assert fit_homography(px[:3], w[:3]) is None  # fewer than 4 points
    few = px.copy()
    few[3:] = np.nan  # only 3 finite rows
    assert fit_homography(few, w) is None
    col = np.array([[0, 0], [1, 0], [2, 0], [3, 0.0]])
    assert fit_homography(col * 100, col) is None  # collinear
    col5 = np.array([[-5, 0], [0, 0], [5, 0], [3, 0], [4, 0.0]])
    assert fit_homography(col5 * 100, col5) is None


def test_apply_h_edge_cases_and_round_trip(broadcast_camera) -> None:
    assert apply_h(np.eye(3), np.zeros((0, 2))).shape == (0, 2)
    assert np.isnan(apply_h(np.eye(3), np.array([[np.nan, 1.0]]))).all()
    _, _, fit = _fit(broadcast_camera)
    pts = np.array([[1.0, 2.0], [-3.0, -7.5], [0.0, 0.0]])
    back = apply_h(np.linalg.inv(fit.H), apply_h(fit.H, pts))
    assert np.allclose(back, pts, atol=1e-9)
    assert apply_h(fit.H, pts[0]).shape == (2,)  # single point
    assert apply_h(fit.H, pts.reshape(3, 1, 2)).shape == (3, 1, 2)  # batched
    with np.errstate(divide="ignore"):  # point on the horizon line (w == 0) -> inf, not an exception
        assert np.isinf(apply_h(np.diag([1.0, 1.0, 0.0]), np.array([[1.0, 2.0]]))).all()


def test_projection_backprojection_round_trip(broadcast_camera) -> None:
    p = np.array([[2.0, 3.0, 1.5], [-3.0, -5.0, 0.2], [0.0, 0.0, 0.0]])
    out = broadcast_camera.backproject_to_height(broadcast_camera.project(p), p[:, 2])
    assert np.allclose(out, p, atol=1e-9)
    _, d = broadcast_camera.ray(np.array([[100.0, 200.0], [960.0, 540.0]]))
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)
    assert d[1, 1] > 0 and d[1, 2] < 0  # principal ray looks away from the camera and downwards
    # plane below ground: still intersects in front of the camera
    below = broadcast_camera.backproject_to_height(np.array([960.0, 200.0]), -1.0)
    assert below[2] == pytest.approx(-1.0)


def test_court_dimensions_match_itf() -> None:
    assert pytest.approx(23.77) == 2 * court_geom.HALF_LENGTH
    assert pytest.approx(10.97) == 2 * court_geom.DOUBLES_HALF_WIDTH
    assert pytest.approx(8.23) == 2 * court_geom.SINGLES_HALF_WIDTH
    assert court_geom.SERVICE_LINE_Y == 6.40
    assert pytest.approx(5.485 + 0.914) == court_geom.NET_POST_X
    kp = CourtModel().keypoints()
    assert np.allclose(kp[:, 1][[0, 1, 4, 6]], CourtModel().baseline_c)
    assert np.allclose(kp[8:10, 1], CourtModel().service_c) and np.allclose(kp[10:12, 1], -CourtModel().service_c)
    assert np.allclose(kp[:, 0][[12, 13]], 0.0)
    assert court_geom.half_of(0.0) == court_geom.Half.NEAR  # tie rule at the net

