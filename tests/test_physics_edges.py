"""Ball flight physics at its edges (physics/flight.py): the drag-free parabola, the direction of drag, the bounce
convention, spin sign, units and ordering, degenerate fits, and recovering a known flight from noise-free
projections (the bounce, not the velocities: one camera without anchors leaves depth open). Values checked by
running the code (gaps found by a test review)."""

from __future__ import annotations

import numpy as np
import pytest

from courtvision.geometry.court import BALL_RADIUS
from courtvision.physics.flight import DRAG_K, G, fit_flight, flight_states, flight_states_batch

# Row: [x_b, y_b, t_b, vin(3), vout(3), spin_in, spin_out]
P0 = np.array([1.0, 5.0, 0.5, 1.0, 10.0, -8.0, 1.0, 9.0, 6.0, 0.0, 0.0])


def _states(P, t, k=DRAG_K):
    pos, vel = flight_states_batch(np.asarray(P, float)[None], np.asarray(t, float)[None], k=k)
    return pos[0], vel[0]


def test_without_drag_the_outgoing_leg_is_an_exact_parabola() -> None:
    P = np.array([1.0, 2.0, 0.5, 0, 0, -5, 3.0, 4.0, 10.0, 0, 0])
    t_apex = 0.5 + 10.0 / G
    pos, vel = _states(P, [t_apex, 1.5], k=0.0)
    assert vel[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert pos[0, 2] == pytest.approx(BALL_RADIUS + 10.0**2 / (2 * G), abs=1e-9)  # 5.1285 m
    assert pos[0, :2] == pytest.approx([1.0 + 3.0 * 10.0 / G, 2.0 + 4.0 * 10.0 / G], abs=1e-9)
    assert vel[1] == pytest.approx([3.0, 4.0, 10.0 - G * 1.0], abs=1e-9)
    assert pos[1, 2] == pytest.approx(BALL_RADIUS + 10.0 * 1.0 - 0.5 * G, abs=1e-9)


def test_without_drag_the_incoming_leg_is_a_free_fall_in_time_reversed_form() -> None:
    # Dropped from rest 0.5 s before the bounce (vin_z = -g*0.5 at the bounce) -> z(t_b - 0.5) = r + g/8.
    P = np.array([0.0, 0.0, 0.5, 0, 0, -G * 0.5, 0, 0, 1.0, 0, 0])
    pos, vel = _states(P, [0.0], k=0.0)
    assert vel[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert pos[0, 2] == pytest.approx(BALL_RADIUS + 0.5 * G * 0.25, abs=1e-9)


def test_drag_slows_the_ball_in_forward_time_on_both_legs() -> None:
    t = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    pos, vel = _states(P0, t)
    h = np.linalg.norm(vel[:, :2], axis=1)
    assert np.all(np.diff(h[:4]) < 0)  # before the bounce the horizontal speed falls towards v_in
    assert h[3] == pytest.approx(np.hypot(1.0, 9.0), abs=1e-9)  # t_b itself is the outgoing leg: v_out
    assert h[0] > np.hypot(1.0, 10.0)  # earlier = faster (drag undone going backward)
    assert np.all(np.diff(h[4:]) < 0)  # after the bounce it keeps falling
    _, v0 = _states(P0, t, k=0.0)
    assert np.all(np.linalg.norm(v0[:, :2], axis=1)[4:] == pytest.approx(np.hypot(1.0, 9.0)))
    assert h[-1] < np.hypot(1.0, 9.0)
    # drag reduces apex height versus vacuum
    tt = np.linspace(0.5, 1.6, 111)
    assert _states(P0, tt)[0][:, 2].max() < _states(P0, tt, k=0.0)[0][:, 2].max()


def test_state_at_the_bounce_instant_is_on_the_ground_with_the_given_velocity() -> None:
    pos, vel = _states(P0, [0.5, 0.5 - 1e-4, 0.5 + 1e-4])
    assert pos[0] == pytest.approx([1.0, 5.0, BALL_RADIUS], abs=1e-12)
    assert vel[0] == pytest.approx(P0[6:9], abs=1e-12)  # t == t_b belongs to the outgoing leg
    assert vel[1] == pytest.approx(P0[3:6], abs=2e-3)
    assert pos[1, 2] == pytest.approx(BALL_RADIUS, abs=2e-3) and pos[2, 2] == pytest.approx(BALL_RADIUS, abs=2e-3)
    # the model does not reflect: vertical velocity flips sign only through the two parameters
    assert vel[1, 2] < 0 < vel[2, 2]


def test_positive_spin_is_topspin_and_dips_the_ball_negative_lifts_it() -> None:
    def z(s):
        Q = P0.copy()
        Q[9] = Q[10] = s
        return _states(Q, [0.3, 0.8])[0][:, 2]

    assert z(0.5) == pytest.approx([1.36905951, 1.1454479], abs=1e-6)
    assert z(-0.5) == pytest.approx([1.57677234, 1.53625865], abs=1e-6)
    assert np.all(z(0.5) < z(0.0)) and np.all(z(0.0) < z(-0.5))


def test_time_order_of_the_query_and_short_parameter_vectors_do_not_change_the_states() -> None:
    a, _ = _states(P0, [0.2, 0.4, 0.6])
    b, _ = _states(P0, [0.6, 0.2, 0.4])
    assert b[1] == pytest.approx(a[0], abs=1e-12) and b[2] == pytest.approx(a[1], abs=1e-12)
    assert b[0] == pytest.approx(a[2], abs=1e-12)
    short, vs = flight_states(P0[:9], np.array([0.3, 0.8]))
    full, vf = flight_states(P0, np.array([0.3, 0.8]))
    assert short == pytest.approx(full) and vs == pytest.approx(vf)


def test_speed_unit_is_km_per_hour() -> None:
    from courtvision.physics.flight import FlightFit
    fit = FlightFit(P0, np.eye(11), 0.0, 10, np.zeros(1), np.zeros((1, 3)), True)
    _, v = flight_states(P0, np.array([0.5]))
    assert fit.speed_kmh_at(0.5) == pytest.approx(float(np.linalg.norm(v[0])) * 3.6)
    assert fit.speed_kmh_at(0.5) == pytest.approx(np.linalg.norm([1.0, 9.0, 6.0]) * 3.6)  # 39.6 km/h
    assert fit.bounce_time == 0.5 and fit.bounce_xy == pytest.approx([1.0, 5.0])


def test_fit_recovers_the_bounce_from_noise_free_projections_of_a_known_flight(broadcast_camera) -> None:
    t = np.arange(0.0, 1.0, 1 / 30)
    pos, _ = flight_states(P0, t)
    obs = broadcast_camera.project(pos)
    fit = fit_flight(t, obs, [broadcast_camera] * len(t), 0.5 + 1 / 30)
    assert fit.success
    assert np.linalg.norm(fit.bounce_xy - P0[:2]) < 0.15  # measured 0.10 m (monocular depth ambiguity, no noise)
    assert fit.bounce_time == pytest.approx(0.5, abs=0.02)  # measured 0.4958
    assert fit.rms_px < 1.0  # measured 0.58 (the spin prior, not noise, leaves the residual)
    assert np.all(fit.positions[:, 2] > 0)  # the fitted ball never goes below the ground
    reproj = broadcast_camera.project(fit.positions)
    assert np.max(np.linalg.norm(reproj - obs, axis=1)) < 3.0
    assert fit.params[5] <= 0 <= fit.params[8]  # incoming descends, outgoing rises (bounds)
    # a far-off bounce guess still lands on the same answer
    far = fit_flight(t, obs, [broadcast_camera] * len(t), 5.0)
    assert far.bounce_time == pytest.approx(0.5, abs=0.02)


def test_fit_degenerate_inputs_return_a_failed_nan_fit_instead_of_raising(broadcast_camera) -> None:
    cams = [broadcast_camera] * 10
    few = fit_flight(np.arange(7) / 30, np.full((7, 2), 500.0), cams[:7], 0.1)
    assert not few.success and few.message == "too few observations" and np.isnan(few.params).all()
    assert few.n_obs == 7
    same = fit_flight(np.full(10, 1.0), np.full((10, 2), 500.0), cams, 1.0)
    assert not same.success and same.message == "bounce window empty" and np.isnan(same.cov).all()
