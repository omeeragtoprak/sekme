from __future__ import annotations

import numpy as np
import pytest

from courtvision.geometry.court import HALF_LENGTH, SINGLES_HALF_WIDTH, Half, rally_zone
from courtvision.officiating.line_call import call_bounce
from courtvision.physics.flight import fit_flight
from courtvision.sim.simulate import simulate_shot


def _fit_sim(cam, shot, rng):
    t_guess = shot.bounce_time + rng.uniform(-1 / 30, 1 / 30)
    return fit_flight(shot.times, shot.pixels, [cam] * len(shot.times), t_guess)


def test_flight_fit_recovers_bounce_in_ideal_conditions(broadcast_camera) -> None:
    rng = np.random.default_rng(3)
    shot = simulate_shot(broadcast_camera, np.array([1.0, -11.5, 1.0]), np.array([-2.5, 9.0]), 110,
                         spin_rpm=0, pixel_noise=0.2, rng=rng)
    fit = _fit_sim(broadcast_camera, shot, rng)
    assert fit.success
    assert np.linalg.norm(fit.bounce_xy - shot.bounce_xy) < 0.05
    assert fit.bounce_time == pytest.approx(shot.bounce_time, abs=0.004)
    assert fit.speed_kmh_at(shot.times[0]) == pytest.approx(110, rel=0.06)


def test_bounce_uncertainty_is_honest_under_unmodelled_spin(broadcast_camera) -> None:
    """Spin is simulated but not modelled; the reported covariance must still cover the truth."""
    rng = np.random.default_rng(11)
    errs, inside95 = [], 0
    n = 16
    for _ in range(n):
        far = rng.random() < 0.5
        start = np.array([rng.uniform(-3, 3), -11.0 if far else 11.0, rng.uniform(0.6, 1.4)])
        tgt = np.array([rng.uniform(-3.8, 3.8), rng.uniform(5, 11.5) * (1 if far else -1)])
        shot = simulate_shot(broadcast_camera, start, tgt, rng.uniform(85, 140), spin_rpm=rng.uniform(0, 2600),
                             pixel_noise=1.5, rng=rng)
        fit = _fit_sim(broadcast_camera, shot, rng)
        e = fit.bounce_xy - shot.bounce_xy
        errs.append(np.linalg.norm(e))
        inside95 += float(e @ np.linalg.solve(fit.bounce_cov, e)) < 5.991  # chi2(2) 95%
    assert np.median(errs) < 0.15, errs
    assert inside95 / n >= 0.8, (inside95, errs)


def test_line_call_verdicts() -> None:
    zone = rally_zone(Half.FAR)
    tight = np.diag([0.03**2, 0.05**2])
    clear_in = call_bounce(np.array([0.0, 8.0]), tight, zone)
    assert clear_in.verdict == "IN" and clear_in.p_in > 0.99

    clear_out = call_bounce(np.array([0.0, HALF_LENGTH + 0.40]), tight, zone)
    assert clear_out.verdict == "OUT" and clear_out.line == "far baseline"
    assert clear_out.margin_m == pytest.approx(-0.38, abs=1e-6)

    close = call_bounce(np.array([SINGLES_HALF_WIDTH + 0.03, 6.0]), tight, zone)
    assert close.verdict == "REVIEW" and 0.05 < close.p_in < 0.95
    assert close.line == "right singles sideline" and close.sigma_m == pytest.approx(0.03)


def test_a_pinned_serve_contact_height_holds_where_the_soft_band_lets_the_fit_drift(broadcast_camera) -> None:
    """Candidate --pin-serve-height: the anchor's height band is enforced, not just preferred.

    The simulated serve is struck at 2.95 m but the anchor says 2.5 m. With the default band (+-0.3 m, 0.1 m
    hinge) the pixels pull the fit past the band's edge; pinned (+-0.05 m, 0.01 m hinge) it stays. The pin is
    only as good as the measured height: here it is wrong on purpose, and the speed suffers (~180 vs 190).
    """
    from courtvision.physics.flight import Anchor
    rng = np.random.default_rng(1)
    shot = simulate_shot(broadcast_camera, np.array([0.5, -11.8, 2.95]), np.array([-1.0, 5.5]), 190, spin_rpm=1500,
                         fps=60.0, pixel_noise=1.0, rng=rng)
    z = {}
    for name, (lo, hi, scale) in {"band": (2.2, 2.8, 0.1), "pin": (2.45, 2.55, 0.01)}.items():
        a = Anchor(t=shot.times[0], xy=np.array([0.5, -11.8]), t_sigma=1 / 60, sigma_x=0.5, sigma_y=0.5,
                   z_min=lo, z_max=hi, z_scale=scale)
        fit = fit_flight(shot.times, shot.pixels, [broadcast_camera] * len(shot.times), shot.bounce_time,
                         pixel_sigma=2.5, anchors=(a,))
        assert fit.success
        z[name] = fit.state_at(float(fit.anchor_times[0]))[0][2]
    assert z["band"] > 2.8 and abs(z["pin"] - 2.5) < 0.06


def test_the_fast_acceleration_equals_the_norm_and_cross_product_form_exactly() -> None:
    from courtvision.physics.flight import DRAG_K, G, _accel

    def reference(v, k, spin):  # the form used before the speed-up
        a = -k * np.linalg.norm(v, axis=1, keepdims=True) * v
        a[:, 2] -= G
        vh = np.hypot(v[:, 0], v[:, 1])
        ok = vh > 1e-6
        if np.any(spin[ok] != 0):
            inv = np.where(ok, 1.0 / np.where(ok, vh, 1.0), 0.0)
            e = np.stack([-v[:, 1] * inv, v[:, 0] * inv, np.zeros(len(v))], axis=1)
            a += spin[:, None] * np.cross(e, v)
        return a

    rng = np.random.default_rng(7)
    for _ in range(500):
        n = int(rng.integers(1, 30))
        v = rng.normal(0, 30, (n, 3))
        v[rng.random(n) < 0.1, :2] = 0.0  # vertical balls: no horizontal direction for the Magnus term
        spin = np.where(rng.random(n) < 0.4, 0.0, rng.normal(0, 0.3, n))
        assert np.array_equal(_accel(v, DRAG_K, spin), reference(v, DRAG_K, spin))
