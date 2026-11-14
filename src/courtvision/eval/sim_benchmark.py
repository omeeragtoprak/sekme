"""Simulation benchmark: how accurate and how honest are bounce estimates for a camera setup?

Generates random rally shots (including effects the fitter only
approximates: speed-dependent bounce, spin decay at the bounce), observes
them through a camera at a given fps / pixel noise, fits them and reports
error percentiles, covariance coverage (how often the truth lies inside the
reported 95% ellipse) and line-call outcomes against ground truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from courtvision.calib.camera import Camera
from courtvision.geometry.court import HALF_LENGTH, SINGLES_HALF_WIDTH, Half, rally_zone
from courtvision.officiating.line_call import call_bounce
from courtvision.physics.flight import fit_flight
from courtvision.sim.simulate import simulate_shot


@dataclass
class BenchmarkReport:
    n_shots: int
    fps: float
    pixel_noise: float
    median_error_cm: float
    p90_error_cm: float
    coverage_95: float  # fraction of truths inside the reported 95% ellipse
    median_sigma_cm: float
    decisive_rate: float  # fraction of calls that were IN/OUT (not REVIEW)
    decisive_accuracy: float  # accuracy of decisive calls against the truth
    wrong_decisive: int
    near_line_shots: int  # truths within 10 cm of a line

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_benchmark(camera: Camera, n_shots: int = 60, fps: float = 30.0, pixel_noise: float = 1.5,
                  seed: int = 7, near_line_fraction: float = 0.5) -> BenchmarkReport:
    rng = np.random.default_rng(seed)
    errs: list[float] = []
    sig: list[float] = []
    cover = decisive = correct = near = 0
    for i in range(n_shots):
        to_far = rng.random() < 0.5
        sgn = 1.0 if to_far else -1.0
        start = np.array([rng.uniform(-3, 3), -11.0 * sgn, rng.uniform(0.6, 1.4)])
        if rng.random() < near_line_fraction:  # stress the decision boundary
            if rng.random() < 0.5:
                tgt = np.array([rng.choice([-1, 1]) * (SINGLES_HALF_WIDTH + rng.uniform(-0.15, 0.15)),
                                sgn * rng.uniform(3, 11)])
            else:
                tgt = np.array([rng.uniform(-3.5, 3.5), sgn * (HALF_LENGTH + rng.uniform(-0.2, 0.2))])
        else:
            tgt = np.array([rng.uniform(-3.8, 3.8), sgn * rng.uniform(4, 11.4)])
        shot = simulate_shot(camera, start, tgt, rng.uniform(80, 150), spin_rpm=rng.uniform(-800, 3000),
                             fps=fps, pixel_noise=pixel_noise, rng=rng)
        t_guess = shot.bounce_time + rng.uniform(-1 / fps, 1 / fps)
        fit = fit_flight(shot.times, shot.pixels, [camera] * len(shot.times), t_guess, pixel_sigma=pixel_noise)
        if not fit.success or not np.all(np.isfinite(fit.bounce_cov)):
            continue
        e = fit.bounce_xy - shot.bounce_xy
        errs.append(float(np.linalg.norm(e)))
        sig.append(float(np.sqrt(np.max(np.linalg.eigvalsh(fit.bounce_cov)))))
        cover += int(float(e @ np.linalg.solve(fit.bounce_cov, e)) < 5.991)
        zone = rally_zone(Half.FAR if to_far else Half.NEAR)
        truth_in = bool(zone.contains(shot.bounce_xy[None], grow=0.02)[0])
        margin, _ = zone.signed_margin(shot.bounce_xy)
        near += int(abs(margin + 0.02) < 0.10)
        call = call_bounce(fit.bounce_xy, fit.bounce_cov, zone, seed=i)
        if call.verdict != "REVIEW":
            decisive += 1
            correct += int((call.verdict == "IN") == truth_in)
    n = len(errs)
    if n == 0:
        raise ValueError("no simulated shot was fitted: nothing to report")
    return BenchmarkReport(
        n_shots=n, fps=fps, pixel_noise=pixel_noise,
        median_error_cm=100 * float(np.median(errs)), p90_error_cm=100 * float(np.percentile(errs, 90)),
        coverage_95=cover / n, median_sigma_cm=100 * float(np.median(sig)),
        decisive_rate=decisive / n, decisive_accuracy=correct / max(decisive, 1),
        wrong_decisive=decisive - correct, near_line_shots=near,
    )
