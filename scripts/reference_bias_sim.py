"""How far is the TrackNet-style reference contact from the true bounce, on simulated shots with exact pixels?

    .venv/bin/python scripts/reference_bias_sim.py      (~10 min on CPU; 300 shots)

Camera as recovered on TrackNet game 8 (centre (0, -43.8, 10.9) m, focal 2657 px, 1280x720). Labels are the
projected true ball centres (no noise) and the bounce frame is the frame nearest the true bounce. The reference
is built exactly as scripts/eval_tracknet.py builds it (contacts() + world_from_px + ball_centre_ground); the
system is fit_flight on the same exact pixels. Court y points away from the camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.calib.camera import Camera  # noqa: E402
from courtvision.eval.labels import world_from_px  # noqa: E402
from courtvision.eval.tracknet import Labels, ball_centre_ground, contacts  # noqa: E402
from courtvision.physics.flight import fit_flight  # noqa: E402
from courtvision.sim.simulate import simulate_shot  # noqa: E402

CENTRE = np.array([0.0, -43.8, 10.9])


def tracknet_camera() -> Camera:
    fwd = np.array([0.0, 0.5, 0.0]) - CENTRE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rot = np.stack([right, np.cross(fwd, right), fwd])
    k = np.array([[2657.0, 0, 640], [0, 2657.0, 360], [0, 0, 1.0]])
    return Camera(k, rot, -rot @ CENTRE)


def main(n_shots: int = 300) -> None:
    cam = tracknet_camera()
    h = cam.ground_homography()
    rng = np.random.default_rng(1)
    rows: dict[str, list] = {"far": [], "near": []}
    for _ in range(n_shots):
        to_far = rng.random() < 0.5
        sgn = 1.0 if to_far else -1.0
        start = np.array([rng.uniform(-3, 3), -11.0 * sgn, rng.uniform(0.6, 1.4)])
        tgt = np.array([rng.uniform(-3.8, 3.8), sgn * rng.uniform(4, 11.4)])
        shot = simulate_shot(cam, start, tgt, rng.uniform(80, 150), spin_rpm=rng.uniform(-800, 3000), fps=30.0,
                             pixel_noise=0.0, rng=rng)
        px = cam.project(shot.positions)
        if not (np.all((px[:, 0] > 0) & (px[:, 0] < 1280)) and np.all((px[:, 1] > 0) & (px[:, 1] < 720))):
            continue
        status = np.zeros(len(px), int)
        status[int(np.argmin(np.abs(shot.times - shot.bounce_time)))] = 2
        ref = [c for c in contacts(Labels(np.ones(len(px), bool), px.copy(), status)) if c.from_tracks]
        if not ref:
            continue
        e_ref = ball_centre_ground(np.asarray(world_from_px(h, tuple(ref[0].px))), CENTRE) - shot.bounce_xy
        fit = fit_flight(shot.times, px, [cam] * len(px), shot.bounce_time, pixel_sigma=1.0)
        e_sys = fit.bounce_xy - shot.bounce_xy if fit.success else np.array([np.nan, np.nan])
        rows["far" if to_far else "near"].append((*e_ref, *e_sys))
    for half, r in rows.items():
        a = np.array(r) * 100
        print(f"{half} half: n={len(a)}  reference - truth: median dx {np.nanmedian(a[:, 0]):+.1f} cm, "
              f"dy {np.nanmedian(a[:, 1]):+.1f} cm  |  system - truth: median dy {np.nanmedian(a[:, 3]):+.1f} cm, "
              f"median |error| {np.nanmedian(np.hypot(a[:, 2], a[:, 3])):.1f} cm")


if __name__ == "__main__":
    main()
