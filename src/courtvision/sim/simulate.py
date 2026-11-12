"""Physics simulator producing ground-truth tennis shots seen by a camera.

Used for tests and for accuracy studies (camera placement advisor). The
simulator deliberately includes effects the fitter does not model —
Magnus lift from spin and a velocity-dependent bounce — so measured errors
are honest rather than self-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.calib.camera import Camera
from courtvision.geometry.court import BALL_RADIUS
from courtvision.physics.flight import DRAG_K, G

# Magnus force F = C (w x v); C chosen so 2500 rpm topspin at 30 m/s gives ~8 m/s^2
# (lift coefficient ~0.25, Goodwill & Haake / Cross measurements).
MAGNUS_COEFF = 6.1e-5  # kg m
BALL_MASS = 0.0577


@dataclass
class SimShot:
    times: np.ndarray  # frame timestamps (s)
    positions: np.ndarray  # true 3D ball centre (N, 3)
    pixels: np.ndarray  # noisy observations (N, 2)
    bounce_xy: np.ndarray  # true ground contact point
    bounce_time: float
    hit_time: float
    hit_speed_kmh: float


def _step(p: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    speed = np.linalg.norm(v)
    a = -DRAG_K * speed * v + MAGNUS_COEFF * np.cross(w, v) / BALL_MASS
    a[2] -= G
    return p + v * dt + 0.5 * a * dt * dt, v + a * dt



def simulate_shot(camera: Camera, start: np.ndarray, target: np.ndarray, speed_kmh: float,
                  spin_rpm: float = 0.0, fps: float = 30.0, pixel_noise: float = 1.5,
                  duration_after_bounce: float = 0.45, rng: np.random.Generator | None = None,
                  phase: float | None = None) -> SimShot:
    """Hit from ``start`` (3D) towards ground ``target`` (2D) at ``speed_kmh`` with topspin ``spin_rpm``.

    The launch elevation is solved by shooting so the ball lands on ``target``.
    """
    rng = rng or np.random.default_rng()
    direction = np.append(target - start[:2], 0.0)
    horiz = direction / np.linalg.norm(direction)
    axis = np.cross(horiz, [0.0, 0.0, 1.0])  # topspin axis
    w = -axis * spin_rpm * 2 * np.pi / 60.0
    speed = speed_kmh / 3.6
    dt = 1e-4

    def fly(elev: float) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        v = speed * (np.cos(elev) * horiz + np.sin(elev) * np.array([0.0, 0.0, 1.0]))
        p = start.astype(float).copy()
        ps, vs = [p], [v]
        while p[2] > BALL_RADIUS and len(ps) < 60000:
            p, v = _step(p, v, w, dt)
            ps.append(p)
            vs.append(v)
        return p, ps, vs

    lo, hi = np.radians(-15), np.radians(35)
    for _ in range(40):  # bisection on landing distance
        mid = 0.5 * (lo + hi)
        p_land, _, _ = fly(mid)
        if np.dot(p_land[:2] - start[:2], horiz[:2]) < np.dot(target - start[:2], horiz[:2]):
            lo = mid
        else:
            hi = mid
    _, ps, vs = fly(0.5 * (lo + hi))
    pre = np.array(ps)
    t_bounce = (len(ps) - 1) * dt
    # Bounce: COR on vertical velocity depends on impact speed; horizontal friction loss.
    v_in = vs[-1]
    cor = 0.78 - 0.002 * abs(v_in[2])
    v_out = np.array([v_in[0] * 0.65, v_in[1] * 0.65, -v_in[2] * cor])
    p = pre[-1].copy()
    p[2] = BALL_RADIUS
    v = v_out
    post = [p]
    n_post = int(duration_after_bounce / dt)
    for _ in range(n_post):
        p, v = _step(p, v, w * 0.5, dt)
        post.append(p)
    traj = np.concatenate([pre, np.array(post[1:])])
    t_all = np.arange(len(traj)) * dt
    off = phase if phase is not None else rng.uniform(0, 1 / fps)
    frame_t = np.arange(off, t_all[-1], 1 / fps)
    idx = np.clip(np.round(frame_t / dt).astype(int), 0, len(traj) - 1)
    pos = traj[idx]
    pix = camera.project(pos) + rng.normal(0, pixel_noise, (len(pos), 2))
    return SimShot(frame_t, pos, pix, pre[-1][:2].copy(), t_bounce, 0.0, speed_kmh)
