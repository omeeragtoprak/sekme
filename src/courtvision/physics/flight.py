"""Ballistic flight model and a bounce-anchored 3D fit to 2D ball tracks.

Between two racket contacts a tennis ball flies, bounces once (normally)
and flies again. We parametrise that flight by the bounce itself —
ground position (x_b, y_b), bounce time t_b, incoming velocity v_in and
outgoing velocity v_out — integrate gravity + quadratic drag backward and
forward from the bounce, project through the per-frame calibrated camera
and minimise reprojection error. The bounce-point covariance falls out of
the Jacobian, which is what the line-call module needs.

Spin is modelled with one Magnus parameter per flight phase (s_in, s_out),
acting perpendicular to the velocity in its vertical plane: positive is
topspin (dips the ball), negative is slice. Spin changes at the bounce, so
the two phases get separate parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from courtvision.calib.camera import Camera
from courtvision.geometry.court import BALL_RADIUS

G = 9.81
AIR_DENSITY = 1.20
DRAG_COEFF = 0.55
BALL_MASS = 0.0577
DRAG_K = 0.5 * AIR_DENSITY * DRAG_COEFF * np.pi * BALL_RADIUS**2 / BALL_MASS  # 1/m


def _accel(v: np.ndarray, k: float, spin: np.ndarray) -> np.ndarray:
    """Acceleration for a batch of velocities (B, 3) with per-member Magnus parameter ``spin`` (B,)."""
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    a = -k * speed * v
    a[:, 2] -= G
    vh = np.hypot(v[:, 0], v[:, 1])
    ok = vh > 1e-6
    if np.any(spin[ok] != 0):
        inv = np.where(ok, 1.0 / np.where(ok, vh, 1.0), 0.0)
        e = np.stack([-v[:, 1] * inv, v[:, 0] * inv, np.zeros(len(v))], axis=1)  # z x v_h, horizontal
        a += spin[:, None] * np.cross(e, v)
    return a


def integrate(p0: np.ndarray, v0: np.ndarray, rel: np.ndarray, k: float = DRAG_K, substeps: int = 2,
              spin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Batched RK4. ``p0``/``v0`` are (B, 3); ``rel`` is (B, N) target times relative to the start,
    all on one side of 0 and ordered by increasing |t|. Returns (B, N, 3) positions and velocities."""
    B, N = rel.shape
    sp = np.zeros(B) if spin is None else spin
    pos = np.empty((B, N, 3))
    vel = np.empty((B, N, 3))
    p, v, t = p0.astype(float).copy(), v0.astype(float).copy(), np.zeros(B)
    for i in range(N):
        span = rel[:, i] - t
        n = max(1, int(np.ceil(np.max(np.abs(span)) * 30 * substeps)))
        h = (span / n)[:, None]
        for _ in range(n):
            k1 = _accel(v, k, sp)
            k2 = _accel(v + 0.5 * h * k1, k, sp)
            k3 = _accel(v + 0.5 * h * k2, k, sp)
            k4 = _accel(v + h * k3, k, sp)
            p = p + h / 6 * (6 * v + h * (k1 + k2 + k3))
            v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # Drag integrated backward in time is anti-damping; keep wild optimiser trials finite.
            np.clip(v, -500.0, 500.0, out=v)
            np.clip(p, -1e4, 1e4, out=p)
        t = rel[:, i]
        pos[:, i], vel[:, i] = p, v
    return pos, vel


def flight_states_batch(P: np.ndarray, times: np.ndarray, k: float = DRAG_K) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities (B, M, 3) for parameter rows ``P`` (B, >=11) at ``times`` (M,) or (B, M).

    Row layout: [x_b, y_b, t_b, vin_x, vin_y, vin_z, vout_x, vout_y, vout_z, spin_in, spin_out].
    The pre/post-bounce split and ordering follow row 0 (other rows are small perturbations of it).
    """
    P = np.atleast_2d(P)
    B = len(P)
    T = np.broadcast_to(np.asarray(times, dtype=float), (B, np.shape(times)[-1]))
    rel = T - P[:, 2:3]
    pb = np.column_stack([P[:, 0], P[:, 1], np.full(B, BALL_RADIUS)])
    M = rel.shape[1]
    pos = np.empty((B, M, 3))
    vel = np.empty((B, M, 3))
    pre = rel[0] < 0
    if pre.any():
        idx = np.flatnonzero(pre)[np.argsort(-rel[0, pre])]  # nearest to the bounce first, going backward
        pos[:, idx], vel[:, idx] = integrate(pb, P[:, 3:6], rel[:, idx], k, spin=P[:, 9])
    if (~pre).any():
        idx = np.flatnonzero(~pre)[np.argsort(rel[0, ~pre])]
        pos[:, idx], vel[:, idx] = integrate(pb, P[:, 6:9], rel[:, idx], k, spin=P[:, 10])
    return pos, vel


def flight_states(params: np.ndarray, times: np.ndarray, k: float = DRAG_K) -> tuple[np.ndarray, np.ndarray]:
    """Single-parameter-set convenience wrapper: (M, 3) positions and velocities."""
    p = np.asarray(params, dtype=float)
    if len(p) < 11:
        p = np.concatenate([p, np.zeros(11 - len(p))])
    pos, vel = flight_states_batch(p[None, :11], np.asarray(times, dtype=float))
    return pos[0], vel[0]


@dataclass
class FlightFit:
    params: np.ndarray  # see flight_states (11 values)
    cov: np.ndarray  # (11, 11) parameter covariance
    rms_px: float
    n_obs: int
    times: np.ndarray
    positions: np.ndarray  # fitted 3D positions at the observation times
    success: bool
    message: str = ""
    anchor_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # fitted contact times

    @property
    def bounce_xy(self) -> np.ndarray:
        return self.params[:2]

    @property
    def bounce_cov(self) -> np.ndarray:
        return self.cov[:2, :2]

    @property
    def bounce_time(self) -> float:
        return float(self.params[2])

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        p, v = flight_states(self.params, np.array([t]))
        return p[0], v[0]

    def speed_kmh_at(self, t: float) -> float:
        return float(np.linalg.norm(self.state_at(t)[1]) * 3.6)


@dataclass(frozen=True)
class Anchor:
    """Soft prior: at time ``t`` the ball is at the racket of a player standing at ``xy``.

    A single far camera leaves the 3D path ambiguous along the viewing
    direction; the players' feet are known exactly on the ground plane, and
    at contact the ball is within reach of them at racket height.
    """

    t: float  # image-based estimate of the contact time; refined by the fit
    xy: np.ndarray
    t_sigma: float = 2 / 30  # the image-y extremum is biased by up to ~1.5 frames
    sigma_x: float = 1.0
    sigma_y: float = 1.3  # depth: reach plus contact in front of/behind the body
    z_min: float = 0.25
    z_max: float = 3.3
    z_scale: float = 0.1  # metres outside [z_min, z_max] that count as one sigma


SPIN_PRIOR_SIGMA = 0.5


def _project(KR: np.ndarray, Kt: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Project (B, N, 3) world points with per-observation cameras KR (N, 3, 3), Kt (N, 3) -> (B, N, 2)."""
    uvw = np.einsum("nij,bnj->bni", KR, pos) + Kt
    return uvw[..., :2] / uvw[..., 2:3]


def _residuals_batch(P: np.ndarray, times: np.ndarray, obs: np.ndarray, KR: np.ndarray, Kt: np.ndarray,
                     anchors: tuple[Anchor, ...], pixel_sigma: float) -> np.ndarray:
    """Residual rows (B, m): pixel residuals, then whitened priors.

    Each row of ``P`` holds 11 flight values followed by one contact time per anchor.
    """
    B, na = len(P), len(anchors)
    t_all = np.concatenate([np.broadcast_to(times, (B, len(times))), P[:, 11:11 + na]], axis=1)
    pos, _ = flight_states_batch(P[:, :11], t_all)
    res = (_project(KR, Kt, pos[:, :len(times)]) - obs).reshape(B, -1)
    extra = []
    for j, a in enumerate(anchors):
        pa = pos[:, len(times) + j]
        ta = P[:, 11 + j]
        extra += [(ta - a.t) / a.t_sigma, (pa[:, 0] - a.xy[0]) / a.sigma_x, (pa[:, 1] - a.xy[1]) / a.sigma_y,
                  (np.maximum(a.z_min - pa[:, 2], 0.0) + np.maximum(pa[:, 2] - a.z_max, 0.0)) / a.z_scale]
    extra += [P[:, 9] / SPIN_PRIOR_SIGMA, P[:, 10] / SPIN_PRIOR_SIGMA]
    return np.concatenate([res, np.stack(extra, axis=1) * pixel_sigma], axis=1)


class _Problem:
    """Residual function and a batched forward-difference Jacobian for scipy's least_squares."""

    def __init__(self, times: np.ndarray, obs: np.ndarray, cams: list[Camera], anchors: tuple[Anchor, ...],
                 pixel_sigma: float, ub: np.ndarray) -> None:
        self.times, self.obs, self.anchors, self.sigma, self.ub = times, obs, anchors, pixel_sigma, ub
        self.KR = np.stack([c.K @ c.R for c in cams])
        self.Kt = np.stack([c.K @ c.t for c in cams])

    def fun(self, x: np.ndarray) -> np.ndarray:
        return _residuals_batch(x[None], self.times, self.obs, self.KR, self.Kt, self.anchors, self.sigma)[0]

    def jac(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-6 * np.maximum(1.0, np.abs(x))
        eps = np.where(x + eps > self.ub, -eps, eps)
        P = np.vstack([x, x + np.diag(eps)])
        R = _residuals_batch(P, self.times, self.obs, self.KR, self.Kt, self.anchors, self.sigma)
        return ((R[1:] - R[0]) / eps[:, None]).T


def initial_guess(times: np.ndarray, obs: np.ndarray, cams: list[Camera], t_bounce: float) -> np.ndarray:
    """Seed: bounce at the ground back-projection of the observation nearest ``t_bounce``;
    horizontal velocities from back-projections of the first/last observations at 1 m."""
    j = int(np.argmin(np.abs(times - t_bounce)))
    pb = cams[j].backproject_to_height(obs[j], BALL_RADIUS)
    first = cams[0].backproject_to_height(obs[0], 1.0)
    last = cams[-1].backproject_to_height(obs[-1], 1.0)
    dt_in = max(t_bounce - times[0], 0.1)
    dt_out = max(times[-1] - t_bounce, 0.1)
    vin = (pb - first) / dt_in
    vin[2] = -(1.0 + 0.5 * G * dt_in**2) / dt_in
    vout = (last - pb) / dt_out
    vout[2] = (1.0 + 0.5 * G * dt_out**2) / dt_out
    return np.array([pb[0], pb[1], t_bounce, *vin, *vout, 0.0, 0.0])


def fit_flight(times: np.ndarray, obs: np.ndarray, cams: list[Camera], t_bounce_guess: float,
               pixel_sigma: float = 1.5, anchors: tuple[Anchor, ...] = ()) -> FlightFit:
    """Two-stage bounce-anchored flight fit; needs >= 8 observations with calibrated cameras.

    Stage 1 is robust (soft-L1) and only serves to flag pixel outliers
    (wrong ball candidates, racket/body occlusions). Stage 2 is a plain
    Gaussian least-squares on the inliers with the priors (anchors, spin)
    whitened by the estimated pixel noise, so priors keep their intended
    weight and the covariance sigma^2 (J^T J)^-1 is meaningful. The noise
    estimate is floored at ``pixel_sigma``: a lucky tight fit on few points
    cannot claim more precision than the detector has.
    """
    nan = FlightFit(np.full(11, np.nan), np.full((11, 11), np.nan), np.nan, len(times), times,
                    np.full((len(times), 3), np.nan), False)
    if len(times) < 8:
        nan.message = "too few observations"
        return nan
    x0 = initial_guess(times, obs, cams, t_bounce_guess)
    dt = float(np.median(np.diff(times)))
    # Physical envelope: bounce speeds stay well under 200 km/h; |spin| < 1/s covers heavy topspin/slice.
    lb = np.array([-20, -20, times[0] + 0.5 * dt, -55, -55, -40, -50, -50, 0, -1.0, -1.0])
    ub = np.array([20, 20, times[-1] - 0.5 * dt, 55, 55, 0, 50, 50, 30, 1.0, 1.0])
    if lb[2] >= ub[2]:
        nan.message = "bounce window empty"
        return nan
    # Contact times: before the first / after the last observation used.
    for a in anchors:
        if a.t <= times[0]:
            lb = np.append(lb, a.t - 4 * dt)
            ub = np.append(ub, times[0] - 0.25 * dt)
        else:
            lb = np.append(lb, times[-1] + 0.25 * dt)
            ub = np.append(ub, a.t + 4 * dt)
        x0 = np.append(x0, np.clip(a.t, lb[-1] + 1e-6, ub[-1] - 1e-6))
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    n = len(times)
    prob = _Problem(times, obs, cams, anchors, pixel_sigma, ub)
    s1 = least_squares(prob.fun, x0, jac=prob.jac, bounds=(lb, ub), loss="soft_l1", f_scale=3.0,
                       x_scale="jac", max_nfev=400)
    r1 = np.linalg.norm(s1.fun[:2 * n].reshape(-1, 2), axis=1)
    scale = 1.4826 * float(np.median(r1))
    inl = r1 < max(3.0 * scale, 2.5 * pixel_sigma)
    if inl.sum() < 8:
        inl = np.ones(n, dtype=bool)
    t_in, o_in = times[inl], obs[inl]
    c_in = [c for c, k in zip(cams, inl, strict=True) if k]
    dof = max(2 * len(t_in) - len(x0), 1)
    sigma = max(float(np.sqrt(np.sum(s1.fun[:2 * n].reshape(-1, 2)[inl] ** 2) / dof)), pixel_sigma)
    sol = s1
    for _ in range(2):  # re-whiten the priors once the inlier noise level is known
        p2 = _Problem(t_in, o_in, c_in, anchors, sigma, ub)
        sol = least_squares(p2.fun, np.clip(sol.x, lb + 1e-9, ub - 1e-9), jac=p2.jac, bounds=(lb, ub),
                            loss="linear", x_scale="jac", max_nfev=200)
        pix = sol.fun[:2 * len(t_in)]
        new_sigma = max(float(np.sqrt(np.sum(pix**2) / dof)), pixel_sigma)
        converged = abs(new_sigma - sigma) < 0.2 * sigma
        sigma = new_sigma
        if converged:
            break
    pix = sol.fun[:2 * len(t_in)]
    rms = float(np.sqrt(np.mean(np.sum(pix.reshape(-1, 2) ** 2, axis=1))))
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * sigma**2
    pos, _ = flight_states(sol.x[:11], times)
    msg = f"{int((~inl).sum())} outlier observation(s) dropped" if (~inl).any() else str(sol.message)
    return FlightFit(sol.x[:11], cov, rms, int(inl.sum()), times, pos, bool(sol.success), msg,
                     anchor_times=sol.x[11:].copy())
