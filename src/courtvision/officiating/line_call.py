"""Probabilistic line calls: never a fake millimetre, always a probability.

The bounce estimate (mean + 2x2 covariance on the ground, metres) is
compared with the legal zone. A ball is IN when its contact footprint
touches the zone or its lines, so the zone's line edges are grown by the
footprint half-length. P(IN) is computed by Monte Carlo (exact for corners),
and the verdict is only IN/OUT when that probability is decisive;
otherwise the call goes to a human (REVIEW).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from courtvision.geometry.court import Zone

FOOTPRINT_HALF = 0.02  # m: centre-to-edge of the compressed contact patch


@dataclass(frozen=True)
class LineCall:
    verdict: str  # "IN" | "OUT" | "REVIEW"
    p_in: float
    margin_m: float  # signed distance of the estimated footprint edge to the governing line (>0 = in)
    sigma_m: float  # 1-sigma uncertainty along that line's normal
    line: str
    zone: str
    x: float
    y: float
    cov: list[list[float]]
    conflict: str | None = None  # independent evidence disagreeing with the geometry

    def to_dict(self) -> dict:
        return asdict(self)


def with_play_evidence(call: LineCall, play_continued: bool | None) -> LineCall:
    """Fuse rally evidence: if the receiver struck the ball after the bounce, an OUT is not decisive.

    Players do occasionally play out balls, so this never turns a call into IN; it
    downgrades OUT to REVIEW and records why. P(in) stays the geometric probability.
    """
    if play_continued and call.verdict == "OUT":
        return replace(call, verdict="REVIEW",
                       conflict="geometri dışarıda diyor ama karşı oyuncu topa vurdu ve ralli sürdü")
    return call


def call_bounce(xy: np.ndarray, cov: np.ndarray, zone: Zone, footprint: float = FOOTPRINT_HALF,
                decisive: float = 0.95, n_samples: int = 40000, seed: int = 0) -> LineCall:
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    cov = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    rng = np.random.default_rng(seed)
    # numpy 2.x on macOS Accelerate emits spurious FP warnings from matmul; results are checked below.
    with np.errstate(all="ignore"):
        samples = xy + rng.standard_normal((n_samples, 2)) @ np.linalg.cholesky(cov).T
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("non-finite Monte Carlo samples for line call")
    p_in = float(np.mean(zone.contains(samples, grow=footprint)))
    margin, line = zone.signed_margin(xy)
    margin += footprint
    # 1-sigma along the governing line's normal: side/centre lines run along y, the others along x.
    along_y = "sideline" in line or "centre" in line
    sigma = float(np.sqrt(cov[0, 0] if along_y else cov[1, 1]))
    if p_in >= decisive:
        verdict = "IN"
    elif p_in <= 1.0 - decisive:
        verdict = "OUT"
    else:
        verdict = "REVIEW"
    return LineCall(verdict, p_in, float(margin), sigma, line, zone.name, float(xy[0]), float(xy[1]),
                    cov.tolist())
