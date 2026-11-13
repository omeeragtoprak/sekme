"""Human ground truth for line calls: a reviewer clicks where the ball touches the court.

The click (image pixels at a calibrated frame) goes through the inverse court homography to the
ground plane, then is judged with exactly the rule the system uses (painted outer edges plus the
contact footprint) in the zone the system judged against. The label's own resolution is reported:
how many centimetres one pixel spans at that spot.
"""

from __future__ import annotations

import numpy as np

from courtvision.geometry.court import Half, Zone, ZoneKind, rally_zone, service_box
from courtvision.officiating.line_call import FOOTPRINT_HALF

CHI2_95_2D = 5.991


def world_from_px(H: list[float] | np.ndarray, px: tuple[float, float]) -> np.ndarray:
    """Ground point (m) of an image pixel, with ``H`` the 3x3 world->image homography (row-major)."""
    Hm = np.asarray(H, dtype=np.float64).reshape(3, 3)
    w = np.linalg.solve(Hm, np.array([px[0], px[1], 1.0]))
    return w[:2] / w[2]


def px_scale_cm(H: list[float] | np.ndarray, px: tuple[float, float]) -> float:
    """Ground distance spanned by one pixel at ``px`` (largest of the two image directions), in cm."""
    p = world_from_px(H, px)
    return 100.0 * max(float(np.linalg.norm(world_from_px(H, (px[0] + 1, px[1])) - p)),
                       float(np.linalg.norm(world_from_px(H, (px[0], px[1] + 1)) - p)))


def zone_from_name(name: str) -> Zone:
    """Rebuild the zone a call was judged in from its name ("far-singles", "service-near-left", ...)."""
    parts = name.split("-")
    if parts[0] == "service" and len(parts) == 3:
        # service_box names the box "left" when the server stands right of centre (x > 0).
        return service_box(Half(parts[1]), 1.0 if parts[2] == "left" else -1.0)
    if len(parts) == 2:
        return rally_zone(Half(parts[0]), ZoneKind(parts[1]))
    raise ValueError(f"unknown zone name: {name}")


def evaluate(world_xy: np.ndarray, call: dict) -> dict:
    """Compare a labelled bounce with the system call (mean, covariance, verdict)."""
    xy = np.asarray(world_xy, dtype=np.float64)
    mean = np.array([call["x"], call["y"]], dtype=np.float64)
    cov = np.asarray(call["cov"], dtype=np.float64)
    d = xy - mean
    d2 = float(d @ np.linalg.solve(cov + 1e-10 * np.eye(2), d))
    zone = zone_from_name(call["zone"])
    inside = bool(zone.contains(xy[None, :], grow=FOOTPRINT_HALF)[0])
    margin, line = zone.signed_margin(xy)
    truth = "IN" if inside else "OUT"
    system = call["verdict"]
    return {
        "error_cm": round(100 * float(np.linalg.norm(d)), 1),
        "mahalanobis2": round(d2, 2),
        "inside_95": d2 <= CHI2_95_2D,
        "truth": truth,
        "truth_margin_cm": round(100 * (float(margin) + FOOTPRINT_HALF), 1),
        "truth_line": line,
        "system": system,
        "agrees": None if system == "REVIEW" else system == truth,
    }


def summarize(records: list[dict]) -> dict:
    """Latest label per shot -> error, calibration of the 95% ellipse and verdict agreement."""
    latest: dict[int, dict] = {}
    for r in sorted(records, key=lambda r: r["created"]):
        latest[r["flight_id"]] = r
    ev = [r["eval"] for r in latest.values()]
    if not ev:
        return {"n": 0}
    err = np.array([e["error_cm"] for e in ev])
    decisive = [e for e in ev if e["agrees"] is not None]
    return {
        "n": len(ev),
        "error_cm_median": round(float(np.median(err)), 1),
        "error_cm_p90": round(float(np.percentile(err, 90)), 1),
        "inside_95_share": round(sum(e["inside_95"] for e in ev) / len(ev), 3),
        "decisive": len(decisive),
        "agreements": sum(bool(e["agrees"]) for e in decisive),
        "reviews": len(ev) - len(decisive),
        "review_truth": {"IN": sum(e["truth"] == "IN" for e in ev if e["agrees"] is None),
                         "OUT": sum(e["truth"] == "OUT" for e in ev if e["agrees"] is None)},
    }


def serve_state(records: list[dict]) -> tuple[list[dict], bool]:
    """Replay the append-only serve-label log (file order): serves not removed, and the latest "complete" flag."""
    serves: dict[str, dict] = {}
    complete = False
    for r in records:
        if r["type"] == "serve":
            serves[r["id"]] = r
        elif r["type"] == "remove":
            serves.pop(r["target"], None)
        elif r["type"] == "complete":
            complete = bool(r["complete"])
    return list(serves.values()), complete


def serve_summary(records: list[dict], system_serve_frames: list[float], fps: float, tol_s: float = 0.5) -> dict:
    """Serve recall (and, once the reviewer says every serve is labelled, precision) against the system.

    Labelled serves (not removed) and system serves are paired one-to-one, smallest frame gap first,
    within ``tol_s`` seconds. Recall is the share of labelled serves the system found. Precision is
    only meaningful when the labels are complete: otherwise an unlabelled real serve would count
    against the system, so it is withheld (``None``).
    """
    labelled, complete = serve_state(records)
    tol = tol_s * fps
    pairs = sorted((abs(s - lab["frame"]), i, j) for i, lab in enumerate(labelled)
                   for j, s in enumerate(system_serve_frames) if abs(s - lab["frame"]) <= tol)
    pair_of: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in pair_of and j not in used:
            pair_of[i] = j
            used.add(j)
    matched = len(pair_of)
    return {
        "labelled": len(labelled),
        "system": len(system_serve_frames),
        "matched": matched,
        "recall": round(matched / len(labelled), 3) if labelled else None,
        "complete": complete,
        "precision": round(matched / len(system_serve_frames), 3) if complete and system_serve_frames else None,
        "tol_s": tol_s,
        "rows": [{"id": lab["id"], "frame": lab["frame"],
                  "matched_system_frame": system_serve_frames[pair_of[i]] if i in pair_of else None,
                  "dt_frames": round(system_serve_frames[pair_of[i]] - lab["frame"], 2) if i in pair_of else None}
                 for i, lab in enumerate(labelled)],
        "unmatched_system": [s for j, s in enumerate(system_serve_frames) if j not in used],
    }
