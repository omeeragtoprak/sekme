"""Event timing against the E2E-Spot tennis labels (Hong et al., ECCV 2022; github.com/jhong93/spot, BSD-3-Clause).

Each labelled point clip lists the frame of every bounce, swing and serve, tagged with the court half
it happens on: ``near_court_bounce`` is a bounce on the near half, ``far_court_swing`` a stroke by the
far player. Frames count from the clip's first frame, as OpenCV decodes the full match video. Only
timing is labelled, not positions, so this measures event detection, not centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Matching windows in seconds: 0.1 s is the TrackNet bounce tolerance (3 frames at 30 fps); a serve is
# matched as in the Review Room's serve labels (SERVE_TOL_S in courtvision.api.app). Seconds are counted at the
# nominal frame rate (29.97 fps as 30), so 0.1 s is 3 frames on the US Open matches, as on TrackNet, and not
# 2.997 (docs/EVAL_SPOT.md, amendment of 2026-11-14).
TOL_S = {"bounce": 0.1, "stroke": 0.1, "serve": 0.5, "serve_toss": 0.5}
LABELS = {f"{side}_court_{kind}" for side in ("near", "far") for kind in ("bounce", "swing", "serve")}


def nominal_fps(fps: float) -> int:
    """25 for 25 fps, 30 for 29.97 fps: the rate the tolerances in seconds are counted at."""
    return int(round(fps))


def check_clip(clip: dict) -> None:
    """A labelled clip: known event labels, frames inside the clip, a frame count matching its name."""
    n = int(clip["num_frames"])
    start, end = (int(v) for v in clip["video"].rsplit("_", 2)[1:])
    if end - start != n:
        raise ValueError(f"{clip['video']}: name says {end - start} frames, num_frames {n}")
    for e in clip["events"]:
        if e.get("label") not in LABELS:
            raise ValueError(f"{clip['video']}: unknown label {e.get('label')!r}")
        f = e.get("frame")
        if not isinstance(f, int) or isinstance(f, bool) or not 0 <= f < n:
            raise ValueError(f"{clip['video']}: frame {f!r} outside the clip's {n} frames")
# A serve the system finds before a labelled serve of the same side, in the same clip and within this time,
# is most likely a fault the labels leave out: precision is also given without such serves (EVAL_SPOT.md).
FAULT_WINDOW_S = 30.0
SWEEP_S = (0.05, 0.1, 0.2)  # descriptive only: bounce and stroke recall at other windows


@dataclass(frozen=True)
class Event:
    kind: str  # bounce | stroke | serve | serve_toss
    side: str  # near | far: the half the bounce lands on, or the player who strikes
    frame: float


def label_events(clip: dict) -> list[Event]:
    """Labelled events of one clip. A serve is also a stroke; both serve kinds are scored against it."""
    out = []
    for e in clip["events"]:
        side, _, kind = e["label"].partition("_court_")
        f = float(e["frame"])
        if kind == "bounce":
            out.append(Event("bounce", side, f))
        else:
            out.append(Event("stroke", side, f))
            if kind == "serve":
                out += [Event("serve", side, f), Event("serve_toss", side, f)]
    return sorted(out, key=lambda e: e.frame)


def first_bounce_flags(events: list[Event]) -> list[bool]:
    """For each bounce (in frame order): is it the first after a stroke, the one a line call is about?"""
    out, seen = [], False
    for e in events:
        if e.kind == "stroke":
            seen = False
        elif e.kind == "bounce":
            out.append(not seen)
            seen = True
    return out


def system_events(report: dict) -> list[Event]:
    """Strokes and toss-detected serves from the hit detector; serves and bounces from the flights.

    ``serve`` is a flight the pipeline called a serve (it decides the service-box call; it needs a
    fitted flight), ``serve_toss`` a hit flagged as a serve by the toss detector alone. A bounce is any
    flight with a measured bounce (3D fit or the image fallback); its side is the half it landed on,
    from its court y (far half: y > 0), not the receiver the bounce search was restricted to.
    """
    out = []
    for h in report.get("hits", []):
        out.append(Event("stroke", h["side"], float(h["frame"])))
        if h.get("serve"):
            out.append(Event("serve_toss", h["side"], float(h["frame"])))
    for f in report.get("flights", []):
        if f.get("kind") == "serve":
            fr = f["contact_frame"] if f.get("contact_frame") is not None else f["start_frame"]
            out.append(Event("serve", str(f["hitter"]), float(fr)))
        if f.get("bounce") and f["bounce"].get("frame") is not None:
            b = f["bounce"]
            side = "unknown" if b.get("y") is None else ("far" if b["y"] > 0 else "near")
            out.append(Event("bounce", side, float(b["frame"])))
    return sorted(out, key=lambda e: e.frame)


def match_events(ref: list[Event], sys: list[Event], tol_frames: float) -> list[tuple[int, int]]:
    """One-to-one pairs (ref index, sys index) within the tolerance, closest gaps first."""
    pairs = sorted((abs(r.frame - s.frame), i, j) for i, r in enumerate(ref) for j, s in enumerate(sys)
                   if abs(r.frame - s.frame) <= tol_frames)
    used_r, used_s, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_r and j not in used_s:
            used_r.add(i)
            used_s.add(j)
            out.append((i, j))
    return out


def score_clip(clip: dict, report: dict | None) -> dict:
    """Per kind: labelled, system, matched, side agreement, timing gaps (frames); first-bounce counts and
    the tolerance sweep for bounces and strokes. ``report=None`` is a failed analysis: no system events."""
    fps = float(clip["fps"])
    nf = nominal_fps(fps)  # tolerances and the fault window count seconds at this rate
    ref_all = label_events(clip)
    sys_all = system_events(report) if report is not None else []
    out = {"video": clip["video"], "fps": fps, "failed": report is None}
    if report is not None:
        cal = report.get("calibration", {})
        out["calibrated_share"] = cal.get("valid_frames", 0) / max(int(clip["num_frames"]), 1)
    for kind, tol in TOL_S.items():
        ref = [e for e in ref_all if e.kind == kind]
        sys = [e for e in sys_all if e.kind == kind]
        pairs = match_events(ref, sys, tol * nf)
        row = {"labelled": len(ref), "system": len(sys), "matched": len(pairs),
               "same_side": sum(ref[i].side == sys[j].side for i, j in pairs),
               "dt": [sys[j].frame - ref[i].frame for i, j in pairs]}
        if kind in ("bounce", "stroke"):
            row["sweep"] = {f"{t:g}": len(match_events(ref, sys, t * nf)) for t in SWEEP_S}
        if kind in ("serve", "serve_toss"):
            used = {j for _, j in pairs}
            row["before_label"] = sum(
                j not in used and any(r.side == e.side and 0 < r.frame - e.frame <= FAULT_WINDOW_S * nf for r in ref)
                for j, e in enumerate(sys))
        if kind == "bounce":
            first = first_bounce_flags(ref_all)
            hit = {i for i, _ in pairs}
            row["first_labelled"] = sum(first)
            row["first_matched"] = sum(f and i in hit for i, f in enumerate(first))
        out[kind] = row
    return out


def summary(rows: list[dict]) -> dict:
    """Pooled precision / recall / F1 per kind, median timing gap, side agreement, first-bounce recall."""
    def ratio(a: float, b: float) -> float:
        return a / b if b else float("nan")

    out = {"clips": len(rows), "failed": sum(r["failed"] for r in rows),
           "events_per_clip": ratio(sum(r["bounce"]["labelled"] + r["stroke"]["labelled"] for r in rows),
                                    len(rows)),
           "calibrated_share": float(np.mean([r["calibrated_share"] for r in rows if not r["failed"]]))
           if any(not r["failed"] for r in rows) else float("nan")}
    for kind in TOL_S:
        n_ref = sum(r[kind]["labelled"] for r in rows)
        n_sys = sum(r[kind]["system"] for r in rows)
        m = sum(r[kind]["matched"] for r in rows)
        p, rc = ratio(m, n_sys), ratio(m, n_ref)
        dt = np.array([d for r in rows for d in r[kind]["dt"]])
        s = {"labelled": n_ref, "system": n_sys, "matched": m, "precision": p, "recall": rc,
             "f1": 2 * p * rc / (p + rc) if m else 0.0,
             "same_side": ratio(sum(r[kind]["same_side"] for r in rows), m),
             "median_dt_frames": float(np.median(dt)) if len(dt) else float("nan")}
        if rows and "sweep" in rows[0][kind]:
            s["recall_sweep"] = {t: ratio(sum(r[kind]["sweep"][t] for r in rows), n_ref)
                                 for t in rows[0][kind]["sweep"]}
        if kind in ("serve", "serve_toss"):
            before = sum(r[kind].get("before_label", 0) for r in rows)
            s["before_label"] = before
            s["precision_excl_faults"] = ratio(m, n_sys - before)
        if kind == "bounce":
            fl = sum(r[kind]["first_labelled"] for r in rows)
            s["first_recall"] = ratio(sum(r[kind]["first_matched"] for r in rows), fl)
            s["first_labelled"] = fl
        out[kind] = s
    return out


def match_name(video: str) -> str:
    """'usopen_2019_mens_final_medvedev_nadal_100940_101296' -> the match (clip start/end frames dropped)."""
    return video.rsplit("_", 2)[0]


def per_match(rows: list[dict]) -> dict[str, dict]:
    """The summary of each match separately: with three or four matches, the spread across them is the
    honest uncertainty (a bootstrap over so few clusters would understate it)."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(match_name(r["video"]), []).append(r)
    return {m: summary(rs) for m, rs in sorted(by.items())}
