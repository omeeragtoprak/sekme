"""Scoring of the six-video test in docs/EVAL_SERVE_SPEED.md, written before any of the videos was downloaded.

Two candidates are decided on the same blind labels (serve contact brackets, server side, stadium board):

* ``--fps-scaled-thresholds`` (default analysis against the scaled one, videos of 50 fps and more);
* ``--pin-serve-height`` (the scaled analysis against the scaled one with the pin).

Matching follows ``scripts/eval_serve_contact.py``: a labelled serve and a system serve of the same side whose
contact estimate lies within 30 frames of the bracket, one-to-one, closest first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MATCH_FRAMES = 30
GOOD_FRAMES = 2
FAULT_WINDOW_S = 30.0
MPH_TO_KMH = 1.609344


def bracket_error(est: float, lo: int, hi: int) -> float:
    return 0.0 if lo <= est <= hi else float(min(abs(est - lo), abs(est - hi)))


def system_serves(report: dict) -> list[dict]:
    """Serve flights: id, side, contact estimate (fitted contact frame, else start frame), speed, start, rejected."""
    out = []
    for f in report["flights"]:
        if f.get("kind") != "serve":
            continue
        est = f.get("contact_frame")
        out.append({"id": f["id"], "side": f["hitter"], "est": float(est if est is not None else f["start_frame"]),
                    "kmh": f.get("hit_speed_kmh"), "start": f["start_frame"], "rejected": f.get("rejected"),
                    "verdict": (f.get("call") or {}).get("verdict")})
    return out


def match(labels: list[dict], serves: list[dict]) -> dict[int, tuple[dict, float]]:
    pairs = sorted((bracket_error(s["est"], lb["lo"], lb["hi"]), i, j)
                   for i, lb in enumerate(labels) for j, s in enumerate(serves)
                   if s["side"] == lb["side"] and bracket_error(s["est"], lb["lo"], lb["hi"]) <= MATCH_FRAMES)
    used_l, used_s, out = set(), set(), {}
    for err, i, j in pairs:
        if i in used_l or j in used_s:
            continue
        used_l.add(i)
        used_s.add(j)
        out[i] = (serves[j], err)
    return out


def board_kmh(label: dict) -> float | None:
    r = label.get("radar")
    if not r:
        return None
    return float(r["value"]) * (MPH_TO_KMH if r.get("unit", "mph") == "mph" else 1.0)


def recall_stats(labels: list[dict], report: dict) -> dict:
    """Labelled, matched, matched within 2 frames, and false serves (unmatched, likely faults set apart)."""
    fps = float(report["video"]["fps"])
    serves = system_serves(report)
    m = match(labels, serves)
    used = {s["id"] for s, _ in m.values()}
    false = 0
    for s in serves:
        if s["id"] in used:
            continue
        fault = any(lb["side"] == s["side"] and 0 < lb["lo"] - s["est"] <= FAULT_WINDOW_S * fps for lb in labels)
        false += not fault
    return {"labelled": len(labels), "matched": len(m), "within2": sum(e <= GOOD_FRAMES for _, e in m.values()),
            "false_serves": false}


def decisive_flips(rep_a: dict, rep_b: dict, tol: float = 2.0) -> list[tuple[float, str, str]]:
    """Bounces found by both runs (within ``tol`` frames, one-to-one) called IN by one and OUT by the other."""
    def calls(r: dict) -> list[tuple[float, str]]:
        return [(float(f["bounce"]["frame"]), f["call"]["verdict"]) for f in r["flights"]
                if f.get("call") and f.get("bounce")]
    a, b = calls(rep_a), calls(rep_b)
    pairs = sorted((abs(fa - fb), i, j) for i, (fa, _) in enumerate(a) for j, (fb, _) in enumerate(b)
                   if abs(fa - fb) <= tol)
    ua, ub, out = set(), set(), []
    for _, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        va, vb = a[i][1], b[j][1]
        if {va, vb} == {"IN", "OUT"}:
            out.append((a[i][0], va, vb))
    return out


@dataclass
class Verdict:
    decision: str  # "accepted" | "rejected" | "inconclusive" | "cannot decide"
    criteria: dict[str, bool] = field(default_factory=dict)
    numbers: dict = field(default_factory=dict)


def fps_rule(videos: list[tuple[float, list[dict], dict, dict]]) -> Verdict:
    """``videos``: (fps, labels, default report, scaled report). Criteria of the second candidate."""
    use = [v for v in videos if v[0] >= 50.0]
    if len(use) < 3:
        return Verdict("cannot decide", numbers={"videos_50fps": len(use)})
    d = [recall_stats(lab, a) for _, lab, a, _ in use]
    s = [recall_stats(lab, b) for _, lab, _, b in use]
    lab_n = sum(x["labelled"] for x in d)
    rec_d, rec_s = 100 * sum(x["matched"] for x in d) / lab_n, 100 * sum(x["matched"] for x in s) / lab_n
    md, ms = sum(x["matched"] for x in d), sum(x["matched"] for x in s)
    sh_d = 100 * sum(x["within2"] for x in d) / md if md else float("nan")
    sh_s = 100 * sum(x["within2"] for x in s) / ms if ms else float("nan")
    flips = [f for _, _, a, b in use for f in decisive_flips(a, b)]
    fd, fs = sum(x["false_serves"] for x in d), sum(x["false_serves"] for x in s)
    crit = {"recall +10 points": rec_s - rec_d >= 10.0,
            "within-2 share falls <= 5 points": (sh_s >= sh_d - 5.0) if md and ms else ms > 0,
            "no decisive flip": not flips,
            "false serves +<= 2": fs - fd <= 2}
    nums = {"videos_50fps": len(use), "labelled": lab_n, "recall_default": rec_d, "recall_scaled": rec_s,
            "within2_share_default": sh_d, "within2_share_scaled": sh_s, "flips": flips,
            "false_serves_default": fd, "false_serves_scaled": fs}
    return Verdict("accepted" if all(crit.values()) else "rejected", crit, nums)


def pin_rule(videos: list[tuple[list[dict], dict, dict]], min_paired: int = 15) -> Verdict:
    """``videos``: (labels, unpinned report, pinned report). Criteria of the pin candidate."""
    rows = []  # (label board, |e_unpinned|, |e_pinned| or None)
    verdict_changes = []
    for labels, rep_u, rep_p in videos:
        su, sp = system_serves(rep_u), {s["start"]: s for s in system_serves(rep_p)}
        for i, (s, _) in match(labels, su).items():
            b = board_kmh(labels[i])
            if b is None or s["kmh"] is None or s["rejected"] is not None:
                continue
            p = sp.get(s["start"])
            ep = (abs(100 * (p["kmh"] - b) / b) if p is not None and p["kmh"] is not None and p["rejected"] is None
                  else None)
            rows.append((b, abs(100 * (s["kmh"] - b) / b), ep))
        for s in su:
            p = sp.get(s["start"])
            if (p["verdict"] if p is not None else None) != s["verdict"]:
                verdict_changes.append((s["start"], s["verdict"], p["verdict"] if p is not None else None))
    n = len(rows)
    if n < min_paired:
        return Verdict("inconclusive", numbers={"paired": n})
    both = [(eu, ep) for _, eu, ep in rows if ep is not None]
    drop = float(np.mean([eu for eu, _ in both]) - np.mean([ep for _, ep in both])) if both else float("nan")
    closer = sum(1 for eu, ep in both if eu - ep >= 0.5)
    further = sum(1 for eu, ep in both if ep - eu >= 0.5) + sum(1 for _, _, ep in rows if ep is None)
    crit = {"mean |error| falls >= 1.5 points": bool(drop >= 1.5),
            "two thirds closer": (closer + further) > 0 and closer >= 2 / 3 * (closer + further),
            "no serve verdict change": not verdict_changes}
    nums = {"paired": n, "both_measured": len(both), "mean_abs_drop_points": drop, "closer": closer,
            "further_or_lost": further, "verdict_changes": verdict_changes}
    return Verdict("accepted" if all(crit.values()) else "rejected", crit, nums)
