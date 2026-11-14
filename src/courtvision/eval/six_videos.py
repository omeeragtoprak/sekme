"""Scoring of the six-video test in docs/EVAL_SERVE_SPEED.md, written before any of the videos was downloaded.

Three candidates are decided on the same blind labels:

* ``--fps-scaled-thresholds``: default analysis against the scaled one (videos of 50 fps and more), serve labels;
* ``--pin-serve-height``: the scaled analysis against the scaled one with the pin, serve labels and the board;
* ``--player-size-from-camera``: default analysis against the candidate, player feet labelled on frame samples.

Serve matching follows ``scripts/eval_serve_contact.py``: a labelled serve and a system serve of the same side
whose contact estimate lies within 30 frames of the bracket, one-to-one, closest first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

MATCH_FRAMES = 30
GOOD_FRAMES = 2
FAULT_WINDOW_S = 30.0
MPH_TO_KMH = 1.609344
SIDES = ("near", "far")


def _pct(num: int, den: int) -> Fraction:
    """A percentage of counts, exact: the rules' "at least" and "no more than" hold at their boundary."""
    return Fraction(100 * num, den)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: object) -> bool:
    return (isinstance(v, int | float) and not isinstance(v, bool)) and bool(np.isfinite(v))


def check_serve_labels(labels: list[dict]) -> None:
    """Every serve label has its bracket, side and board reading (or null); a missing field is not a default."""
    for k, lb in enumerate(labels):
        if not isinstance(lb, dict) or not {"lo", "hi", "side", "radar"} <= lb.keys():
            raise ValueError(f"serve label {k}: needs lo, hi, side and radar (null when not read)")
        if not (_is_int(lb["lo"]) and _is_int(lb["hi"]) and 0 <= lb["lo"] <= lb["hi"]):
            raise ValueError(f"serve label {k}: bracket must be frames 0 <= lo <= hi")
        if lb["side"] not in SIDES:
            raise ValueError(f"serve label {k}: side must be near or far")
        r = lb["radar"]
        if r is not None and not (isinstance(r, dict) and r.get("unit") in ("mph", "km/h") and _is_num(r.get("value"))
                                  and r["value"] > 0):
            raise ValueError(f"serve label {k}: radar must be null or {{value, unit: mph | km/h}}")


def check_player_labels(labels: list[dict]) -> None:
    """Every frame label says whether it is the main camera and gives both players (feet [x, y] or null)."""
    seen = set()
    for k, lb in enumerate(labels):
        if not isinstance(lb, dict) or not {"frame", "main", "near", "far"} <= lb.keys():
            raise ValueError(f"player label {k}: needs frame, main, near and far")
        if not _is_int(lb["frame"]) or lb["frame"] < 0 or not isinstance(lb["main"], bool):
            raise ValueError(f"player label {k}: frame must be a frame number and main true or false")
        for side in SIDES:
            v = lb[side]
            if v is not None and not (isinstance(v, list) and len(v) == 2 and all(_is_num(c) for c in v)):
                raise ValueError(f"player label {k}: {side} must be null or [x, y]")
        if lb["frame"] in seen:
            raise ValueError(f"player label {k}: frame {lb['frame']} labelled twice")
        seen.add(lb["frame"])


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
    return float(r["value"]) * (MPH_TO_KMH if r["unit"] == "mph" else 1.0)


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


def _bounce_calls(r: dict) -> list[tuple[float, str]]:
    return [(float(f["bounce"]["frame"]), f["call"]["verdict"]) for f in r["flights"]
            if f.get("call") and f.get("bounce")]


def decisive_flips(rep_a: dict, rep_b: dict, tol: float = 2.0) -> list[tuple[float, str, str]]:
    """Bounces found by both runs (within ``tol`` frames) called IN by one and OUT by the other. Every pair within
    the tolerance counts (two bounces of one rally are never 2 frames apart), so a one-to-one pairing that picks
    the agreeing neighbour cannot hide a flip."""
    return [(fa, va, vb) for fa, va in _bounce_calls(rep_a) for fb, vb in _bounce_calls(rep_b)
            if abs(fa - fb) <= tol and {va, vb} == {"IN", "OUT"}]


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
    if lab_n == 0:
        return Verdict("cannot decide", numbers={"videos_50fps": len(use), "labelled": 0})
    md, ms = sum(x["matched"] for x in d), sum(x["matched"] for x in s)
    wd, ws = sum(x["within2"] for x in d), sum(x["within2"] for x in s)
    rec_d, rec_s = _pct(md, lab_n), _pct(ms, lab_n)
    sh_d, sh_s = (_pct(wd, md) if md else None), (_pct(ws, ms) if ms else None)
    flips = [f for _, _, a, b in use for f in decisive_flips(a, b)]
    fd, fs = sum(x["false_serves"] for x in d), sum(x["false_serves"] for x in s)
    crit = {"recall +10 points": rec_s - rec_d >= 10,
            "within-2 share falls <= 5 points": (sh_s >= sh_d - 5) if sh_d is not None and sh_s is not None else ms > 0,
            "no decisive flip": not flips,
            "false serves +<= 2": fs - fd <= 2}
    nums = {"videos_50fps": len(use), "labelled": lab_n, "recall_default": float(rec_d), "recall_scaled": float(rec_s),
            "within2_share_default": float(sh_d) if sh_d is not None else None,
            "within2_share_scaled": float(sh_s) if sh_s is not None else None, "flips": flips,
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
        su_by = {s["start"]: s for s in su}
        for start in sorted(set(su_by) | set(sp)):  # serves of either run
            vu = su_by[start]["verdict"] if start in su_by else None
            vp = sp[start]["verdict"] if start in sp else None
            if vu != vp:
                verdict_changes.append((start, vu, vp))
    n = len(rows)
    if n < min_paired:
        return Verdict("inconclusive", numbers={"paired": n, "verdict_changes": verdict_changes})
    both = [(eu, ep) for _, eu, ep in rows if ep is not None]
    drop = float(np.mean([eu for eu, _ in both]) - np.mean([ep for _, ep in both])) if both else float("nan")
    closer = sum(1 for eu, ep in both if eu - ep >= 0.5)
    further = sum(1 for eu, ep in both if ep - eu >= 0.5) + sum(1 for _, _, ep in rows if ep is None)
    crit = {"mean |error| falls >= 1.5 points": bool(drop >= 1.5),
            "two thirds closer": (closer + further) > 0 and 3 * closer >= 2 * (closer + further),
            "no serve verdict change": not verdict_changes}
    nums = {"paired": n, "both_measured": len(both), "mean_abs_drop_points": drop, "closer": closer,
            "further_or_lost": further, "verdict_changes": verdict_changes}
    return Verdict("accepted" if all(crit.values()) else "rejected", crit, nums)


# --- Third candidate: player size gate from the camera (--player-size-from-camera) ----------------------------
# Blind labels on a fixed frame sample: for each main-camera frame, each player's feet in the image, or none.

PRESENCE_FRAMES = 80
DISAGREE_FRAMES = 60
# Tracked feet count as the labelled player within this share of the tracked box's height (a ball kid beside
# the far player is out), but never less than PRESENCE_MIN_TOL_PX at 1080 lines (labelling precision).
PRESENCE_TOL_FRAC = 0.3
PRESENCE_MIN_TOL_PX = 15.0


def _rng(key: str) -> np.random.Generator:
    import hashlib

    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def presence_sample(name: str, n_frames: int, k: int = PRESENCE_FRAMES) -> list[int]:
    """The frames to label for one video, drawn from its name and frame count alone (no system output)."""
    return sorted(int(i) for i in _rng(name).choice(n_frames, size=min(k, n_frames), replace=False))


def _feet(frames: dict, side: str, i: int) -> tuple[float, float, float] | None:
    box = frames["players"][side]["box"][i]
    if box is None or any(v is None for v in box):
        return None
    return (box[0] + box[2]) / 2, box[3], box[3] - box[1]


def _found(feet: tuple[float, float, float] | None, truth, height: int) -> bool:
    if feet is None or truth is None:
        return False
    tol = max(PRESENCE_MIN_TOL_PX * height / 1080.0, PRESENCE_TOL_FRAC * feet[2])
    return bool(np.hypot(feet[0] - truth[0], feet[1] - truth[1]) <= tol)


def disagreement_sample(name: str, fa: dict, fb: dict, valid: list[bool], k: int = DISAGREE_FRAMES) -> list[int]:
    """Calibrated frames where the two runs pick different players (one tracks and the other does not, or their
    feet are further apart than the tolerance) for either side; up to ``k``, drawn with a seed from the name.
    The labeller sees only the frames, never which run picked what."""
    def differ(i: int, side: str) -> bool:
        a, b = _feet(fa, side, i), _feet(fb, side, i)
        if (a is None) != (b is None):
            return True
        if a is None or b is None:
            return False
        return np.hypot(a[0] - b[0], a[1] - b[1]) > PRESENCE_TOL_FRAC * min(a[2], b[2])
    cand = [i for i in range(len(valid)) if valid[i] and (differ(i, "near") or differ(i, "far"))]
    if len(cand) <= k:
        return cand
    return sorted(int(i) for i in _rng(name + ":disagree").choice(cand, size=k, replace=False))


def presence_stats(labels: list[dict], frames: dict, height: int) -> dict:
    """``labels``: [{"frame", "main": bool, "near": [x, y] | None, "far": [x, y] | None}] (image feet points).
    Per player: frames where the player is labelled in view, found (tracked feet within tolerance), and wrong
    (tracked, but away from the labelled feet or with the player labelled out of view)."""
    out: dict = {"main_frames": 0}
    for side in ("near", "far"):
        out[side] = {"present": 0, "found": 0, "wrong": 0}
    for lb in labels:
        if not lb.get("main"):
            continue
        out["main_frames"] += 1
        for side in ("near", "far"):
            feet, truth = _feet(frames, side, lb["frame"]), lb.get(side)
            out[side]["present"] += truth is not None
            if feet is None:
                continue
            if _found(feet, truth, height):
                out[side]["found"] += 1
            else:
                out[side]["wrong"] += 1
    return out


def disagreement_stats(labels: list[dict], fa: dict, fb: dict, height: int) -> dict:
    """Per player, on disagreement frames: frames where exactly one run is right, and how many are run b's.
    Right: tracked within tolerance of a labelled player, or not tracking a player labelled out of view."""
    out = {side: {"one_right": 0, "b_right": 0} for side in ("near", "far")}
    for lb in labels:
        if not lb.get("main"):
            continue
        for side in ("near", "far"):
            truth = lb.get(side)
            ok = []
            for fr in (fa, fb):
                feet = _feet(fr, side, lb["frame"])
                ok.append(_found(feet, truth, height) if truth is not None else feet is None)
            if ok[0] != ok[1]:
                out[side]["one_right"] += 1
                out[side]["b_right"] += ok[1]
    return out


def player_rule(videos: list[tuple[list[dict], list[dict], dict, dict, int]], min_near: int = 100,
                min_one_right: int = 10) -> Verdict:
    """``videos``: (random-sample labels, disagreement labels, default frames.json, candidate frames.json,
    frame height)."""
    tot = {k: {"main_frames": 0, "near": {"present": 0, "found": 0, "wrong": 0},
               "far": {"present": 0, "found": 0, "wrong": 0}} for k in ("default", "candidate")}
    dis = {side: {"one_right": 0, "b_right": 0} for side in ("near", "far")}
    for labels, dlabels, fa, fb, h in videos:
        for k, fj in (("default", fa), ("candidate", fb)):
            st = presence_stats(labels, fj, h)
            tot[k]["main_frames"] += st["main_frames"]
            for side in ("near", "far"):
                for m in ("present", "found", "wrong"):
                    tot[k][side][m] += st[side][m]
        ds = disagreement_stats(dlabels, fa, fb, h)
        for side in ("near", "far"):
            for m in ("one_right", "b_right"):
                dis[side][m] += ds[side][m]
    near_present = tot["default"]["near"]["present"]
    if near_present < min_near:
        return Verdict("inconclusive", numbers={"near_present": near_present})

    def cov(k: str, side: str) -> Fraction:
        return _pct(tot[k][side]["found"], max(tot[k][side]["present"], 1))

    def wrong(k: str) -> Fraction:
        # Share of player-frames: two per labelled main-camera frame (near and far), whether or not the player is
        # in view, since tracking a player labelled out of view is a wrong selection too.
        return _pct(tot[k]["near"]["wrong"] + tot[k]["far"]["wrong"], max(2 * tot[k]["main_frames"], 1))

    crit = {"near coverage +5 points": cov("candidate", "near") - cov("default", "near") >= 5,
            "far coverage falls <= 1 point": cov("candidate", "far") >= cov("default", "far") - 1,
            "wrong selections rise <= 1 point": wrong("candidate") <= wrong("default") + 1}
    for side in ("near", "far"):  # judged only where at least min_one_right frames have exactly one run right
        d = dis[side]
        if d["one_right"] >= min_one_right:
            crit[f"disagreements ({side}): candidate right in >= 2/3"] = 3 * d["b_right"] >= 2 * d["one_right"]
    nums = {"near_present": near_present, "far_present": tot["default"]["far"]["present"],
            "near_coverage": [float(cov("default", "near")), float(cov("candidate", "near"))],
            "far_coverage": [float(cov("default", "far")), float(cov("candidate", "far"))],
            "wrong_share": [float(wrong("default")), float(wrong("candidate"))], "disagreements": dis, "totals": tot}
    return Verdict("accepted" if all(crit.values()) else "rejected", crit, nums)


# --- Reported, not used by any rule ------------------------------------------------------------------------------


def _summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, float)
    return {"n": len(xs), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def call_changes(pairs: list[tuple[dict, dict]], tol: float = 2.0) -> dict:
    """Over (run a, run b) report pairs: calls paired by bounce frame (one-to-one, closest first), verdict changes
    (REVIEW included) on serves and on other flights, calls only in one run, and for the paired calls the bounce
    shift (cm) and the change in the bounce's standard deviation (cm) and in P(in), serves and others apart."""
    out: dict = {"calls_a": 0, "calls_b": 0, "only_a": 0, "only_b": 0,
                 "verdict_changes": {"serve": [], "other": []}}
    acc = {g: {"shift_cm": [], "sigma_change_cm": [], "p_in_change": []} for g in ("serve", "other")}
    for rep_a, rep_b in pairs:
        a = [f for f in rep_a["flights"] if f.get("call") and f.get("bounce")]
        b = [f for f in rep_b["flights"] if f.get("call") and f.get("bounce")]
        cand = sorted((abs(fa["bounce"]["frame"] - fb["bounce"]["frame"]), i, j)
                      for i, fa in enumerate(a) for j, fb in enumerate(b)
                      if abs(fa["bounce"]["frame"] - fb["bounce"]["frame"]) <= tol)
        ua, ub = set(), set()
        for _, i, j in cand:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            g = "serve" if a[i].get("kind") == "serve" else "other"
            ca, cb = a[i]["call"], b[j]["call"]
            if ca["verdict"] != cb["verdict"]:
                out["verdict_changes"][g].append((float(a[i]["bounce"]["frame"]), ca["verdict"], cb["verdict"]))
            if all(c.get(k) is not None for c in (ca, cb) for k in ("x", "y")):
                acc[g]["shift_cm"].append(100 * float(np.hypot(ca["x"] - cb["x"], ca["y"] - cb["y"])))
            if ca.get("sigma_m") is not None and cb.get("sigma_m") is not None:
                acc[g]["sigma_change_cm"].append(100 * (cb["sigma_m"] - ca["sigma_m"]))
            if ca.get("p_in") is not None and cb.get("p_in") is not None:
                acc[g]["p_in_change"].append(abs(cb["p_in"] - ca["p_in"]))
        out["calls_a"] += len(a)
        out["calls_b"] += len(b)
        out["only_a"] += len(a) - len(ua)
        out["only_b"] += len(b) - len(ub)
    out["review_changes"] = sum("REVIEW" in (va, vb) for g in ("serve", "other")
                                for _, va, vb in out["verdict_changes"][g])
    out["paired"] = {g: {k: _summary(v) for k, v in acc[g].items()} for g in acc}
    return out


def speed_report(pairs: list[tuple[list[dict], dict]]) -> dict:
    """Over (serve labels, report) pairs: the matched serves with a board reading and a trusted fitted speed, as
    error = 100 (fitted - board) / board: mean signed and |error|, the share within 5 %, by side and by board
    speed (< 100 mph, >= 115 mph); and the pipeline's own automatic board pairs."""
    rows, auto = [], []
    for labels, rep in pairs:
        for i, (s, _) in match(labels, system_serves(rep)).items():
            b = board_kmh(labels[i])
            if b is not None and s["kmh"] is not None and s["rejected"] is None:
                rows.append((s["side"], b / MPH_TO_KMH, 100 * (s["kmh"] - b) / b))
        auto += [100 * (f["hit_speed_kmh"] - f["board_speed_kmh"]) / f["board_speed_kmh"] for f in rep["flights"]
                 if f.get("kind") == "serve" and f.get("board_speed_kmh") and f.get("hit_speed_kmh") is not None
                 and f.get("rejected") is None]

    def stats(e: list[float]) -> dict:
        if not e:
            return {"n": 0}
        a = np.asarray(e, float)
        return {"n": len(e), "mean_signed": float(a.mean()), "mean_abs": float(np.abs(a).mean()),
                "within_5pct": float(np.mean(np.abs(a) <= 5.0))}

    return {"labelled_board": stats([e for _, _, e in rows]),
            "near": stats([e for sd, _, e in rows if sd == "near"]),
            "far": stats([e for sd, _, e in rows if sd == "far"]),
            "board_below_100mph": stats([e for _, mph, e in rows if mph < 100]),
            "board_115mph_up": stats([e for _, mph, e in rows if mph >= 115]),
            "automatic_pairs": stats(auto)}
