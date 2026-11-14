"""Score serve contact timing against serve contacts bracketed by eye (docs/EVAL_SERVE_CONTACT.md).

    .venv/bin/python scripts/eval_serve_contact.py LABELS.json v3_old=RUN v3_new=RUN v4_old=RUN v4_new=RUN

LABELS.json: {"video": ..., "serves": [{"lo": int, "hi": int, "side": "near"|"far",
              "radar": {"value": float, "unit": "km/h"|"mph"} | null}, ...]}
RUN: a run directory (its report.json). Names ending in _old/_new pair up per net for rule 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MATCH_FRAMES = 30  # 0.5 s at 59.94 fps
GOOD_FRAMES = 2
MOVED_FRAMES = 0.5


def bracket_error(est: float, lo: int, hi: int) -> float:
    return 0.0 if lo <= est <= hi else float(min(abs(est - lo), abs(est - hi)))


def system_serves(report: dict) -> list[dict]:
    out = []
    for f in report["flights"]:
        if f.get("kind") != "serve":
            continue
        est = f.get("contact_frame")
        out.append({"id": f["id"], "side": f["hitter"], "est": float(est if est is not None else f["start_frame"]),
                    "kmh": f.get("hit_speed_kmh")})
    return out


def match(labels: list[dict], serves: list[dict]) -> dict[int, tuple[dict, float]]:
    """Label index -> (system serve, contact error); one-to-one, closest first, same side, within 0.5 s."""
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


def summary(labels: list[dict], m: dict[int, tuple[dict, float]]) -> dict:
    errs = np.array([e for _, e in m.values()])
    radar = []
    for i, (s, _) in m.items():
        r = labels[i].get("radar")
        if r and s["kmh"] is not None:
            board = r["value"] * (1.609344 if r["unit"] == "mph" else 1.0)
            radar.append(100 * (s["kmh"] - board) / board)
    return {"labelled": len(labels), "matched": len(m), "good": int(np.sum(errs <= GOOD_FRAMES)),
            "mean_err": float(errs.mean()) if len(errs) else float("nan"),
            "median_err": float(np.median(errs)) if len(errs) else float("nan"),
            "radar_n": len(radar), "radar_mean_abs_pct": float(np.mean(np.abs(radar))) if radar else float("nan")}


def main() -> None:
    labels = json.loads(Path(sys.argv[1]).read_text())["serves"]
    runs = dict(a.split("=", 1) for a in sys.argv[2:])
    matched, stats = {}, {}
    for name, d in runs.items():
        matched[name] = match(labels, system_serves(json.loads((Path(d) / "report.json").read_text())))
        stats[name] = summary(labels, matched[name])
        s = stats[name]
        print(f"{name:8s} labelled {s['labelled']}  matched {s['matched']}  good(<= {GOOD_FRAMES} fr) {s['good']}  "
              f"mean err {s['mean_err']:.2f}  median {s['median_err']:.2f}  "
              f"radar n={s['radar_n']} mean|err| {s['radar_mean_abs_pct']:.1f} %")
    print("\nper serve: lo-hi side | " + " | ".join(runs))
    for i, lb in enumerate(labels):
        cells = []
        for name in runs:
            hit = matched[name].get(i)
            cells.append(f"{hit[0]['est']:8.1f} ({hit[1]:4.1f})" if hit else "       -       ")
        print(f"{lb['lo']:6d}-{lb['hi']:<6d} {lb['side']:4s} | " + " | ".join(cells))

    nets = sorted({n.rsplit("_", 1)[0] for n in runs if n.endswith(("_old", "_new"))})
    if not nets:
        return
    for net in nets:
        if f"{net}_old" not in runs or f"{net}_new" not in runs:
            sys.exit(f"{net}: need both {net}_old and {net}_new")
    print("\nrule 1 (gap contact change; mean error over the labels matched in both runs):")
    moved: set[int] = set()
    verdict = {"no_worse": True, "better": False, "good_kept": True}
    for net in nets:
        old, new = stats[f"{net}_old"], stats[f"{net}_new"]
        mo, mn = matched[f"{net}_old"], matched[f"{net}_new"]
        moved |= {i for i in set(mo) | set(mn)
                  if i not in mo or i not in mn or abs(mo[i][0]["est"] - mn[i][0]["est"]) > MOVED_FRAMES}
        both = sorted(set(mo) & set(mn))
        po = float(np.mean([mo[i][1] for i in both])) if both else float("nan")
        pn = float(np.mean([mn[i][1] for i in both])) if both else float("nan")
        verdict["no_worse"] &= bool(pn <= po)
        verdict["better"] |= bool(pn < po)
        verdict["good_kept"] &= new["good"] >= old["good"]
        print(f"  {net}: paired mean err {po:.2f} -> {pn:.2f} (n={len(both)}), good {old['good']} -> {new['good']}, "
              f"matches gained {len(set(mn) - set(mo))}, lost {len(set(mo) - set(mn))}")
    print(f"  distinct labelled serves whose match or contact estimate changed: {len(moved)}")
    accepted = len(moved) >= 3 and all(verdict.values())
    if len(moved) < 3:
        print("  -> fewer than 3 serves changed: this footage cannot decide; not adopted")
    else:
        print(f"  -> {'ACCEPTED' if accepted else 'REJECTED'} {verdict}")
    if {"v3", "v4"} <= set(nets):
        code = "new" if accepted else "old"
        g3, g4 = stats[f"v3_{code}"]["good"], stats[f"v4_{code}"]["good"]
        decision = "v3 becomes the default again" if g3 - g4 >= 2 else (
            "v4 stays" + (" (no measurable difference on serves)" if abs(g3 - g4) <= 1 else ""))
        print(f"\nrule 2 (v3 vs v4, code '{code}'): good serves v3 {g3}, v4 {g4} -> {decision}")


if __name__ == "__main__":
    main()
