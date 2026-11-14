"""Run and score the six-video test of docs/EVAL_SERVE_SPEED.md (written before any video was downloaded).

    .venv/bin/python scripts/eval_six_videos.py run   --videos DIR            # one GPU pass per video, then variants
    .venv/bin/python scripts/eval_six_videos.py score --labels DIR [--out F]  # both rules, once

``run``: for each DIR/<name>.mp4, the default analysis (perception once), then from that cache
``scaled`` (--fps-scaled-thresholds), ``scaled_pin`` (+ --pin-serve-height) and, reported only, ``bridge``
(--toss-bridge), under data/six/runs/<name>/. ``score``: labels DIR/<name>.json (as
results/serve_contact_labels_2025final.json: serves with lo, hi, side, radar); refuses to score a partial set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.eval.six_videos import fps_rule, pin_rule, recall_stats  # noqa: E402

RUNS = ROOT / "data" / "six" / "runs"
VARIANTS = {"scaled": ["--fps-scaled-thresholds"], "scaled_pin": ["--fps-scaled-thresholds", "--pin-serve-height"],
            "bridge": ["--toss-bridge"]}


def analyse(video: Path, out: Path, extra: list[str]) -> None:
    if (out / "report.json").exists():
        return
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(video),
                        "--out", str(out), "--no-render", *extra],
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{video.name} {out.name}: analysis failed: {r.stderr.strip().splitlines()[-1:]}")


def run(videos: Path) -> None:
    for v in sorted(videos.glob("*.mp4")):
        base = RUNS / v.stem / "default"
        analyse(v, base, [])  # the only GPU pass for this video
        for name, flags in VARIANTS.items():
            d = RUNS / v.stem / name
            d.mkdir(parents=True, exist_ok=True)
            for f in ("perception.pkl", "broadcast.pkl", "pose.pkl"):
                if (base / f).exists() and not (d / f).exists():
                    shutil.copy(base / f, d / f)
            analyse(v, d, ["--reuse-perception", *flags])
        print(f"{v.stem}: done", flush=True)


def score(labels_dir: Path, out: Path | None) -> None:
    names = sorted(p.stem for p in labels_dir.glob("*.json"))
    if not names:
        raise SystemExit(f"no label files in {labels_dir}")
    missing = [f"{n}/{k}" for n in names for k in ("default", *VARIANTS) if not (RUNS / n / k / "report.json").exists()]
    if missing:  # scoring a partial set would quietly change the test
        raise SystemExit(f"runs missing: {missing[:4]}")
    rep = {n: {k: json.loads((RUNS / n / k / "report.json").read_text()) for k in ("default", *VARIANTS)}
           for n in names}
    lab = {n: json.loads((labels_dir / f"{n}.json").read_text())["serves"] for n in names}
    fv = fps_rule([(float(rep[n]["default"]["video"]["fps"]), lab[n], rep[n]["default"], rep[n]["scaled"])
                   for n in names])
    pv = pin_rule([(lab[n], rep[n]["scaled"], rep[n]["scaled_pin"]) for n in names])
    per_video = {n: {k: recall_stats(lab[n], rep[n][k]) for k in ("default", *VARIANTS)} | {
        "fps": rep[n]["default"]["video"]["fps"]} for n in names}
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    result = {"code_sha": sha,
              "labels_sha256": {n: hashlib.sha256((labels_dir / f"{n}.json").read_bytes()).hexdigest() for n in names},
              "fps_scaled_thresholds": asdict(fv), "pin_serve_height": asdict(pv), "per_video": per_video}
    for name, v in (("--fps-scaled-thresholds", fv), ("--pin-serve-height", pv)):
        print(f"{name}: {v.decision.upper()}")
        for c, ok in v.criteria.items():
            print(f"   {'holds' if ok else 'FAILS'}  {c}")
        print(f"   {json.dumps(v.numbers, default=str)}")
    for n, st in per_video.items():
        print(f"{n}: {json.dumps(st)}")
    if out is not None:
        out.write_text(json.dumps(result, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score"))
    ap.add_argument("--videos", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.videos)
    else:
        score(a.labels, a.out)


if __name__ == "__main__":
    main()
