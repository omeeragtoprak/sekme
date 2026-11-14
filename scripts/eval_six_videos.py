"""Run and score the six-video test of docs/EVAL_SERVE_SPEED.md (written before any video was downloaded).

    .venv/bin/python scripts/eval_six_videos.py sample --videos DIR --labels DIR # frame lists for the presence labels
    .venv/bin/python scripts/eval_six_videos.py run    --videos DIR            # one GPU pass per video, then variants
    .venv/bin/python scripts/eval_six_videos.py disagreements --labels DIR    # frames where the player runs differ
    .venv/bin/python scripts/eval_six_videos.py score  --labels DIR [--out F]  # the three rules, once

``sample``: DIR/<name>_presence_frames.json, the frames to label for player presence (fixed by the name).
``run``: for each DIR/<name>.mp4, the default analysis (perception once), then from that cache
``scaled`` (--fps-scaled-thresholds), ``scaled_pin`` (+ --pin-serve-height), ``players``
(--player-size-from-camera) and, reported only, ``bridge`` (--toss-bridge), under data/six/runs/<name>/.
``disagreements``: DIR/<name>_disagree_frames.json, frames where default and ``players`` pick different players.
``score``: serve labels DIR/<name>.json (as results/serve_contact_labels_2025final.json), player labels
DIR/<name>_presence.json and DIR/<name>_disagree.json ([{"frame", "main", "near": [x, y] | null, "far": ...}]);
refuses partial sets, labels on other frames than listed, and runs made with other code than HEAD.
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

from courtvision.eval.six_videos import (  # noqa: E402
    disagreement_sample,
    fps_rule,
    pin_rule,
    player_rule,
    presence_sample,
    recall_stats,
)

RUNS = ROOT / "data" / "six" / "runs"
VARIANTS = {"scaled": ["--fps-scaled-thresholds"], "scaled_pin": ["--fps-scaled-thresholds", "--pin-serve-height"],
            "players": ["--player-size-from-camera"], "bridge": ["--toss-bridge"]}


def sample(videos: Path, labels_dir: Path) -> None:
    import cv2

    labels_dir.mkdir(parents=True, exist_ok=True)
    for v in sorted(videos.glob("*.mp4")):
        cap, n = cv2.VideoCapture(str(v)), 0
        while cap.grab():  # count as the pipeline decodes: in order, no seeking
            n += 1
        frames = presence_sample(v.stem, n)
        (labels_dir / f"{v.stem}_presence_frames.json").write_text(json.dumps({"n_frames": n, "frames": frames}))
        print(f"{v.stem}: {n} frames, {len(frames)} to label", flush=True)


def head_sha() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def same_code(run: Path) -> bool:
    """The run was made from a commit whose src/ equals HEAD's."""
    marker = run / "code_sha.txt"
    if not marker.exists():
        return False
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", marker.read_text().strip(), "HEAD", "--", "src"])
    return r.returncode == 0


def analyse(video: Path, out: Path, extra: list[str]) -> None:
    if (out / "report.json").exists():
        return  # its code_sha.txt says which code made it; score refuses runs from other code
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(video),
                        "--out", str(out), "--no-render", *extra],
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{video.name} {out.name}: analysis failed: {r.stderr.strip().splitlines()[-1:]}")
    (out / "code_sha.txt").write_text(head_sha())


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


def disagreements(labels_dir: Path) -> None:
    """After ``run``: frames where default and --player-size-from-camera pick different players, to be labelled
    like the presence frames (the labeller sees only the frames)."""
    for d in sorted(RUNS.iterdir()):
        fa = json.loads((d / "default" / "frames.json").read_text())
        fb = json.loads((d / "players" / "frames.json").read_text())
        frames = disagreement_sample(d.name, fa, fb, fa["calib_valid"])
        (labels_dir / f"{d.name}_disagree_frames.json").write_text(json.dumps({"frames": frames}))
        print(f"{d.name}: {len(frames)} disagreement frames to label", flush=True)


def _check_frames(labels: list[dict], listed: Path, n: str) -> None:
    want = sorted(json.loads(listed.read_text())["frames"])
    got = sorted(lb["frame"] for lb in labels)
    if got != want:
        raise SystemExit(f"{n}: labelled frames differ from {listed.name}")


def score(labels_dir: Path, out: Path | None) -> None:
    names = sorted(p.stem for p in labels_dir.glob("*.json") if not p.stem.endswith(("_presence", "_presence_frames")))
    if not names:
        raise SystemExit(f"no label files in {labels_dir}")
    missing = [f"{n}/{k}" for n in names for k in ("default", *VARIANTS) if not (RUNS / n / k / "report.json").exists()]
    if missing:  # scoring a partial set would quietly change the test
        raise SystemExit(f"runs missing: {missing[:4]}")
    stale = [f"{n}/{k}" for n in names for k in ("default", *VARIANTS) if not same_code(RUNS / n / k)]
    if stale:  # every run must come from the code being scored (label commits in between are fine)
        raise SystemExit(f"runs made with other code: {stale[:4]}")
    rep = {n: {k: json.loads((RUNS / n / k / "report.json").read_text()) for k in ("default", *VARIANTS)}
           for n in names}
    lab = {n: json.loads((labels_dir / f"{n}.json").read_text())["serves"] for n in names}
    fv = fps_rule([(float(rep[n]["default"]["video"]["fps"]), lab[n], rep[n]["default"], rep[n]["scaled"])
                   for n in names])
    pv = pin_rule([(lab[n], rep[n]["scaled"], rep[n]["scaled_pin"]) for n in names])
    pres = {n: labels_dir / f"{n}_presence.json" for n in names}
    dis = {n: labels_dir / f"{n}_disagree.json" for n in names}
    have = [p.exists() for p in [*pres.values(), *dis.values()]]
    if any(have) and not all(have):  # partial player labels would quietly change the test
        raise SystemExit("presence or disagreement labels missing for some videos")
    plv = None
    if all(have):
        fr = {n: {k: json.loads((RUNS / n / k / "frames.json").read_text()) for k in ("default", "players")}
              for n in names}
        pl, dl = {}, {}
        for n in names:
            pl[n], dl[n] = json.loads(pres[n].read_text()), json.loads(dis[n].read_text())
            _check_frames(pl[n], labels_dir / f"{n}_presence_frames.json", n)
            _check_frames(dl[n], labels_dir / f"{n}_disagree_frames.json", n)
        plv = player_rule([(pl[n], dl[n], fr[n]["default"], fr[n]["players"], int(rep[n]["default"]["video"]["height"]))
                           for n in names])
    per_video = {n: {k: recall_stats(lab[n], rep[n][k]) for k in ("default", *VARIANTS)} | {
        "fps": rep[n]["default"]["video"]["fps"]} for n in names}
    result = {"code_sha": head_sha(),
              "labels_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(labels_dir.glob("*.json"))},
              "fps_scaled_thresholds": asdict(fv), "pin_serve_height": asdict(pv),
              "player_size_from_camera": asdict(plv) if plv else "no presence labels", "per_video": per_video}
    if plv is None:
        print("--player-size-from-camera: no player labels yet, not scored")
    for name, v in (("--fps-scaled-thresholds", fv), ("--pin-serve-height", pv),
                    ("--player-size-from-camera", plv)):
        if v is None:
            continue
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
    ap.add_argument("cmd", choices=("sample", "run", "disagreements", "score"))
    ap.add_argument("--videos", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.videos, a.labels)
    elif a.cmd == "run":
        run(a.videos)
    elif a.cmd == "disagreements":
        disagreements(a.labels)
    else:
        score(a.labels, a.out)


if __name__ == "__main__":
    main()
