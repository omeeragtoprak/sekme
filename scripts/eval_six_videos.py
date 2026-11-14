"""Run and score the six-video test of docs/EVAL_SERVE_SPEED.md (written before any video was downloaded).

    .venv/bin/python scripts/eval_six_videos.py sample --videos DIR --labels DIR # frame lists for the presence labels
    .venv/bin/python scripts/eval_six_videos.py run    --videos DIR            # one GPU pass per video, then variants
    .venv/bin/python scripts/eval_six_videos.py disagreements --labels DIR    # frames where the player runs differ
    .venv/bin/python scripts/eval_six_videos.py score  --labels DIR --out F    # the rules, once

The videos are the six of ``six_videos_fetch.MATCHES``, files DIR/<name>.mp4 (handled three at a time: each
command acts on the videos present and keeps what earlier calls made).
``sample``: DIR/<name>_presence_frames.json, the frames to label for player presence (fixed by the name and frame
count; an existing list is kept, and a different one is refused).
``run``: for each video, the default analysis (perception once), then from that cache ``scaled``
(--fps-scaled-thresholds), ``scaled_pin`` (+ --pin-serve-height), ``players`` (--player-size-from-camera) and,
reported only, ``bridge`` (--toss-bridge) and ``bridge_pin`` (+ --pin-serve-height), under data/six/runs/<name>/.
Runs only from committed analysis code; each run records the commit it was made from.
``disagreements``: DIR/<name>_disagree_frames.json, frames where default and ``players`` pick different players.
``score``: serve labels DIR/<name>.json (as results/serve_contact_labels_2025final.json), player labels
DIR/<name>_presence.json and DIR/<name>_disagree.json ([{"frame", "main", "near": [x, y] | null, "far": ...}]).
Refuses: a set other than the six, missing runs, runs from other code than HEAD or with other flags, video or
models than expected, frame lists that are not the drawn ones, labels on other frames than listed, malformed
labels, uncommitted changes to the analysis or the scorer, and an existing result file.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from six_videos_fetch import MATCHES  # noqa: E402

from courtvision.eval.provenance import (  # noqa: E402
    analysis_code,
    clean,
    code_unchanged_since,
    head_sha,
    same_code,
    tracked_and_clean,
)
from courtvision.eval.six_videos import (  # noqa: E402
    call_changes,
    check_player_labels,
    check_serve_labels,
    disagreement_sample,
    fps_rule,
    pin_rule,
    player_rule,
    presence_sample,
    recall_stats,
    speed_report,
)

NAMES = sorted(MATCHES)
RUNS = ROOT / "data" / "six" / "runs"
MANIFEST = ROOT / "data" / "six" / "manifest.json"
BALL_MODEL = ROOT / "models" / "sekme_ball.pt"  # ball v4, the default the plan names
VARIANTS = {"scaled": ["--fps-scaled-thresholds"], "scaled_pin": ["--fps-scaled-thresholds", "--pin-serve-height"],
            "players": ["--player-size-from-camera"], "bridge": ["--toss-bridge"],
            "bridge_pin": ["--toss-bridge", "--pin-serve-height"]}
KINDS = ("default", *VARIANTS)
# The report's "candidates" record of each run: the flags it must have been made with, no more, no fewer.
FLAGS = {"default": set(), "scaled": {"fps_scaled_thresholds"},
         "scaled_pin": {"fps_scaled_thresholds", "pin_serve_height"}, "players": {"player_size_from_camera"},
         "bridge": {"serve_toss_bridge_s"}, "bridge_pin": {"serve_toss_bridge_s", "pin_serve_height"}}
CACHES = ("perception.pkl", "broadcast.pkl", "pose.pkl")
SCORER_CODE = ["src", "scripts/eval_six_videos.py", "scripts/six_videos_fetch.py"]
PLAYER_FILES = ("_presence", "_presence_frames", "_disagree", "_disagree_frames")


def videos_in(videos: Path) -> list[Path]:
    others = sorted(p.name for p in videos.glob("*.mp4") if p.stem not in MATCHES)
    if others:
        print(f"not among the six, ignored: {others}", flush=True)
    return [videos / f"{n}.mp4" for n in NAMES if (videos / f"{n}.mp4").exists()]


def write_once(path: Path, data: dict) -> None:
    """Frame lists are fixed once made (they are committed before labelling): keep an equal one, refuse another."""
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise SystemExit(f"{path.name} exists and differs from the one drawn now: not overwritten")
        print(f"{path.name}: kept (unchanged)", flush=True)
        return
    path.write_text(json.dumps(data))


def sample(videos: Path, labels_dir: Path) -> None:
    from courtvision.io.video import iter_frames

    labels_dir.mkdir(parents=True, exist_ok=True)
    for v in videos_in(videos):
        n = 0
        for i, _ in iter_frames(v):  # the pipeline's own reader: the labels' frame numbers are the analysis's
            n = i + 1
        frames = presence_sample(v.stem, n)
        write_once(labels_dir / f"{v.stem}_presence_frames.json", {"n_frames": n, "frames": frames})
        print(f"{v.stem}: {n} frames, {len(frames)} to label", flush=True)


def analyse(video: Path, out: Path, extra: list[str]) -> None:
    if (out / "report.json").exists():
        return  # its code_sha.txt says which code made it; score refuses runs from other code
    code = analysis_code()
    if not clean(code):
        raise SystemExit("the analysis code has uncommitted changes: commit them first (runs must name their code)")
    sha = head_sha()  # taken before the run: the code the process loads
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(video),
                        "--out", str(out), "--no-render", *extra],
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{video.name} {out.name}: analysis failed: {r.stderr.strip().splitlines()[-1:]}")
    if not code_unchanged_since(sha, code):
        (out / "report.json").unlink()  # which code it ran is unclear: it is made again on the next call
        raise SystemExit(f"{video.name} {out.name}: the analysis code changed during the run; run again")
    (out / "code_sha.txt").write_text(sha)


def run(videos: Path) -> None:
    for v in videos_in(videos):
        base = RUNS / v.stem / "default"
        analyse(v, base, [])  # the only GPU pass for this video
        for name, flags in VARIANTS.items():
            d = RUNS / v.stem / name
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "report.json").exists():
                for f in CACHES:
                    if (base / f).exists() and not (d / f).exists():
                        shutil.copy(base / f, d / f)
                analyse(v, d, ["--reuse-perception", *flags])
            for f in CACHES:  # the caches are kept once, in default/ (disk)
                (d / f).unlink(missing_ok=True)
        print(f"{v.stem}: done", flush=True)


def disagreements(labels_dir: Path) -> None:
    """After ``run``: frames where default and --player-size-from-camera pick different players, to be labelled
    like the presence frames (the labeller sees only the frames)."""
    for n in NAMES:
        da, db = RUNS / n / "default", RUNS / n / "players"
        if not ((da / "frames.json").exists() and (db / "frames.json").exists()):
            print(f"{n}: runs not made yet", flush=True)
            continue
        if not (same_code(da) and same_code(db)):
            raise SystemExit(f"{n}: runs made with other code than HEAD")
        fa, fb = json.loads((da / "frames.json").read_text()), json.loads((db / "frames.json").read_text())
        frames = disagreement_sample(n, fa, fb, fa["calib_valid"])
        write_once(labels_dir / f"{n}_disagree_frames.json", {"frames": frames})
        print(f"{n}: {len(frames)} disagreement frames to label", flush=True)


def label_names(labels_dir: Path) -> list[str]:
    """The videos: one serve-label file DIR/<name>.json each (the player label and frame-list files are not)."""
    return sorted(p.stem for p in labels_dir.glob("*.json") if not p.stem.endswith(PLAYER_FILES))


def check_provenance(n: str, rep: dict[str, dict], manifest: dict, ball_sha: str | None) -> list[str]:
    """Every run of a video: made with exactly its flags, from the downloaded file, with the same models (ball v4)."""
    problems = []
    want_video = manifest.get(n, {}).get("sha256")
    if want_video is None:
        problems.append(f"{n}: no SHA-256 in {MANIFEST.name}")
    models = {m: c.get("sha256") for m, c in rep["default"]["models"].items()}
    if ball_sha is not None and models.get("ball") != ball_sha:
        problems.append(f"{n}: the ball model is not {BALL_MODEL.name}")
    for k in KINDS:
        r = rep[k]
        got = set(r.get("candidates", {}))
        if got != FLAGS[k]:
            problems.append(f"{n}/{k}: made with {sorted(got)}, expected {sorted(FLAGS[k])}")
        if r["video"].get("sha256") != want_video:
            problems.append(f"{n}/{k}: video is not the downloaded file")
        if {m: c.get("sha256") for m, c in r["models"].items()} != models:
            problems.append(f"{n}/{k}: models differ from the default run's")
    return problems


def _check_frames(labels: list[dict], listed: Path, n: str) -> None:
    want = sorted(json.loads(listed.read_text())["frames"])
    got = sorted(lb["frame"] for lb in labels)
    if got != want:
        raise SystemExit(f"{n}: labelled frames differ from {listed.name}")


def committed(labels_dir: Path) -> bool:
    """Every label file is tracked by git and unchanged since its commit (labels are committed before scoring)."""
    return tracked_and_clean(sorted(labels_dir.glob("*.json")))


def score(labels_dir: Path, out: Path, require_committed: bool = True) -> None:
    names = label_names(labels_dir)
    if names != NAMES:  # a partial or different set would quietly change the test
        raise SystemExit(f"score needs the serve labels of exactly the six videos; missing "
                         f"{sorted(set(NAMES) - set(names))}, not among them {sorted(set(names) - set(NAMES))}")
    if out.exists():
        raise SystemExit(f"{out} exists: the test is scored once")
    if not clean(SCORER_CODE):
        raise SystemExit("uncommitted changes in src or the scorer: the result must name the code that made it")
    if require_committed and not committed(labels_dir):
        raise SystemExit(f"the labels in {labels_dir} are not all committed: commit them before scoring")
    missing = [f"{n}/{k}" for n in NAMES for k in KINDS if not (RUNS / n / k / "report.json").exists()]
    if missing:
        raise SystemExit(f"runs missing: {missing[:4]}")
    stale = [f"{n}/{k}" for n in NAMES for k in KINDS if not same_code(RUNS / n / k)]
    if stale:  # every run must come from the code being scored (label commits in between are fine)
        raise SystemExit(f"runs made with other code: {stale[:4]}")
    rep = {n: {k: json.loads((RUNS / n / k / "report.json").read_text()) for k in KINDS} for n in NAMES}
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if not BALL_MODEL.exists():
        raise SystemExit(f"{BALL_MODEL} missing: the runs cannot be checked to use ball v4")
    ball_sha = hashlib.sha256(BALL_MODEL.read_bytes()).hexdigest()
    problems = [p for n in NAMES for p in check_provenance(n, rep[n], manifest, ball_sha)]
    first = {m: c.get("sha256") for m, c in rep[NAMES[0]]["default"]["models"].items()}
    problems += [f"{n}: models differ from {NAMES[0]}'s" for n in NAMES[1:]
                 if {m: c.get("sha256") for m, c in rep[n]["default"]["models"].items()} != first]
    if problems:
        raise SystemExit("\n".join(problems[:8]))
    lab = {}
    for n in NAMES:
        lab[n] = json.loads((labels_dir / f"{n}.json").read_text())["serves"]
        try:
            check_serve_labels(lab[n])
        except ValueError as e:
            raise SystemExit(f"{n}: {e}") from None
    fv = fps_rule([(float(rep[n]["default"]["video"]["fps"]), lab[n], rep[n]["default"], rep[n]["scaled"])
                   for n in NAMES])
    pv = pin_rule([(lab[n], rep[n]["scaled"], rep[n]["scaled_pin"]) for n in NAMES])
    pres = {n: labels_dir / f"{n}_presence.json" for n in NAMES}
    dis = {n: labels_dir / f"{n}_disagree.json" for n in NAMES}
    have = [p.exists() for p in [*pres.values(), *dis.values()]]
    if any(have) and not all(have):  # partial player labels would quietly change the test
        raise SystemExit("presence or disagreement labels missing for some videos")
    plv, outside_main = None, {}
    if all(have):
        fr = {n: {k: json.loads((RUNS / n / k / "frames.json").read_text()) for k in ("default", "players")}
              for n in NAMES}
        pl, dl = {}, {}
        for n in NAMES:
            pl[n], dl[n] = json.loads(pres[n].read_text()), json.loads(dis[n].read_text())
            try:
                check_player_labels(pl[n])
                check_player_labels(dl[n])
            except ValueError as e:
                raise SystemExit(f"{n}: {e}") from None
            listed = json.loads((labels_dir / f"{n}_presence_frames.json").read_text())
            n_frames = int(rep[n]["default"]["video"]["n_frames"])
            if listed["n_frames"] != n_frames or listed["frames"] != presence_sample(n, n_frames):
                raise SystemExit(f"{n}: the presence frame list is not the one drawn for its {n_frames} frames")
            fa, fb = fr[n]["default"], fr[n]["players"]
            drawn = disagreement_sample(n, fa, fb, fa["calib_valid"])
            if json.loads((labels_dir / f"{n}_disagree_frames.json").read_text())["frames"] != drawn:
                raise SystemExit(f"{n}: the disagreement frame list is not the one drawn from the runs")
            _check_frames(pl[n], labels_dir / f"{n}_presence_frames.json", n)
            _check_frames(dl[n], labels_dir / f"{n}_disagree_frames.json", n)
            outside_main[n] = sum(not lb["main"] for lb in pl[n])
        plv = player_rule([(pl[n], dl[n], fr[n]["default"], fr[n]["players"], int(rep[n]["default"]["video"]["height"]))
                           for n in NAMES])
    per_video = {n: {k: recall_stats(lab[n], rep[n][k]) for k in KINDS} | {
        "fps": rep[n]["default"]["video"]["fps"]} for n in NAMES}
    reported = {  # not used by any rule (docs/EVAL_SERVE_SPEED.md, "Reported, not used")
        "fps_scaled_thresholds": {
            "ball_detected_frames": {k: sum(rep[n][k]["ball"]["detected_frames"] for n in NAMES)
                                     for k in ("default", "scaled")},
            "calls": call_changes([(rep[n]["default"], rep[n]["scaled"]) for n in NAMES]),
            "speeds": {k: speed_report([(lab[n], rep[n][k]) for n in NAMES]) for k in ("default", "scaled")}},
        "pin_serve_height": {
            "calls": call_changes([(rep[n]["scaled"], rep[n]["scaled_pin"]) for n in NAMES]),
            "speeds": {k: speed_report([(lab[n], rep[n][k]) for n in NAMES])
                       for k in ("scaled", "scaled_pin", "bridge", "bridge_pin")},
            "with_toss_bridge": asdict(pin_rule([(lab[n], rep[n]["bridge"], rep[n]["bridge_pin"]) for n in NAMES],
                                                min_paired=0))},
        "player_size_from_camera": {
            "calls": call_changes([(rep[n]["default"], rep[n]["players"]) for n in NAMES]),
            "speeds": speed_report([(lab[n], rep[n]["players"]) for n in NAMES]),
            "labelled_frames_outside_main_camera": outside_main or "no presence labels"}}
    result = {"code_sha": head_sha(),
              "labels_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(labels_dir.glob("*.json"))},
              "fps_scaled_thresholds": asdict(fv), "pin_serve_height": asdict(pv),
              "player_size_from_camera": asdict(plv) if plv else "no presence labels", "per_video": per_video,
              "reported_not_used": reported}
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
        if a.out is None:
            ap.error("score needs --out (the result file, written once)")
        score(a.labels, a.out)


if __name__ == "__main__":
    main()
