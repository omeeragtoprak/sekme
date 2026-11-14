"""Score the pipeline's events (bounces, strokes, serves) on the E2E-Spot tennis labels.

    .venv/bin/python scripts/eval_spot.py sample --name dev3|test4
    .venv/bin/python scripts/eval_spot.py run    --name dev3 [--runs-dir runs_spot_dev] [--candidate-flag F ...]
    .venv/bin/python scripts/eval_spot.py run    --name test4 [--slot N] [--delete-videos]
    .venv/bin/python scripts/eval_spot.py score  --name dev3|test4 [--runs-dir ...] [--slot N]

Labels: data/spot/tennis/{train,val,test}.json from github.com/jhong93/spot (data/tennis, BSD-3-Clause).
Videos: one file per match, data/spot/videos/<match>.mp4 (YouTube ids in that repo's videos.csv), 720p.

The samples are the matches of docs/EVAL_SPOT.md's table, every clip of each (``dev3``: train and val clips,
``test4``: test clips); their clip counts are checked against the table and a sample file is never redrawn.
The labels number frames as OpenCV decodes the full match video (the repo's frames_as_jpg.py), so ``run``
decodes each match once, in order, with OpenCV and writes each clip's frames itself: frame ``start`` of the
clip is frame ``start`` of the decode by construction, with no seeking. It stops if the video's frame rate
differs from the labels' (another upload) or the video ends before the last clip. Each clip is analysed as soon
as it is written and then deleted (to keep disk use small). Each match video's SHA-256 and size are recorded
before its first clip is cut (data/spot/videos_manifest.json).

Test matches: ``run --name test4`` is the baseline report (runs_spot, decides nothing). A candidate runs only as
an entry of the committed ledger results/spot_test_candidates.json (at most two entries, ever; its rule written
in docs/EVAL_SPOT.md first): ``run --name test4 --slot N`` analyses each clip with the current default
(runs_spot_base_slot<N>) and then the candidate from that run's cached detections (runs_spot_cand_slot<N>).
Runs only from committed analysis code; every clip run records its commit. A clip whose analysis raises is
recorded as failed and scored as a clip with no system events (docs/EVAL_SPOT.md); any other exit (a usage error,
a killed process) stops the run, as do three failures in a row or more than 10 % in a match, before the match
video is deleted.
``score`` checks every run (commit, flags, frame count, models, video manifest) and, on the test matches, writes
its result once. Nothing here downloads anything. Method and rules: docs/EVAL_SPOT.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.eval.provenance import (  # noqa: E402
    analysis_code,
    clean,
    code_unchanged_since,
    head_sha,
    same_code,
    sha256_file,
    tracked_and_clean,
)
from courtvision.eval.spot import TOL_S, check_clip, match_name, per_match, score_clip, summary  # noqa: E402

DATA = ROOT / "data" / "spot"
# docs/EVAL_SPOT.md, "Matches": every clip of each match (dev: train and val clips together).
SAMPLES = {
    "dev3": {"splits": ["train", "val"],
             "matches": {"usopen_2019_womens_round1_williams_sharapova": 97,
                         "usopen_2021_womens_semifinal_raducanu_sakkari": 140,
                         "wimbledon_2019_womens_round1_gauff_williams": 142}},
    "test4": {"splits": ["test"],
              "matches": {"wimbledon_2019_womens_final_halep_williams": 64,
                          "usopen_2019_womens_osaka_gauff": 70,
                          "usopen_2019_womens_final_andreescu_williams": 91,
                          "wimbledon_2018_womens_quarterfinal_williams_giorgi": 78}},
}
TEST = "test4"
TEST_BASELINE_RUNS = "runs_spot"
LEDGER = ROOT / "results" / "spot_test_candidates.json"  # committed; at most two candidates ever
MAX_CANDIDATES = 2
VIDEO_MANIFEST = DATA / "videos_manifest.json"
VIDEO_HEIGHT = 720
MAX_FAILED_IN_ROW = 3
MAX_FAILED_SHARE = 0.10
# CLI flag -> the report's "candidates" record of a run made with it.
FLAG_KEYS = {"toss-bridge": "serve_toss_bridge_s", "pin-serve-height": "pin_serve_height",
             "fps-scaled-thresholds": "fps_scaled_thresholds", "player-size-from-camera": "player_size_from_camera"}
SCORER_CODE = ["src", "scripts/eval_spot.py"]
_CODE: list[str] = []  # the analysis code, fixed when a run starts


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def clip_frames(video: str) -> tuple[int, int]:
    start, end = video.rsplit("_", 2)[1:]
    return int(start), int(end)


def analyse(clip_path: Path, run: Path, extra: tuple[str, ...] = ()) -> None:
    sha = head_sha()  # taken before the run: the code the process loads
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(clip_path),
                        "--out", str(run), "--no-render", "--no-broadcast", *extra],
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):  # a usage error (2) or a killed process (< 0): the environment, not the clip
        raise SystemExit(f"{run.name}: analysis exited with {r.returncode} ({r.stderr.strip().splitlines()[-1:]}); "
                         "nothing recorded, the run stops")
    run.mkdir(parents=True, exist_ok=True)
    if _CODE and not code_unchanged_since(sha, _CODE):
        (run / "report.json").unlink(missing_ok=True)
        raise SystemExit(f"{run.name}: the analysis code changed during the run; nothing recorded")
    if r.returncode == 1:  # the analysis raised on this clip: recorded, scored as no system events
        (run / "FAILED.txt").write_text(r.stderr[-4000:])
        print(f"  {run.name}: analysis failed: {r.stderr.strip().splitlines()[-1:]}", flush=True)
    (run / "code_sha.txt").write_text(sha)


def analyse_candidate(clip_path: Path, base: Path, run: Path, flags: list[str]) -> None:
    """A candidate on the baseline's cached detections of the same clip file: the two runs differ only by
    the candidate's code path, and no second GPU pass is needed."""
    run.mkdir(parents=True, exist_ok=True)
    if not (base / "perception.pkl").exists():
        (run / "FAILED.txt").write_text("baseline analysis failed: no cached perception")
        (run / "code_sha.txt").write_text(head_sha())
        return
    for f in ("perception.pkl", "pose.pkl"):
        if (base / f).exists():
            shutil.copy(base / f, run / f)
    analyse(clip_path, run, ("--reuse-perception", *(f"--{x}" for x in flags)))


def _done(run: Path) -> bool:
    return (run / "report.json").exists() or (run / "FAILED.txt").exists()


def run_match(src: Path, clips: list[dict], runs: Path, tmp: Path, keep: bool,
              cand: tuple[Path, list[str]] | None = None) -> None:
    """Cut and analyse every clip not done yet; with ``cand`` = (runs dir, CLI flags) also the candidate,
    right after the baseline and before the clip is deleted."""
    def done(name: str) -> bool:
        return _done(runs / name) and (cand is None or _done(cand[0] / name))

    todo = sorted((c for c in clips if not done(c["video"])), key=lambda c: clip_frames(c["video"])[0])
    if not todo:
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not all(np.isclose(fps, c["fps"]) for c in todo):
        raise SystemExit(f"{src.name}: {fps} fps, labels say {todo[0]['fps']}: not the labelled upload")
    open_: dict[str, tuple[subprocess.Popen, int, Path]] = {}
    k, i, failed_in_row = 0, 0, 0
    while k < len(todo) or open_:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"{src.name} ended at frame {i}, before {[*open_][:1] or todo[k]['video']}")
        while k < len(todo) and clip_frames(todo[k]["video"])[0] == i:
            c = todo[k]
            out = tmp / f"{c['video']}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                  "-s", f"{w}x{h}", "-r", f"{c['fps']:.6f}", "-i", "-", "-c:v", "libx264",
                                  "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", str(out)],
                                 stdin=subprocess.PIPE)
            open_[c["video"]] = (p, 0, out)
            k += 1
        for name, (p, n, out) in list(open_.items()):
            p.stdin.write(frame.tobytes())
            open_[name] = (p, n + 1, out)
            if n + 1 == clip_frames(name)[1] - clip_frames(name)[0]:
                p.stdin.close()
                if p.wait() != 0:
                    raise SystemExit(f"ffmpeg failed writing {out}")
                del open_[name]
                got = int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT))
                if got != n + 1:
                    raise SystemExit(f"{out.name}: wrote {n + 1} frames, the file holds {got}")
                if not _done(runs / name):
                    analyse(out, runs / name)
                    failed_in_row = failed_in_row + 1 if (runs / name / "FAILED.txt").exists() else 0
                    if failed_in_row >= MAX_FAILED_IN_ROW:  # systematic, not a property of the clips
                        raise SystemExit(f"{MAX_FAILED_IN_ROW} analyses failed in a row (last {name}): stopped, "
                                         f"video kept; see the FAILED.txt files")
                if cand is not None and not _done(cand[0] / name):
                    analyse_candidate(out, runs / name, cand[0] / name, cand[1])
                if not keep:
                    out.unlink()
        i += 1
    cap.release()


def check_flags(flags: list[str]) -> None:
    unknown = [f for f in flags if f not in FLAG_KEYS]
    if unknown:
        raise SystemExit(f"unknown candidate flag(s) {unknown}; known: {sorted(FLAG_KEYS)}")


def ledger() -> list[dict]:
    """The committed list of candidates run on the test matches (their rules are in docs/EVAL_SPOT.md)."""
    if not LEDGER.exists():
        return []
    if not tracked_and_clean([LEDGER]):
        raise SystemExit(f"{_rel(LEDGER)} has uncommitted changes: commit the candidate (and its rule)")
    entries = json.loads(LEDGER.read_text())["candidates"]
    if len(entries) > MAX_CANDIDATES:
        raise SystemExit(f"{len(entries)} candidates in the ledger: the test matches allow {MAX_CANDIDATES}, ever")
    if sorted(e["slot"] for e in entries) != list(range(1, len(entries) + 1)):
        raise SystemExit("ledger slots must be 1, 2")
    for e in entries:
        check_flags(e["flags"])
    return entries


def candidate(slot: int) -> list[str]:
    """The flags of the candidate in this test slot (1 or 2) of the committed ledger."""
    e = next((e for e in ledger() if e["slot"] == slot), None)
    if e is None:
        raise SystemExit(f"slot {slot} is not in the committed ledger {_rel(LEDGER)}")
    return list(e["flags"])


def record_video(m: str, src: Path) -> None:
    """The match video's identity, before any clip is cut from it (the video is deleted afterwards)."""
    cap = cv2.VideoCapture(str(src))
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(
        cv2.CAP_PROP_FPS)
    cap.release()
    if h != VIDEO_HEIGHT:
        raise SystemExit(f"{m}: {w}x{h}; the plan uses the {VIDEO_HEIGHT}p upload")
    man = json.loads(VIDEO_MANIFEST.read_text()) if VIDEO_MANIFEST.exists() else {}
    sha = sha256_file(src)
    if m in man and man[m]["sha256"] != sha:
        raise SystemExit(f"{m}: not the file recorded before ({man[m]['sha256'][:12]}): another upload")
    man[m] = {"sha256": sha, "width": w, "height": h, "fps": fps}
    VIDEO_MANIFEST.write_text(json.dumps(man, indent=1))


def make_sample(name: str) -> Path:
    spec = SAMPLES[name]
    clips = [c for s in spec["splits"] for c in json.loads((DATA / "tennis" / f"{s}.json").read_text())
             if match_name(c["video"]) in spec["matches"]]
    counts = {m: sum(match_name(c["video"]) == m for c in clips) for m in spec["matches"]}
    if counts != spec["matches"]:
        raise SystemExit(f"clip counts {counts} differ from the plan's table {spec['matches']}")
    for c in clips:
        try:
            check_clip(c)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    labels_sha = {s: sha256_file(DATA / "tennis" / f"{s}.json") for s in spec["splits"]}
    data = {"split": spec["splits"], "matches": sorted(spec["matches"]), "labels_sha256": labels_sha, "clips": clips}
    path = DATA / f"sample_{name}.json"
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise SystemExit(f"{path.name} exists and differs from the one drawn now: not overwritten")
        print(f"{path.name}: kept (unchanged)")
        return path
    path.write_text(json.dumps(data, indent=1))
    print(f"{len(clips)} clips from {len(spec['matches'])} matches -> {path}")
    return path


def check_runs(sample: dict, runs: Path, flags: list[str]) -> tuple[list[str], dict]:
    """Every clip run: made from the analysis code at HEAD, with exactly these flags, on the clip's frames, with
    the same models as the others; every match in the video manifest at 720p."""
    problems, models = [], {}
    want = {FLAG_KEYS[f] for f in flags}
    for c in sample["clips"]:
        run = runs / c["video"]
        if not same_code(run):
            problems.append(f"{runs.name}/{c['video']}: made with other code than HEAD (or no code_sha.txt)")
        if not (run / "report.json").exists():
            continue
        rep = json.loads((run / "report.json").read_text())
        got = set(rep.get("candidates", {}))
        if got != want:
            problems.append(f"{runs.name}/{c['video']}: made with {sorted(got)}, expected {sorted(want)}")
        if rep["video"].get("n_frames") != c["num_frames"]:
            problems.append(f"{runs.name}/{c['video']}: {rep['video'].get('n_frames')} frames, "
                            f"labels {c['num_frames']}")
        m = {k: v.get("sha256") for k, v in rep.get("models", {}).items()}
        if not models:
            models = m
        elif m != models:
            problems.append(f"{runs.name}/{c['video']}: other models than the first clip's")
    man = json.loads(VIDEO_MANIFEST.read_text()) if VIDEO_MANIFEST.exists() else {}
    for mname in sample["matches"]:
        if mname not in man:
            problems.append(f"{mname}: no SHA-256 in {VIDEO_MANIFEST.name}")
        elif man[mname]["height"] != VIDEO_HEIGHT:
            problems.append(f"{mname}: {man[mname]['height']}p, not {VIDEO_HEIGHT}p")
    return problems, models


def score_runs(sample: dict, runs: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for c in sample["clips"]:
        run = runs / c["video"]
        if (run / "report.json").exists():
            rows.append(score_clip(c, json.loads((run / "report.json").read_text())))
        elif (run / "FAILED.txt").exists():
            rows.append(score_clip(c, None))
        else:
            missing.append(c["video"])
    return rows, missing


def compare(base: dict, cand: dict, pm_base: dict, pm_cand: dict) -> dict:
    """Baseline against candidate on the measures the candidate rules use (docs/EVAL_SPOT.md, rule 2 and the
    candidate's own rule), pooled and per match, in points, with the matched counts. The decision is the
    candidate's written rule, applied to these numbers."""
    keys = {"serve_recall": ("serve", "recall"), "serve_precision_excl_faults": ("serve", "precision_excl_faults"),
            "first_bounce_recall": ("bounce", "first_recall"), "bounce_precision": ("bounce", "precision")}

    def one(b: dict, c: dict) -> dict:
        out = {k: {"baseline": b[a][f], "candidate": c[a][f], "change_points": 100 * (c[a][f] - b[a][f])}
               for k, (a, f) in keys.items()}
        out["serve_matched"] = [b["serve"]["matched"], c["serve"]["matched"]]
        out["serve_labelled"] = b["serve"]["labelled"]
        return out

    return {"pooled": one(base, cand), "per_match": {m: one(pm_base[m], pm_cand[m]) for m in pm_base}}


def line(name: str, k: dict) -> str:
    extra = (f"  precision without likely faults {k['precision_excl_faults']:.3f} ({k['before_label']} set apart)"
             if "precision_excl_faults" in k else "")
    return (f"{name:10s} labelled {k['labelled']:5d}  system {k['system']:5d}  matched {k['matched']:5d}  "
            f"recall {k['recall']:.3f}  precision {k['precision']:.3f}  F1 {k['f1']:.3f}  side {k['same_side']:.3f}  "
            f"median dt {k['median_dt_frames']:+.1f} fr" + extra)


def report(title: str, rows: list[dict]) -> tuple[dict, dict]:
    s = summary(rows)
    print(f"{title}: {s['clips']} clips, {s['failed']} failed analyses, {s['events_per_clip']:.1f} labelled "
          f"bounces+strokes per clip, calibrated frames {s['calibrated_share']:.1%}")
    for kind in TOL_S:
        print(line(kind, s[kind]))
    b = s["bounce"]
    print(f"first bounces {b['first_labelled']}: recall {b['first_recall']:.3f}; recall at "
          + ", ".join(f"{t} s {v:.3f}" for t, v in b["recall_sweep"].items()) + " (bounces, descriptive)")
    print("per match (the spread across matches is the uncertainty):")
    pm = per_match(rows)
    for m, v in pm.items():
        print(f"  {m:55s} clips {v['clips']:4d} ev/clip {v['events_per_clip']:.1f} cal {v['calibrated_share']:.0%}"
              f"  bounce R {v['bounce']['recall']:.3f} P {v['bounce']['precision']:.3f}"
              f"  first R {v['bounce']['first_recall']:.3f}  stroke R {v['stroke']['recall']:.3f}"
              f"  serve R {v['serve']['recall']:.3f} ({v['serve']['matched']}/{v['serve']['labelled']})"
              f" P {v['serve']['precision']:.3f}")
    return s, pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "run", "score"))
    ap.add_argument("--name", required=True, choices=sorted(SAMPLES), help="sample: data/spot/sample_<name>.json")
    ap.add_argument("--videos", default=str(DATA / "videos"))
    ap.add_argument("--runs-dir", help=f"dev runs (default runs_spot_dev); the test baseline is {TEST_BASELINE_RUNS}")
    ap.add_argument("--keep-clips", action="store_true")
    ap.add_argument("--candidate-flag", action="append", default=[], metavar="FLAG",
                    help="dev only: also run the candidate (sekme analyze --FLAG) on each clip from the baseline's "
                         "cached detections, e.g. toss-bridge (repeatable)")
    ap.add_argument("--candidate-runs-dir", help="dev only: where the candidate's runs go (default: <runs-dir>_cand)")
    ap.add_argument("--slot", type=int, help="test only: the ledger's candidate slot (1 or 2) to run or score")
    ap.add_argument("--delete-videos", action="store_true",
                    help="delete a match video once every one of its clips has a report or failure record")
    args = ap.parse_args()

    if args.cmd == "sample":  # every clip of the named matches, fixed before any system output is seen
        make_sample(args.name)
        return

    test = args.name == TEST
    if test and (args.runs_dir or args.candidate_flag or args.candidate_runs_dir):
        raise SystemExit("on the test matches the runs directories and flags are fixed: use --slot N")
    if not test and args.slot is not None:
        raise SystemExit("--slot is for the test matches; on dev use --candidate-flag")
    check_flags(args.candidate_flag)
    if test and args.slot is not None:
        flags = candidate(args.slot)
        base_dir, cand_dir = f"runs_spot_base_slot{args.slot}", f"runs_spot_cand_slot{args.slot}"
    elif test:
        flags, base_dir, cand_dir = [], TEST_BASELINE_RUNS, None
    else:
        flags = args.candidate_flag
        base_dir = args.runs_dir or "runs_spot_dev"
        cand_dir = (args.candidate_runs_dir or f"{base_dir}_cand") if flags else None
    sample_path = DATA / f"sample_{args.name}.json"
    sample = json.loads(sample_path.read_text())
    runs, cand_runs = DATA / base_dir, (DATA / cand_dir if cand_dir else None)

    if args.cmd == "run":  # one clip analysed at a time: one GPU job
        _CODE[:] = analysis_code()
        if not clean(_CODE):
            raise SystemExit("the analysis code has uncommitted changes: commit them first (runs must name their code)")
        for m in sample["matches"]:
            src = Path(args.videos) / f"{m}.mp4"
            if not src.exists():
                print(f"{m}: no video yet, skipped", flush=True)
                continue
            record_video(m, src)
            clips = [c for c in sample["clips"] if match_name(c["video"]) == m]
            run_match(src, clips, runs, DATA / "clips_tmp", args.keep_clips, (cand_runs, flags) if cand_runs else None)
            failed = sum((runs / c["video"] / "FAILED.txt").exists() for c in clips)
            if failed > MAX_FAILED_SHARE * len(clips):
                raise SystemExit(f"{m}: {failed} of {len(clips)} analyses failed (more than "
                                 f"{MAX_FAILED_SHARE:.0%}): stopped, video kept; see the FAILED.txt files")
            if args.delete_videos:  # run_match returns only when every clip of the match is done
                src.unlink()
        return

    # score
    out = DATA / (f"results_{args.name}_{base_dir}.json" if cand_dir is None or not test
                  else f"results_{args.name}_slot{args.slot}.json")
    if test and out.exists():
        raise SystemExit(f"{out.name} exists: a test result is written once")
    if test and not clean(SCORER_CODE):
        raise SystemExit("uncommitted changes in src or the scorer: the result must name the code that made it")
    results: dict = {}
    problems: list[str] = []
    for label, d, f in (("baseline", runs, []), ("candidate", cand_runs, flags)):
        if d is None:
            continue
        rows, missing = score_runs(sample, d)
        if missing:  # scoring a partial set would quietly change the sample
            raise SystemExit(f"{d.name}: {len(missing)} clips not run yet, e.g. {missing[:3]}")
        pr, models = check_runs(sample, d, f)
        problems += pr
        s, pm = report(f"{args.name} {label} ({d.name})", rows)
        results[label] = {"runs_dir": d.name, "flags": f, "models": models, "summary": s, "per_match": pm,
                          "clips": rows}
    if "candidate" in results and results["candidate"]["models"] != results["baseline"]["models"]:
        problems.append("baseline and candidate runs used different models")
    if problems:
        if test:
            raise SystemExit("\n".join(problems[:8]))
        print(f"WARNING (dev): {len(problems)} provenance problems, e.g. {problems[:3]}")
    if "candidate" in results:
        b, c = results["baseline"], results["candidate"]
        results["comparison"] = compare(b["summary"], c["summary"], b["per_match"], c["per_match"])
    man = json.loads(VIDEO_MANIFEST.read_text()) if VIDEO_MANIFEST.exists() else {}
    results["provenance"] = {
        "code_sha": head_sha(), "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "labels_sha256": sample.get("labels_sha256"), "videos": {m: man.get(m) for m in sample["matches"]},
        "tolerances_s": TOL_S, "tolerance_rate": "nominal (29.97 fps counted as 30)", "problems": problems}
    out.write_text(json.dumps(results))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
