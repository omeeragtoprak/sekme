"""Score the pipeline's events (bounces, strokes, serves) on the E2E-Spot tennis labels.

    .venv/bin/python scripts/eval_spot.py sample --name test4 --split test --matches M1 M2 ...
    .venv/bin/python scripts/eval_spot.py run    --name test4 [--runs-dir runs_spot] [--keep-clips]
    .venv/bin/python scripts/eval_spot.py score  --name test4 [--runs-dir runs_spot]

Labels: data/spot/tennis/{train,val,test}.json from github.com/jhong93/spot (data/tennis, BSD-3-Clause).
Videos: one file per match, data/spot/videos/<match>.mp4 (YouTube ids in that repo's videos.csv).

The labels number frames as OpenCV decodes the full match video (the repo's frames_as_jpg.py), so
``run`` decodes each match once, in order, with OpenCV and writes each clip's frames itself: frame
``start`` of the clip is frame ``start`` of the decode by construction, with no seeking. It stops if
the video's frame rate differs from the labels' (another upload) or the video ends before the last
clip. Each clip is analysed as soon as it is written and then deleted (to keep disk use small).
Nothing here downloads anything. Method and rules: docs/EVAL_SPOT.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.eval.spot import TOL_S, match_name, per_match, score_clip, summary  # noqa: E402

DATA = ROOT / "data" / "spot"


def clip_frames(video: str) -> tuple[int, int]:
    start, end = video.rsplit("_", 2)[1:]
    return int(start), int(end)


def analyse(clip_path: Path, run: Path) -> None:
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(clip_path),
                        "--out", str(run), "--no-render", "--no-broadcast"],
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                       capture_output=True, text=True)
    if r.returncode != 0:  # recorded, and scored as a clip with no system events (docs/EVAL_SPOT.md)
        run.mkdir(parents=True, exist_ok=True)
        (run / "FAILED.txt").write_text(r.stderr[-4000:])
        print(f"  {run.name}: analysis failed: {r.stderr.strip().splitlines()[-1:]}", flush=True)


def run_match(src: Path, clips: list[dict], runs: Path, tmp: Path, keep: bool) -> None:
    todo = sorted((c for c in clips if not ((runs / c["video"] / "report.json").exists()
                                           or (runs / c["video"] / "FAILED.txt").exists())),
                  key=lambda c: clip_frames(c["video"])[0])
    if not todo:
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not all(np.isclose(fps, c["fps"]) for c in todo):
        raise SystemExit(f"{src.name}: {fps} fps, labels say {todo[0]['fps']}: not the labelled upload")
    open_: dict[str, tuple[subprocess.Popen, int, Path]] = {}
    k, i = 0, 0
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
                analyse(out, runs / name)
                if not keep:
                    out.unlink()
        i += 1
    cap.release()


def provenance(sample_path: Path, runs: Path, sample: dict) -> dict:
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    models = next((json.loads(p.read_text()).get("models") for c in sample["clips"]
                   if (p := runs / c["video"] / "report.json").exists()), None)
    return {"code_sha": sha, "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "models": {k: v.get("sha256") for k, v in (models or {}).items()}, "tolerances_s": TOL_S}


def line(name: str, k: dict) -> str:
    return (f"{name:10s} labelled {k['labelled']:5d}  system {k['system']:5d}  recall {k['recall']:.3f}  "
            f"precision {k['precision']:.3f}  F1 {k['f1']:.3f}  side {k['same_side']:.3f}  "
            f"median dt {k['median_dt_frames']:+.1f} fr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "run", "score"))
    ap.add_argument("--name", required=True, help="sample name: data/spot/sample_<name>.json")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--videos", default=str(DATA / "videos"))
    ap.add_argument("--runs-dir", default="runs_spot")
    ap.add_argument("--keep-clips", action="store_true")
    ap.add_argument("--delete-videos", action="store_true",
                    help="delete a match video once every one of its clips has a report or failure record")
    args = ap.parse_args()
    sample_path = DATA / f"sample_{args.name}.json"

    if args.cmd == "sample":  # every clip of the named matches, fixed before any system output is seen
        clips = [c for s in args.split for c in json.loads((DATA / "tennis" / f"{s}.json").read_text())
                 if match_name(c["video"]) in set(args.matches)]
        missing = set(args.matches) - {match_name(c["video"]) for c in clips}
        if missing:
            raise SystemExit(f"not in {args.split}: {sorted(missing)}")
        bad = [c["video"] for c in clips if np.subtract(*clip_frames(c["video"])[::-1]) != c["num_frames"]]
        if bad:
            raise SystemExit(f"clip names disagree with num_frames: {bad[:3]}")
        sample_path.write_text(json.dumps({"split": args.split, "matches": sorted(args.matches),
                                           "clips": clips}, indent=1))
        print(f"{len(clips)} clips from {len(args.matches)} matches -> {sample_path}")
        return

    sample = json.loads(sample_path.read_text())
    runs = DATA / args.runs_dir
    if args.cmd == "run":  # one clip analysed at a time: one GPU job
        for m in sample["matches"]:
            src = Path(args.videos) / f"{m}.mp4"
            if not src.exists():
                print(f"{m}: no video yet, skipped", flush=True)
                continue
            run_match(src, [c for c in sample["clips"] if match_name(c["video"]) == m], runs, DATA / "clips_tmp",
                      args.keep_clips)
            if args.delete_videos:  # run_match returns only when every clip of the match is done
                src.unlink()
        return

    rows, missing = [], []
    for c in sample["clips"]:
        run = runs / c["video"]
        if (run / "report.json").exists():
            rows.append(score_clip(c, json.loads((run / "report.json").read_text())))
        elif (run / "FAILED.txt").exists():
            rows.append(score_clip(c, None))
        else:
            missing.append(c["video"])
    if missing:  # scoring a partial set would quietly change the sample
        raise SystemExit(f"{len(missing)} clips not run yet, e.g. {missing[:3]}")
    s = summary(rows)
    print(f"{args.name} ({args.runs_dir}): {s['clips']} clips, {s['failed']} failed analyses, "
          f"{s['events_per_clip']:.1f} labelled bounces+strokes per clip, "
          f"calibrated frames {s['calibrated_share']:.1%}")
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
              f"  serve R {v['serve']['recall']:.3f} P {v['serve']['precision']:.3f}")
    (DATA / f"results_{args.name}_{args.runs_dir}.json").write_text(json.dumps(
        {"provenance": provenance(sample_path, runs, sample), "summary": s, "per_match": pm, "clips": rows}))


if __name__ == "__main__":
    main()
