"""Run the pipeline on the TrackNet tennis clips and score its bounces against the human labels.

    .venv/bin/python scripts/eval_tracknet.py [--games game8 game9 game10] [--jobs 1]

Clips are turned into 30 fps videos under data/tracknet/videos, analysed into data/tracknet/runs/<id>
(skipped when a report already exists) and scored with courtvision.eval.tracknet. Results go to
data/tracknet/results.json; a summary is printed. Games 8-10 are the WASB test split (the ball
detector never saw them); games 1-7 were its training data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.eval.labels import zone_from_name  # noqa: E402
from courtvision.eval.tracknet import GAME_LABEL_OFFSET_PX, contacts, load_labels, match  # noqa: E402
from courtvision.officiating.line_call import call_bounce  # noqa: E402

DATA = ROOT / "data" / "tracknet"
TEST_GAMES = ("game8", "game9", "game10")


def clip_video(clip: Path, cid: str) -> Path:
    out = DATA / "videos" / f"{cid}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(clip / "%04d.jpg"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def analyse(video: Path, cid: str, rerun: bool = False, runs_dir: str = "runs", ball_model: str | None = None,
            models: list[str] | None = None) -> Path | None:
    run = DATA / runs_dir / cid
    cached = (run / "perception.pkl").exists()
    if rerun and cached or not (run / "report.json").exists():
        extra = ["--reuse-perception"] if rerun and cached else []
        r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-m", "courtvision.cli", "analyze", str(video),
                            "--out", str(run), "--no-render", "--no-broadcast", *extra,
                            *(["--ball-model", ball_model] if ball_model else []),
                            *[a for m in models or [] for a in ("--model", m)]],
                           env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {cid}: analysis failed: {r.stderr.strip().splitlines()[-1:]}", flush=True)
            return None
    return run


def score_clip(clip: Path, run: Path, offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    lab = load_labels(clip / "Label.csv", offset)
    refs = contacts(lab)
    # First bounce after a stroke: the one a line call is about (later ones end a point already decided).
    first, seen = {}, False
    for i, st in enumerate(lab.status):
        if st == 1:
            seen = False
        elif st == 2:
            first[i], seen = not seen, True
    report = json.loads((run / "report.json").read_text())
    H = json.loads((run / "frames.json").read_text())["H"]
    bounces = [(f["bounce"]["frame"], np.array([f["bounce"]["x"], f["bounce"]["y"]]), f["call"]["verdict"])
               for f in report["flights"] if f.get("call") and f.get("bounce")]
    zones = {round(f["bounce"]["frame"], 3): f["call"]["zone"] for f in report["flights"] if f.get("call")}
    matches, false_alarms = match(refs, bounces, H, camera=report["calibration"].get("camera_center_m"))
    rows = []
    for m, c in zip(matches, sorted(refs, key=lambda c: c.frame), strict=True):
        ref_verdict = None
        if m.sys_frame is not None and m.ref_xy is not None:
            zone = zone_from_name(zones[round(m.sys_frame, 3)])
            ref_verdict = call_bounce(m.ref_xy, np.eye(2) * 1e-10, zone).verdict
        k = min(first, key=lambda i: abs(i - c.frame)) if first else None
        rows.append({"ref_frame": m.ref_frame, "first": first.get(k, True), "from_tracks": c.from_tracks,
                     "sys_frame": m.sys_frame,
                     "calibrated": m.ref_xy is not None, "error_m": m.error_m,
                     "dx_m": None if m.error_m is None else float(m.sys_xy[0] - m.ref_xy[0]),
                     "dy_m": None if m.error_m is None else float(m.sys_xy[1] - m.ref_xy[1]),
                     "sys_verdict": m.sys_verdict, "ref_verdict": ref_verdict})
    return {"refs": rows, "false_alarms": false_alarms, "n_frames": int(len(lab.status)),
            "calib_valid": report["calibration"]["valid_frames"], "flights": len(report["flights"]),
            "reference_offset_px": list(offset)}


def summary(res: dict[str, dict], games: tuple[str, ...]) -> dict:
    rows = [r for cid, c in res.items() if cid.split("_")[0] in games for r in c["refs"]]
    fa = sum(c["false_alarms"] for cid, c in res.items() if cid.split("_")[0] in games)
    cal = [r for r in rows if r["calibrated"]]
    hit = [r for r in cal if r["sys_frame"] is not None]
    err = np.array([r["error_m"] for r in hit]) * 100
    dt = np.array([r["sys_frame"] - r["ref_frame"] for r in hit])
    decisive = [r for r in hit if r["sys_verdict"] in ("IN", "OUT")]
    agree = sum(r["sys_verdict"] == r["ref_verdict"] for r in decisive)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if len(a) else float("nan")

    first = [r for r in cal if r.get("first", True)]
    first_hit = [r for r in first if r["sys_frame"] is not None]
    return {"clips": sum(cid.split("_")[0] in games for cid in res), "ref_bounces": len(rows),
            "first_bounces": len(first), "first_found": len(first_hit),
            "first_recall": len(first_hit) / len(first) if first else float("nan"),
            "ref_on_calibrated_frames": len(cal), "found": len(hit),
            "recall": len(hit) / len(cal) if cal else float("nan"), "false_alarms": fa,
            "precision": len(hit) / (len(hit) + fa) if hit or fa else float("nan"),
            "error_cm_median": q(err, 50), "error_cm_p90": q(err, 90),
            "timing_frames_median_abs": q(np.abs(dt), 50),
            "decisive": len(decisive), "decisive_agree": agree,
            "disagreements": [r for r in decisive if r["sys_verdict"] != r["ref_verdict"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--rerun", action="store_true", help="re-run events and fits on cached detections")
    ap.add_argument("--results", default="results.json", help="results file name under data/tracknet")
    ap.add_argument("--runs-dir", default="runs", help="run folder under data/tracknet (keeps model variants apart)")
    ap.add_argument("--ball-model", help="ball network checkpoint passed to the analysis")
    ap.add_argument("--model", action="append", default=[], help="KEY=PATH passed to the analysis (repeatable)")
    ap.add_argument("--centred-reference", action="store_true",
                    help="reference contacts from labels minus their game's GAME_LABEL_OFFSET_PX (image centre)")
    args = ap.parse_args()
    games = sorted((DATA / "Dataset").glob("game*"), key=lambda p: int(p.name[4:]))
    if args.games:
        games = [g for g in games if g.name in args.games]
    out = DATA / args.results
    res = json.loads(out.read_text()) if out.exists() else {}
    for g in games:
        for clip in sorted(g.glob("Clip*"), key=lambda p: int(p.name[4:])):
            cid = f"{g.name}_{clip.name}"
            run = analyse(clip_video(clip, cid), cid, args.rerun, args.runs_dir, args.ball_model, args.model)
            if run is None:
                continue
            res[cid] = score_clip(clip, run, GAME_LABEL_OFFSET_PX[g.name] if args.centred_reference else (0.0, 0.0))
            print(f"  {cid}: refs {len(res[cid]['refs'])}, found "
                  f"{sum(r['sys_frame'] is not None for r in res[cid]['refs'])}, false {res[cid]['false_alarms']}",
                  flush=True)
            out.write_text(json.dumps(res, indent=1))
    all_games = tuple(f"game{i}" for i in range(1, 11))
    for name, gs in (("test (game8-10, unseen by the ball detector)", TEST_GAMES), ("all games", all_games)):
        s = summary(res, gs)
        print(name, json.dumps({k: v for k, v in s.items() if k != "disagreements"}, indent=1))
        for r in s["disagreements"]:
            print("   disagree", r)


if __name__ == "__main__":
    main()
