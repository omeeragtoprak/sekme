"""Command line: analyse a video, serve the review room, run the simulation benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from courtvision.errors import MissingModelError, UnreadableVideoError


def _analyze(args: argparse.Namespace) -> int:
    from courtvision.events.serves import TOSS_BRIDGE_S
    from courtvision.pipeline import PipelineConfig, analyze

    out = Path(args.out or Path("runs") / Path(args.video).stem)
    t0 = time.time()

    def progress(stage: str, frac: float) -> None:
        print(f"\r[{time.time() - t0:6.1f}s] {stage:<10s} {100 * frac:5.1f}%", end="", file=sys.stderr, flush=True)

    cfg = PipelineConfig(device=args.device, render=not args.no_render, reuse_perception=args.reuse_perception,
                         broadcast=not args.no_broadcast, ball_stride=args.ball_stride,
                         serve_toss_bridge_s=TOSS_BRIDGE_S if args.toss_bridge else 0.0,
                         pin_serve_height=args.pin_serve_height, fps_scaled_thresholds=args.fps_scaled_thresholds,
                         player_size_from_camera=args.player_size_from_camera)
    if args.ball_model:
        cfg.models["ball"] = str(Path(args.ball_model).resolve())
    for item in args.model:
        key, _, path = item.partition("=")
        if key not in cfg.models or not path:
            raise SystemExit(f"--model expects KEY=PATH with KEY in {sorted(cfg.models)}, got {item!r}")
        cfg.models[key] = str(Path(path).resolve())
    if args.fit_workers is not None:
        os.environ["SEKME_FIT_WORKERS"] = str(max(1, args.fit_workers))
    report = analyze(args.video, out, cfg, progress)
    print(file=sys.stderr)
    if report["video"].get("irregular_frame_intervals"):
        print(f"uyarı: değişken kare hızı ({report['video']['irregular_frame_intervals']} düzensiz kare aralığı); "
              "hızlar ve işaretler kayabilir", file=sys.stderr)
    cal = report["calibration"]
    print(f"calibrated frames {cal['valid_frames']}/{cal['total_frames']}, line rmse {cal['line_rmse_px_median']} px")
    for f in report["flights"]:
        call = f["call"]
        verdict = f"{call['verdict']:6s} P(in)={call['p_in']:.3f} margin {100 * call['margin_m']:+.0f}±" \
                  f"{100 * call['sigma_m']:.0f} cm ({call['line']})" if call else "no call: " + "; ".join(f["notes"])
        print(f"  flight {f['id']}: {f['hitter']:>4s} -> {f['receiver']:<4s} {verdict}")
    print(f"report: {out / 'report.json'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from courtvision.api.app import create_app
    from courtvision.pipeline import PipelineConfig, analyze

    def run_analysis(video: Path, out: Path, progress) -> None:
        analyze(video, out, PipelineConfig(device=args.device), progress)

    local = args.host in ("127.0.0.1", "localhost", "::1")
    if not local:
        print(f"uyarı: İnceleme Odası {args.host} adresinde, kimlik doğrulaması olmadan açılıyor; ağdaki herkes "
              "analizleri görebilir ve video yükleyebilir", file=sys.stderr)
    app = create_app(Path(args.runs), analyze_fn=run_analysis, allowed_hosts=None if local else ["*"])
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    import numpy as np

    from courtvision.calib.camera import Camera
    from courtvision.eval.sim_benchmark import run_benchmark

    fwd = np.array([0.0, 1.5, 0.0]) - np.array([0.0, -args.distance, args.height])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    center = np.array([0.0, -args.distance, args.height])
    w, h = args.width, int(args.width * 9 / 16)
    K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1.0]])
    rep = run_benchmark(Camera(K, R, -R @ center), args.shots, args.fps, args.noise)
    print(rep.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sekme", description="SEKME: tennis line calls from a single broadcast camera")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a tennis video")
    a.add_argument("video")
    a.add_argument("--out")
    a.add_argument("--device", default="auto")
    a.add_argument("--no-render", action="store_true")
    a.add_argument("--no-broadcast", action="store_true", help="skip reading the score graphic and speed board")
    a.add_argument("--reuse-perception", action="store_true", help="reuse cached detections from a previous run")
    a.add_argument("--ball-model", help="ball network checkpoint (SEKME U-Net or WASB); default from the model cards")
    a.add_argument("--ball-stride", type=int, default=0, help="frames between the ball network's inputs (0: auto)")
    a.add_argument("--toss-bridge", action="store_true",
                   help="candidate serve detector: bridge a missed toss apex (off by default, docs/EVAL_SPOT.md)")
    a.add_argument("--pin-serve-height", action="store_true",
                   help="candidate: pin the flight fit to the measured serve contact height (off by default, "
                        "docs/EVAL_SERVE_SPEED.md)")
    a.add_argument("--player-size-from-camera", action="store_true",
                   help="candidate: player size gate from the calibrated camera (off by default, "
                        "docs/EVAL_SERVE_SPEED.md)")
    a.add_argument("--fps-scaled-thresholds", action="store_true",
                   help="candidate: frame counts set on 30 fps video scaled by round(fps / 30) (off by default, "
                        "docs/EVAL_SERVE_SPEED.md)")
    a.add_argument("--fit-workers", type=int, default=None, metavar="N",
                   help="processes for the 3D flight fits (same results; default SEKME_FIT_WORKERS or 1). "
                        "About 3x faster re-analysis of a long run with 4")
    a.add_argument("--model", action="append", default=[], metavar="KEY=PATH",
                   help="override a network checkpoint: court, ball, players or pose (repeatable)")
    a.set_defaults(fn=_analyze)
    s = sub.add_parser("serve", help="start the review room web app")
    s.add_argument("--runs", default="runs")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=_serve)
    b = sub.add_parser("benchmark", help="simulated accuracy for a camera setup")
    b.add_argument("--fps", type=float, default=30.0)
    b.add_argument("--noise", type=float, default=1.5, help="ball detection noise (px)")
    b.add_argument("--distance", type=float, default=28.0, help="camera distance behind the net (m)")
    b.add_argument("--height", type=float, default=11.0, help="camera height (m)")
    b.add_argument("--focal", type=float, default=1750.0, help="focal length (px)")
    b.add_argument("--width", type=int, default=1920)
    b.add_argument("--shots", type=int, default=40)
    b.set_defaults(fn=_benchmark)
    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except (UnreadableVideoError, MissingModelError) as exc:  # a wrong input or setup, not a bug: no traceback
        print(f"hata: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
