"""End-to-end analysis: video -> calibrated court, tracks, events, calls, report and overlays.

Pass 1 decodes the video once and feeds the court calibrator, the ball
model and the person tracker. Analysis (calibration smoothing, player
selection, hits/bounces, 3D fits, line calls) runs on the collected
tracks. Pass 2 re-decodes the video to render the overlay videos.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from courtvision import __version__
from courtvision.analytics.kinematics import player_kinematics
from courtvision.broadcast import ocr
from courtvision.broadcast.scoreboard import BroadcastScan, scan_video
from courtvision.errors import MissingModelError, UnreadableVideoError
from courtvision.eval.broadcast_check import Serve, check, serve_anchored_names, side_names
from courtvision.events.flights import Flight, build_flights, measurable_segments
from courtvision.events.hits import detect_hits_in_segments
from courtvision.events.serves import detect_serves, merge_serves
from courtvision.events.strokes import classify, serve_hand, vote_hand
from courtvision.geometry.court import CourtModel
from courtvision.io.video import VideoWriter, irregular_frame_intervals, iter_frames, link_source, probe
from courtvision.perception.ball import BallDetector, ball_arch, drop_court_fixed, track_ball
from courtvision.perception.court import (
    CourtCalibrator,
    CourtKeypointDetector,
    calibrate_sequence,
    frame_hist,
    pick_device,
)
from courtvision.perception.players import person_tracker, select_players
from courtvision.perception.pose import pose_estimator, poses_at
from courtvision.physics.flight import flight_states
from courtvision.render.hud import OverlayState, Renderer

ROOT = Path(__file__).resolve().parents[2]

# Networks this project designed and trained (docs/MODELS.md), by the ``arch`` their checkpoint carries.
OWN_CARDS = {
    "sekme-court-unet": {"source": "this project: SEKME court U-Net, scripts/train_court.py on the "
                                   "TennisCourtDetector labels (8 841 broadcast frames)",
                         "license": "AGPL-3.0 (own weights; the dataset has no stated licence)"},
    "sekme-ball-unet": {"source": "this project: SEKME ball U-Net, scripts/train_ball.py on TrackNet tennis",
                        "license": "AGPL-3.0 (own weights; the TrackNet data has no stated licence)"},
    "sekme-person-unet": {"source": "this project: SEKME person U-Net, scripts/train_players.py, distilled from "
                                    "YOLO26-s boxes on broadcast frames",
                          "license": "AGPL-3.0 (own weights)"},
    "sekme-pose-unet": {"source": "this project: SEKME pose U-Net, scripts/train_pose.py, distilled from "
                                  "YOLO26-s-pose keypoints on player crops",
                        "license": "AGPL-3.0 (own weights)"},
}

# Published networks, used when a checkpoint is not one of ours (default or selected with --model).
THIRD_PARTY_CARDS = {
    "court": {"source": "github.com/yastrebksv/TennisCourtDetector", "license": "none published (research use only)"},
    "ball": {"source": "github.com/nttcom/WASB-SBDT (BMVC 2023)", "license": "MIT"},
    "players": {"source": "Ultralytics YOLO26-s", "license": "AGPL-3.0"},
    "pose": {"source": "Ultralytics YOLO26-s-pose (COCO 17 keypoints)", "license": "AGPL-3.0"},
}

# Default checkpoint per role (override with sekme analyze --model KEY=PATH).
MODEL_FILES = {
    "court": "models/sekme_court.pt",
    "ball": "models/sekme_ball.pt",
    "players": "models/yolo26s.pt",
    "pose": "models/yolo26s-pose.pt",
}

BALL_TRAIN_FPS = 30.0  # both ball networks were trained on 30 fps TrackNet video


def frame_scale(fps: float) -> int:
    """Frames per 30 fps frame (1 at 25-30 fps, 2 at 50-60 fps): the ball net's input stride, and the factor
    for frame counts set on 30 fps video (candidate --fps-scaled-thresholds)."""
    return max(1, round(fps / BALL_TRAIN_FPS))


def _own_arch(path: str) -> str | None:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  third-party checkpoints need their own classes to unpickle
        return None
    return ckpt.get("arch") if isinstance(ckpt, dict) else None


# Settings the cached perception depends on besides the networks: the calibrator's keyframe interval, the shot-cut
# threshold (cuts, and the person tracker reset at each cut) and the person detector's input size.
PERCEPTION_SETTINGS = ("keyframe_interval", "cut_threshold", "player_imgsz")


def perception_key(video_sha: str, cfg: PipelineConfig) -> dict:
    """What cached detections depend on: the video, the court/ball/person networks, the ball stride and the
    perception settings."""
    return {"video_sha256": video_sha, "ball_stride": cfg.ball_stride,
            "models": {k: _sha256(cfg.models[k]) for k in ("court", "ball", "players")},
            "settings": {k: getattr(cfg, k) for k in PERCEPTION_SETTINGS}}


def same_perception_key(cached: dict | None, key: dict) -> bool:
    """A cache written before the settings were part of its key was made with their defaults, which have not
    changed since the settings were introduced (2026-10-27): it is read as such, not recomputed. A cache without a
    usable key (none, or null) is recomputed."""
    if not cached:
        return False
    defaults = {f.name: f.default for f in fields(PipelineConfig) if f.name in PERCEPTION_SETTINGS}
    return {"settings": defaults, **cached} == key


def pose_key(video_sha: str, model_sha: str, reqs: list[tuple[int, str, np.ndarray]]) -> dict:
    """What cached poses depend on: the video, the pose network, and for each request the frame, the hitter's side
    and the box the crop is cut from (another player selection at the same frame must not reuse the pose)."""
    def box(b: np.ndarray) -> list[float | None]:
        return [round(float(v), 1) if np.isfinite(v) else None for v in np.asarray(b, dtype=float)]
    return {"video_sha256": video_sha, "model_sha256": model_sha,
            "requests": [(int(f), side, box(b)) for f, side, b in reqs]}


def _model_card(key: str, path: str) -> dict:
    arch = ball_arch(path) if key == "ball" else _own_arch(path)
    card = OWN_CARDS.get(arch or "", THIRD_PARTY_CARDS[key])
    if arch in OWN_CARDS:  # our checkpoints record their own training split and epoch choice
        # weights_only=False: their validation scores are numpy scalars (ball_arch loads them the same way)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        card = {**card, **{k: ckpt[k] for k in ("trained_on", "selected_on", "epoch") if k in ckpt}}
    return {**card, "file": f"models/{Path(path).name}", "sha256": _sha256(path)}


# Added to a report's limitations when the video's frame timestamps are not evenly spaced (the pipeline takes a
# frame's time as its index divided by the frame rate).
VFR_NOTE = ("Videonun kare hızı değişken: {n} kare aralığı ortancadan %25'ten fazla sapıyor (ör. telefonun "
            "düşürdüğü kareler). Zamanlar kare sırasından hesaplandığı için bu aralıkların çevresindeki hızlar ve "
            "video üstündeki işaretler kayabilir.")

LIMITATIONS = [
    "Tek yayın kamerası: sekme konumlarında santimetre ile desimetre arası belirsizlik var; her karar içeride olma "
    "olasılığını ve 1σ payını gösterir; %95'te kesinleşmeyen kararlar düşük güvenli olarak işaretlenir.",
    "Topun yere temas izi sabit 2 cm yarı uzunlukla modelleniyor; hızlı kortlarda gerçek iz daha uzun olabilir.",
    "Lens bozulması modellenmiyor; bunun yerine her karara kalibrasyon belirsizliği (en az 0,5 piksel) eklenir.",
    "Servisler top atışından bulunuyor (ölü toptan sonra dip çizgideki oyuncunun üstünde yavaş, dikeye yakın top) ve "
    "kamera ışınında ölçülen temas yüksekliğiyle doğrulanıyor; çok az gerçek serviste denendi. Temastan hemen sonraki "
    "kareler çok bulanıksa top ağı onları kaçırabilir ve servis tanınmaz. Kamera geçişinden önce atılan servisler "
    "kurtarılamıyor.",
    "File üstü yükseklik 3B uydurmadan geliyor ve tek kamerayla yaklaşık 0,1–0,25 m 1σ belirsizlik taşıyor; "
    "değer bu belirsizlikle birlikte gösterilir.",
    "Bağımsız ölçüm: elle etiketlenmiş TrackNet maçlarında sekme konumu ortanca ~16 cm (etiketin kendi belirsizliği "
    "~10 cm'nin altını ayırt edemez); çizgiye santimetre yakın kararların doğruluğu henüz ölçülmedi. İnceleme "
    "Odası'ndaki etiketleme aracıyla (gerçek sekme noktasına tıklama) kendi görüntünüzde ölçülebilir.",
    "Skor ve sayı sonucu henüz çıkarılmıyor; klip başlamadan önce vurulmuş topların vuruş hızı verilmiyor.",
    "Yayın tabelası ve stadyum radar panosu Apple Vision ile (yalnızca macOS) okunur ve yalnızca doğrulama için "
    "kullanılır, kararları etkilemez. Radar karşılaştırması yalnızca pano değişiminin zamanı servisten sonra "
    "≤ 4 s içinde bilinen servislerde yapılır; örnek az olduğu için tek servis farkı ±%6 civarındadır.",
    "Ralli vuruşlarında 3B uydurmanın temas yüksekliği zayıf belirleniyor (iki forehand 2,4–2,6 m çıktı); bu yüzden "
    "yalnızca kamera ışınında ölçülen servis temas yüksekliği gösterilir. Vuruş türü (forehand/backhand/vole/smaç) "
    "poz ve top konumundan çıkarılır; US Open klibinde 16 forehand/backhand gözle kontrol edildi, 16'sı doğru.",
]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PipelineConfig:
    device: str = "auto"
    keyframe_interval: int = 15
    cut_threshold: float = 0.55
    render: bool = True
    reuse_perception: bool = False  # reuse runs/<id>/perception.pkl when the video hash matches
    broadcast: bool = True  # read the score graphic and the serve-speed board (macOS Vision) as a check
    player_imgsz: int = 1280
    ball_stride: int = 0  # frames between the ball network's three inputs; 0: match its 30 fps training video
    serve_toss_bridge_s: float = 0.0  # candidate, off by default (docs/EVAL_SPOT.md): 0.3 bridges a missed toss apex
    pin_serve_height: bool = False  # candidate, off by default (docs/EVAL_SERVE_SPEED.md)
    fps_scaled_thresholds: bool = False  # candidate, off by default (docs/EVAL_SERVE_SPEED.md): 60 fps frame counts
    player_size_from_camera: bool = False  # candidate, off by default (docs/EVAL_SERVE_SPEED.md): camera size gate
    models: dict[str, str] = field(default_factory=lambda: {k: str(ROOT / v) for k, v in MODEL_FILES.items()})


def _f(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _flight_json(i: int, fl: Flight, fps: float) -> dict:
    d: dict = {
        "id": i, "start_frame": fl.start_frame, "end_frame": fl.end_frame, "hitter": fl.hitter,
        "receiver": str(fl.receiver), "kind": fl.kind, "bounce_frame": fl.bounce_frame,
        "hit_speed_kmh": _f(fl.hit_speed_kmh, 1), "bounce_speed_kmh": _f(fl.bounce_speed_kmh, 1),
        "net_clearance_m": _f(fl.net_clearance_m),
        "net_clearance_sd_m": _f(fl.net_clearance_sd_m), "hit_height_m": _f(fl.hit_height_m),
        "contact_frame": _f(fl.contact_time * fps, 2) if fl.contact_time is not None else None,
        "play_continued": fl.play_continued, "rejected": fl.rejected, "notes": fl.notes,
        "bounce": None, "call": None, "trajectory": None, "fit": None,
    }
    fit = fl.fit
    if fit is not None and fit.success and np.isfinite(fit.rms_px):
        d["fit"] = {"rms_px": _f(fit.rms_px, 2), "n_obs": fit.n_obs, "spin_in": _f(fit.params[9]),
                    "spin_out": _f(fit.params[10])}
    if fit is not None and fit.success and fl.rejected is None and np.all(np.isfinite(fit.bounce_cov)):
        cov = fit.bounce_cov
        d["bounce"] = {"t": _f(fit.bounce_time, 4), "frame": _f(fit.bounce_time * fps, 2),
                       "x": _f(fit.bounce_xy[0], 4), "y": _f(fit.bounce_xy[1], 4),
                       "cov": [[float(v) for v in row] for row in cov],
                       "sd_x_cm": _f(100 * np.sqrt(cov[0, 0]), 1), "sd_y_cm": _f(100 * np.sqrt(cov[1, 1]), 1)}
        # Sample between the fitted contacts (not the image-based hit frames) so the replay's
        # speed readout matches the reported contact speed.
        t0 = fl.contact_time if fl.contact_time is not None else fl.start_frame / fps
        t1 = float(fit.anchor_times[-1]) if len(fit.anchor_times) and fit.anchor_times[-1] > fit.bounce_time \
            else fl.end_frame / fps
        ts = np.linspace(t0, t1, 60)
        pos, _ = flight_states(fit.params, ts)
        d["trajectory"] = [[_f(t, 4), _f(p[0]), _f(p[1]), _f(p[2])] for t, p in zip(ts, pos, strict=True)]
    elif fl.bounce_2d is not None:
        f, g, cov = fl.bounce_2d
        d["bounce"] = {"t": _f(f / fps, 4), "frame": _f(f, 2), "x": _f(g[0], 4), "y": _f(g[1], 4),
                       "cov": [[float(v) for v in row] for row in cov], "method": "2d",
                       "sd_x_cm": _f(100 * np.sqrt(cov[0, 0]), 1), "sd_y_cm": _f(100 * np.sqrt(cov[1, 1]), 1)}
    if d["bounce"] is not None:
        d["bounce"].setdefault("method", "3d")
    if fl.call is not None:
        d["call"] = fl.call.to_dict()
    return d


def _perceive(video, info, cfg: PipelineConfig, court: CourtModel, device, prog):
    """Single decode pass feeding the court calibrator, the ball model and the person tracker."""
    calibrator = CourtCalibrator(CourtKeypointDetector(cfg.models["court"], device), court, cfg.keyframe_interval)
    ball_det = BallDetector(cfg.models["ball"], device, half=True)
    stride = cfg.ball_stride or frame_scale(info.fps)
    ball_det.start((info.width, info.height), stride)
    people = person_tracker(cfg.models["players"], device.type, imgsz=cfg.player_imgsz)
    pending, dets, cuts = [], [], [0]
    prev_hist = None
    n = 0
    # The court calibrator (mostly CPU line fitting, ~21 ms/frame) runs in one worker thread, in frame order,
    # while the ball and person networks run here.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, frame in iter_frames(video):
            if i == 0 and frame.shape[:2] != (info.height, info.width):  # sizes would scale every detection wrongly
                raise UnreadableVideoError(f"{Path(video).name}: kareler {frame.shape[1]}×{frame.shape[0]} çözülüyor, "
                                           f"video {info.width}×{info.height} bildiriyor (döndürme?)")
            hist = frame_hist(frame)
            is_cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < cfg.cut_threshold
            if is_cut:
                cuts.append(i)
            prev_hist = hist
            pending.append(pool.submit(calibrator, i, frame, is_cut or i == 0))
            ball_det.push(i, frame)
            if is_cut and hasattr(people, "reset"):
                people.reset()
            dets.append(people(frame))
            n = i + 1
            if i % 10 == 0:
                prog("perception", i / max(info.n_frames, 1))
        meas = [f.result() for f in pending]
    return meas, dets, cuts, ball_det.finish(n), n


def _broadcast_scan(video, fps: float, out: Path, video_sha: str, cfg: PipelineConfig) -> BroadcastScan:
    """OCR the broadcast graphics, cached next to the perception cache (same reuse rule)."""
    cache = out / "broadcast.pkl"
    if cfg.reuse_perception and cache.is_file():
        with open(cache, "rb") as f:  # local cache written by this pipeline (trusted)
            cached = pickle.load(f)
        if cached.get("video_sha256") == video_sha:
            return cached["scan"]
    scan = scan_video(video, fps)
    with open(cache, "wb") as f:
        pickle.dump({"video_sha256": video_sha, "scan": scan}, f)
    return scan


def _annotate_from_broadcast(flights: list[dict], broadcast: dict, scan: BroadcastScan | None, fps: float,
                             cuts: list[int]) -> None:
    """Board speed and server name on serves; player names on every shot where the graphic allows it: from the
    point's serve when its server is marked, else from the start ends and the end changes on the graphic."""
    if scan is None:
        return
    pairs = {p["flight_id"]: p for p in broadcast["serve_speed"]["pairs"]}
    servers = {x["flight_id"]: x["server"] for x in broadcast["server_sides"]["serves"]}
    start_end = broadcast["server_sides"]["start_end"]
    point_ends = [round(c["t"][1] * fps) for c in broadcast["score_changes"]]
    anchored = serve_anchored_names(flights, servers, scan.names, cuts, point_ends)
    for f in flights:
        f["board_speed_kmh"] = pairs[f["id"]]["board_kmh"] if f["id"] in pairs else None
        f["server_name"] = servers.get(f["id"])
        names = anchored.get(f["id"]) or side_names(scan, start_end, f["start_frame"] / fps)
        f["hitter_name"] = names.get(f["hitter"]) if names else None
        f["receiver_name"] = names.get(f["receiver"]) if names else None


def _annotate_strokes(video, fjson: list[dict], flights: list[Flight], ball, players, kin, out: Path, video_sha: str,
                      cfg: PipelineConfig, device) -> dict:
    """Pose at each contact -> hitting hand (from serves) and stroke type on every shot that starts with a hit."""
    idx = [i for i, fl in enumerate(flights) if fl.contact_time is not None or fl.kind == "serve"]
    reqs = [(flights[i].start_frame, players[flights[i].hitter].boxes[flights[i].start_frame]) for i in idx]
    cache = out / "pose.pkl"
    key = pose_key(video_sha, _sha256(cfg.models["pose"]),
                   [(flights[i].start_frame, flights[i].hitter, r[1]) for i, r in zip(idx, reqs, strict=True)])
    poses = None
    if cfg.reuse_perception and cache.is_file():
        with open(cache, "rb") as f:  # local cache written by this pipeline (trusted)
            c = pickle.load(f)
        if c.get("key") == key:
            poses = c["poses"]
    if poses is None:
        est = pose_estimator(cfg.models["pose"], device.type)
        poses = poses_at(video, reqs, est)
        with open(cache, "wb") as f:
            pickle.dump({"key": key, "poses": poses}, f)
    pose_of = dict(zip(idx, poses, strict=True))

    def who(f: dict) -> str:
        return f.get("hitter_name") or f"side:{f['hitter']}"

    votes: dict[str, list[str]] = {}
    for i in idx:
        if flights[i].kind == "serve" and (h := serve_hand(pose_of[i])) is not None:
            votes.setdefault(who(fjson[i]), []).append(h)
    hands = {k: v for k, v in ((k, vote_hand(v)) for k, v in votes.items()) if v is not None}
    for i, (fl, f) in enumerate(zip(flights, fjson, strict=True)):
        if i not in pose_of:
            f.update(stroke=None, stroke_wing=None, stroke_confident=False, stroke_cue="vuruş anı klipte görülmüyor",
                     hand=None, hand_source=None)
            continue
        s = fl.start_frame
        hand = hands.get(who(f))
        source = "servis" if hand else "varsayılan"
        prev = next((p for p in flights if p.end_frame == s and p.hitter != fl.hitter), None)
        bounced = None if prev is None else prev.bounce_frame is not None
        bxy = next((ball.xy[j] for j in (s, s - 1, s + 1, s - 2) if 0 <= j < len(ball.xy)
                    and np.isfinite(ball.xy[j, 0])), None)
        pos = kin[fl.hitter].position[s]
        st = classify(fl.kind == "serve", pose_of[i], bxy, players[fl.hitter].boxes[s], fl.hitter, hand or "right",
                      bounced, float(pos[1]) if np.isfinite(pos[1]) else None)
        f.update(stroke=st.kind, stroke_wing=st.wing, stroke_confident=st.confident, stroke_cue=st.cue,
                 hand=hand or "right", hand_source=source)
    return {"model": "yolo26s-pose", "poses": sum(p is not None for p in poses), "requested": len(poses),
            "hands": {k.removeprefix("side:"): v for k, v in hands.items()}}


def analyze(video: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None,
            progress: Callable[[str, float], None] | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    info = probe(video)  # before the run folder exists: an unreadable file leaves no empty run behind
    missing = [f"{k}: {Path(v).name}" for k, v in cfg.models.items() if not Path(v).is_file()]
    if missing:
        raise MissingModelError(f"model dosyası bulunamadı ({', '.join(missing)}); ağırlıklar repoda değil, "
                                "docs/MODELS.md")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prog = progress or (lambda stage, frac: None)
    timings: dict[str, float] = {}
    court = CourtModel()
    device = pick_device(cfg.device)

    t0 = time.time()
    video_sha = _sha256(video)
    cache = out / "perception.pkl"
    perc_key = perception_key(video_sha, cfg)
    cached = None
    if cfg.reuse_perception and cache.is_file():
        with open(cache, "rb") as f:  # local cache written by this pipeline (trusted)
            cached = pickle.load(f)
        if not same_perception_key(cached.get("key", {}), perc_key):  # other video, networks or settings: recompute
            cached = None
    if cached is not None:
        meas, dets, cuts, cands, n = cached["meas"], cached["dets"], cached["cuts"], cached["cands"], cached["n"]
    else:
        meas, dets, cuts, cands, n = _perceive(video, info, cfg, court, device, prog)
        with open(cache, "wb") as f:
            pickle.dump({"key": perc_key, "meas": meas, "dets": dets, "cuts": cuts, "cands": cands, "n": n}, f)
    timings["perception_s"] = time.time() - t0

    t1 = time.time()
    prog("analysis", 0.0)
    calib = calibrate_sequence(meas, cuts, (info.width, info.height), court)
    # Frame counts below were set on 30 fps video; the candidate scales them like the ball net's input stride.
    fscale = frame_scale(info.fps) if cfg.fps_scaled_thresholds else 1
    ball = track_ball(drop_court_fixed(cands, [c.H_inv if c.valid else None for c in calib],
                                       win=15 * fscale, min_frames=10 * fscale), info.width)
    players = select_players(dets, calib, info.fps, size_from_camera=cfg.player_size_from_camera)
    kin = {side: player_kinematics(tr.world, info.fps) for side, tr in players.items()}
    segments = measurable_segments(calib, cuts, min_len=int(info.fps))
    hits = detect_hits_in_segments(ball, players, info.fps, info.height, segments)
    hits = merge_serves(hits, detect_serves(ball, players, info.fps, segments, bridge_s=cfg.serve_toss_bridge_s,
                                            frame_scale=fscale), info.fps)
    flights = build_flights(ball, hits, calib, info.fps, players, segments, frame_w=info.width,
                            pin_serve_height=cfg.pin_serve_height)
    timings["analysis_s"] = time.time() - t1

    t3 = time.time()
    prog("broadcast", 0.0)
    scan = _broadcast_scan(video, info.fps, out, video_sha, cfg) if cfg.broadcast and ocr.available() else None
    serves = [Serve(i, fl.contact_time if fl.contact_time is not None else fl.start_frame / info.fps, fl.hitter,
                    None if fl.rejected else fl.hit_speed_kmh)  # a rejected fit's speed is not a measurement
              for i, fl in enumerate(flights) if fl.kind == "serve"]
    measurable_t = [(a / info.fps, b / info.fps) for a, b in segments]
    broadcast = check(scan, serves, measurable=measurable_t) if scan is not None else {
        "ocr": None, "reason": "disabled" if not cfg.broadcast else "Apple Vision OCR is not available"}
    if scan is not None:
        timings["broadcast_s"] = time.time() - t3

    valid = [c for c in calib if c.valid]
    cams = [c.camera for c in valid if c.camera is not None]
    irregular = irregular_frame_intervals(video)
    report = {
        "schema": "courtvision.report/1",
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "video": {"file": Path(video).name, "width": info.width, "height": info.height, "fps": info.fps,
                  "n_frames": n, "duration_s": round(n / info.fps, 3), "sha256": video_sha,
                  "irregular_frame_intervals": irregular},
        "models": {k: _model_card(k, v) for k, v in cfg.models.items()},
        "candidates": {k: v for k, v in (("serve_toss_bridge_s", cfg.serve_toss_bridge_s),
                                         ("pin_serve_height", cfg.pin_serve_height),
                                         ("fps_scaled_thresholds", cfg.fps_scaled_thresholds),
                                         ("player_size_from_camera", cfg.player_size_from_camera)) if v},
        "calibration": {
            "valid_frames": len(valid), "total_frames": n, "shot_cuts": cuts,
            "line_support_median": _f(np.median([c.support for c in valid]) if valid else None),
            "line_rmse_px_median": _f(np.nanmedian([c.line_rmse_px for c in valid]) if valid else None, 2),
            "focal_px_median": _f(np.median([c.focal for c in cams]) if cams else None, 1),
            "camera_center_m": [_f(v, 2) for v in np.median([c.center for c in cams], axis=0)] if cams else None,
            "keyframes": int(sum(m.from_keyframe for m in meas)),
            "measurable_segments": [[a, b] for a, b in segments],
        },
        "ball": {"detected_frames": int((ball.source == 1).sum()), "interpolated_frames": int((ball.source == 2).sum()),
                 "missing_frames": int((ball.source == 0).sum())},
        "players": {
            side: {"label": "P1" if side == "near" else "P2", "track_ids": players[side].track_ids,
                   "coverage": _f(np.isfinite(players[side].world[:, 0]).mean()),
                   "distance_m": _f(kin[side].distance_m[-1] if len(kin[side].distance_m) else None, 1),
                   "peak_kmh": _f(kin[side].peak_kmh, 1),
                   "mean_kmh": _f(kin[side].mean_kmh, 1)}
            for side in players
        },
        "hits": [{"frame": h.frame, "t": _f(h.frame / info.fps), "side": h.side, "next_to_player": h.near_player,
                  "serve": h.serve}
                 for h in hits],
        "flights": [_flight_json(i, fl, info.fps) for i, fl in enumerate(flights)],
        "broadcast": broadcast,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "limitations": LIMITATIONS + ([VFR_NOTE.format(n=irregular)] if irregular else []),
    }
    _annotate_from_broadcast(report["flights"], broadcast, scan, info.fps, report["calibration"]["shot_cuts"])
    t4 = time.time()
    report["strokes"] = _annotate_strokes(video, report["flights"], flights, ball, players, kin, out, video_sha, cfg,
                                          device)
    timings["pose_s"] = time.time() - t4
    report["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    frames = {
        "fps": info.fps, "n": n,
        "calib_valid": [bool(c.valid) for c in calib],
        "H": [None if c.H is None or not c.valid else [round(float(v), 8) for v in c.H.ravel()] for c in calib],
        "ball": [None if not np.isfinite(ball.xy[i, 0]) else [round(float(ball.xy[i, 0]), 1),
                 round(float(ball.xy[i, 1]), 1), int(ball.source[i])] for i in range(n)],
        "players": {
            side: {
                "box": [None if not np.isfinite(b[0]) else [round(float(v), 1) for v in b]
                        for b in players[side].boxes],
                "world": [None if not np.isfinite(p[0]) else [round(float(p[0]), 3), round(float(p[1]), 3)]
                          for p in kin[side].position],
                "speed_kmh": [None if not np.isfinite(s) else round(float(s), 2) for s in kin[side].speed_kmh],
                "dist_m": [round(float(d), 2) for d in kin[side].distance_m],
            } for side in players
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "frames.json").write_text(json.dumps(frames, separators=(",", ":")))
    link_source(Path(video), out)

    if cfg.render:
        t2 = time.time()
        start_end = broadcast["server_sides"]["start_end"] if scan is not None else None
        render_overlays(video, out, report, frames, ball, players, kin, calib, info, court, prog, scan, start_end)
        timings["render_s"] = time.time() - t2
        report["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
        report["outputs"] = {"classic": "classic.mp4", "pro": "pro.mp4"}
        (out / "report.json").write_text(json.dumps(report, indent=1))
    prog("done", 1.0)
    return report


def _score_rows(scan: BroadcastScan | None, t: float, max_age: float = 20.0) -> list[dict] | None:
    """The last score graphic read at or before ``t`` (stale after ``max_age`` seconds)."""
    if scan is None:
        return None
    seen = [s for s in scan.scores if s.t <= t and t - s.t <= max_age]
    if not seen:
        return None
    r = seen[-1]
    # Graphics often show the 0 in an empty box that OCR does not read: one side has points, so does the other.
    shown = any(row.points for row in r.rows)
    return [{"name": row.name, "games": list(row.games), "points": row.points or ("0" if shown else None),
             "serving": row.serving} for row in r.rows]


def render_overlays(video, out: Path, report: dict, frames: dict, ball, players, kin, calib, info, court,
                    prog: Callable[[str, float], None], scan: BroadcastScan | None = None,
                    start_end: dict[str, str] | None = None) -> None:
    rnd = Renderer(info.width, info.height, court)
    fps = info.fps
    flights = report["flights"]
    bounce_frames = [(f["bounce"]["frame"], f) for f in flights if f["bounce"] is not None and f["call"]]
    w_classic = VideoWriter(out / "classic.mp4", info.width, info.height, fps, audio_from=str(video))
    w_pro = VideoWriter(out / "pro.mp4", info.width, info.height, fps, audio_from=str(video))
    last_speed: float | None = None
    try:
        for i, frame in iter_frames(video):
            c = calib[i]
            for f in flights:
                if f["start_frame"] <= i and f["hit_speed_kmh"] is not None and not f["rejected"]:
                    last_speed = f["hit_speed_kmh"]
            ps = {}
            for side in ("near", "far"):
                b = players[side].boxes[i]
                pos = kin[side].position[i]
                spd = kin[side].speed_kmh[i]
                so_far = kin[side].speed_kmh[:i + 1]
                ps[side] = {"box": b if np.isfinite(b[0]) else None,
                            "world": pos if np.isfinite(pos[0]) else None,
                            "speed_kmh": float(spd) if np.isfinite(spd) else None,
                            "dist_m": float(kin[side].distance_m[i]),
                            "peak_kmh": float(np.nanmax(so_far)) if np.isfinite(so_far).any() else 0.0,
                            "track_id": players[side].track_ids[0] if players[side].track_ids else None}
            bounces, latest = [], None
            for bf, f in bounce_frames:
                if bf <= i:
                    call = f["call"]
                    bounces.append({"xy": [call["x"], call["y"]], "cov": call["cov"], "verdict": call["verdict"],
                                    "p_in": call["p_in"], "age": i - bf, "play_continued": f["play_continued"]})
                    latest = {**call, "speed_kmh": None if f["rejected"] else f["hit_speed_kmh"],
                              "net_clearance_m": f["net_clearance_m"],
                              "play_continued": f["play_continued"],
                              "net_clearance_sd_m": f["net_clearance_sd_m"]}
            trail = [ball.xy[j] for j in range(max(0, i - 10), i + 1) if np.isfinite(ball.xy[j, 0])]
            note = "" if c.valid else c.reason
            st = OverlayState(i, i / fps, c.valid, c.H if c.valid else None, note,
                              ball.xy[i] if np.isfinite(ball.xy[i, 0]) else None, trail, ps, last_speed,
                              bounces[-3:], latest, _score_rows(scan, i / fps),
                              side_names(scan, start_end, i / fps) if scan is not None else None)
            w_classic.write(rnd.classic(frame, st))
            w_pro.write(rnd.pro(frame, st))
            if i % 10 == 0:
                prog("render", i / max(info.n_frames, 1))
    finally:
        w_classic.close()
        w_pro.close()

