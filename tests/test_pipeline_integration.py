"""End-to-end run on a real broadcast clip (skipped when weights or the sample video are absent)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "videos" / "vienna_djokovic_sonego.mp4"
WEIGHTS = [ROOT / "models" / n for n in ("sekme_court.pt", "sekme_ball.pt", "yolo26s.pt")]

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not SAMPLE.exists() or not all(w.exists() for w in WEIGHTS) or not shutil.which("ffmpeg"),
                    reason="sample clip / model weights / ffmpeg not available")
def test_pipeline_end_to_end_on_real_clip(tmp_path: Path, monkeypatch) -> None:
    from courtvision import pipeline
    from courtvision.pipeline import PipelineConfig, analyze

    calls: list[int] = []  # far strokes from 2D breaks run by default (adopted 2026-11-17, docs/EVAL_SPOT.md)
    real_far_breaks = pipeline.far_breaks
    monkeypatch.setattr(pipeline, "far_breaks", lambda *a, **k: calls.append(1) or real_far_breaks(*a, **k))
    moved: list[int] = []  # and the far contact from a line and a curve (adopted 2026-11-17, test slot 2)
    real_move = pipeline.move_far_contacts
    monkeypatch.setattr(pipeline, "move_far_contacts", lambda *a, **k: moved.append(1) or real_move(*a, **k))

    clip = tmp_path / "clip.mp4"
    # First 2.4 s: one full flight (near-player hit at frame ~9, far bounce at ~47, far hit at ~56).
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SAMPLE), "-t", "2.4", "-c:v", "libx264",
                    "-preset", "veryfast", "-c:a", "aac", str(clip)], check=True)
    report = analyze(clip, tmp_path / "run", PipelineConfig(render=False))

    assert report["schema"] == "courtvision.report/1"
    assert calls == [1] and moved == [1] and report["candidates"] == {}  # adopted, not candidates: no flag recorded
    cal = report["calibration"]
    assert cal["valid_frames"] == cal["total_frames"]
    assert cal["line_rmse_px_median"] < 1.5
    # Broadcast main camera: on the court axis, tens of metres back, ~10-15 m up.
    cx, cy, cz = cal["camera_center_m"]
    assert abs(cx) < 2.0 and -60 < cy < -30 and 8 < cz < 20
    assert report["ball"]["detected_frames"] >= 0.9 * cal["total_frames"]
    assert {h["side"] for h in report["hits"]} >= {"near"}
    called = [f for f in report["flights"] if f["call"]]
    assert called, report["flights"]
    f = called[0]
    assert f["call"]["verdict"] == "IN"  # play continued: the umpire did not call it out
    assert 60 < f["hit_speed_kmh"] < 180 and 0.0 < f["hit_height_m"] < 3.3 and f["net_clearance_m"] > 0
    assert json.loads((tmp_path / "run" / "frames.json").read_text())["n"] == cal["total_frames"]
    # Frame differences are measured during perception, the same as a pass of their own; one shot, no edit.
    import pickle

    import numpy as np

    from courtvision.perception.court import frame_differences
    with open(tmp_path / "run" / "perception.pkl", "rb") as fh:  # written by this test
        diffs = pickle.load(fh)["diffs"]
    assert np.array_equal(diffs, frame_differences(clip)) and cal["same_angle_edits"] == []
    assert not (tmp_path / "run" / "frame_diffs.pkl").exists()  # no extra pass when perception measured them


@pytest.mark.skipif(not SAMPLE.exists() or not all(w.exists() for w in WEIGHTS) or not shutil.which("ffmpeg"),
                    reason="sample clip / model weights / ffmpeg not available")
def test_a_variable_rate_clip_is_analysed_and_played_from_its_constant_rate_copy(tmp_path: Path) -> None:
    from courtvision.pipeline import CONSTANT_RATE_FILE, PipelineConfig, analyze

    clip = tmp_path / "gappy.mp4"
    # The sample's first 2.4 s with frames 20-49 at two thirds of the rate (every third one removed, timestamps kept).
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SAMPLE), "-t", "2.4", "-vf",
                    "select='not(between(n\\,20\\,49)*eq(mod(n\\,3)\\,1))'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "veryfast", "-an", str(clip)], check=True)
    run = tmp_path / "run"
    report = analyze(clip, run, PipelineConfig(render=False, broadcast=False))
    v = report["video"]
    assert v["file"] == "gappy.mp4" and v["irregular_frame_intervals"] == 10
    copy = v["constant_rate_copy"]
    assert copy["file"] == CONSTANT_RATE_FILE and copy["added_frames"] == 10 and copy["frames"] == v["n_frames"]
    assert copy["repeated_frames"] == 10 and copy["repeated_ball_dropped"] and not report["candidates"]
    assert (run / "source.mp4").resolve() == (run / CONSTANT_RATE_FILE).resolve()  # the Review Room plays the copy
    assert "sabit" in report["limitations"][0] and report["timings_s"]["constant_rate_copy_s"] >= 0
    assert json.loads((run / "frames.json").read_text())["n"] == v["n_frames"]
    written = (run / CONSTANT_RATE_FILE).stat().st_mtime_ns
    again = analyze(clip, run, PipelineConfig(render=False, broadcast=False, reuse_perception=True))
    assert (run / CONSTANT_RATE_FILE).stat().st_mtime_ns == written  # reused, not re-encoded
    assert again["video"]["constant_rate_copy"]["sha256"] == copy["sha256"]
    assert [f["call"] for f in again["flights"]] == [f["call"] for f in report["flights"]]
    repeats = json.loads((run / "constant_rate.json").read_text())["repeated"]
    assert repeats == list(range(22, 50, 3))  # the slots of the removed frames
    assert again["video"]["constant_rate_copy"]["repeated_ball_dropped"] and "10 karede top yok" in \
        again["limitations"][0]
    ball = json.loads((run / "frames.json").read_text())["ball"]
    assert not [i for i in repeats if ball[i] and ball[i][2] == 1]  # never detected on a repeated frame
    kept = analyze(clip, run, PipelineConfig(render=False, broadcast=False, reuse_perception=True,
                                             drop_repeated_ball=False))
    assert not kept["video"]["constant_rate_copy"]["repeated_ball_dropped"]
    detected = [i for i, b in enumerate(json.loads((run / "frames.json").read_text())["ball"]) if b and b[2] == 1]
    assert set(repeats) & set(detected)  # without it the ball is "seen" on repeated frames, one frame late


def test_perception_cache_key_changes_with_the_networks(tmp_path):
    from courtvision.pipeline import PipelineConfig, perception_key

    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    cfg = PipelineConfig()
    cfg.models = {"court": str(a), "ball": str(a), "players": str(a), "pose": str(a)}
    k1 = perception_key("video", cfg)
    cfg.models["ball"] = str(b)
    assert perception_key("video", cfg) != k1  # another ball network must not reuse cached detections
    cfg.models["ball"] = str(a)
    cfg.ball_stride = 2
    assert perception_key("video", cfg) != k1


def test_model_card_reads_our_checkpoints_with_numpy_scores(tmp_path):
    import numpy as np
    import torch

    from courtvision.pipeline import _model_card

    path = tmp_path / "ball.pt"
    torch.save({"arch": "sekme-ball-unet", "model": {}, "epoch": 10, "val": {"f1": np.float64(0.85)},
                "trained_on": "TrackNet tennis games 1-7"}, path)
    card = _model_card("ball", str(path))  # a report must not fail on the checkpoint's own metadata
    assert card["trained_on"] == "TrackNet tennis games 1-7" and card["epoch"] == 10
    assert card["source"].startswith("this project")


def test_perception_cache_key_covers_the_settings_and_reads_old_caches_as_defaults(tmp_path):
    from courtvision.pipeline import PipelineConfig, perception_key, same_perception_key

    a = tmp_path / "a.pt"
    a.write_bytes(b"one")
    cfg = PipelineConfig()
    cfg.models = {"court": str(a), "ball": str(a), "players": str(a), "pose": str(a)}
    k = perception_key("video", cfg)
    old = {kk: v for kk, v in k.items() if kk != "settings"}  # a cache written before the settings were keyed
    assert same_perception_key(old, k) and same_perception_key(k, k)
    for name, value in (("keyframe_interval", 30), ("cut_threshold", 0.6), ("player_imgsz", 960)):
        c = PipelineConfig()
        c.models = cfg.models
        setattr(c, name, value)
        assert not same_perception_key(k, perception_key("video", c))
        assert not same_perception_key(old, perception_key("video", c))  # an old cache had the defaults
    # A cache whose key is unknown (written with a null key, or none at all) is recomputed, not a crash.
    assert not same_perception_key(None, k) and not same_perception_key({}, k)


def test_pose_cache_key_covers_the_hitter_and_its_box():
    import numpy as np

    from courtvision.pipeline import pose_key

    box = np.array([10.0, 20.0, 60.0, 180.0])
    k = pose_key("video", "model", [(100, "near", box)])
    assert k == pose_key("video", "model", [(100, "near", box.copy())])
    assert k != pose_key("video", "model", [(100, "far", box)])
    assert k != pose_key("video", "model", [(100, "near", box + 30)])  # another player picked at that frame
    nan = np.full(4, np.nan)
    assert pose_key("v", "m", [(5, "near", nan)]) == pose_key("v", "m", [(5, "near", nan.copy())])


def test_cached_detections_are_not_reused_with_another_ball_step(tmp_path):
    from courtvision.pipeline import PipelineConfig, perception_key, same_perception_key

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")  # (the real checkpoints are not in the repository)
    cfg = PipelineConfig()
    cfg.models = {k: str(weights) for k in ("court", "ball", "players", "pose")}
    k30, k60 = perception_key("video", cfg, 1), perception_key("video", cfg, 2)
    assert same_perception_key(k30, k30) and not same_perception_key(k30, k60)  # e.g. the frame rate read anew
    old = {k: v for k, v in k60.items() if k != "ball_step"}  # written before the step was part of the key
    assert same_perception_key(old, k60) and same_perception_key(old, k30)
