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
def test_pipeline_end_to_end_on_real_clip(tmp_path: Path) -> None:
    from courtvision.pipeline import PipelineConfig, analyze

    clip = tmp_path / "clip.mp4"
    # First 2.4 s: one full flight (near-player hit at frame ~9, far bounce at ~47, far hit at ~56).
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SAMPLE), "-t", "2.4", "-c:v", "libx264",
                    "-preset", "veryfast", "-c:a", "aac", str(clip)], check=True)
    report = analyze(clip, tmp_path / "run", PipelineConfig(render=False))

    assert report["schema"] == "courtvision.report/1"
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
