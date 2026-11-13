"""The E2E-Spot clips must start exactly at their labelled frame (scripts/eval_spot.py run_match)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
W, H = 288, 64


def _bars(i: int) -> np.ndarray:  # the frame number as 9 black/white bars: survives lossy coding
    img = np.zeros((H, W, 3), np.uint8)
    for b in range(9):
        if (i >> b) & 1:
            img[:, b * 32:(b + 1) * 32] = 255
    return img


def _read(img: np.ndarray) -> int:
    return sum(1 << b for b in range(9) if img[:, b * 32 + 8:(b + 1) * 32 - 8, 1].mean() > 128)


def test_clips_start_and_end_on_their_labelled_frames(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_spot

    src = tmp_path / "match.mp4"  # B-frames and a non-zero start time, as in real uploads
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-bf", "3", "-g", "48",
                          "-pix_fmt", "yuv420p", "-output_ts_offset", "1.5", str(src)], stdin=subprocess.PIPE)
    for i in range(130):
        p.stdin.write(_bars(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    fps = cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FPS)
    clips = [{"video": f"m_{a}_{b}", "fps": fps, "num_frames": b - a} for a, b in ((0, 20), (50, 80), (70, 90))]
    seen = {}

    def fake_analyse(clip: Path, run: Path) -> None:
        cap, frames = cv2.VideoCapture(str(clip)), []
        while (r := cap.read())[0]:
            frames.append(_read(r[1]))
        seen[run.name] = (frames[0], frames[-1], len(frames))
        run.mkdir(parents=True)
        (run / "report.json").write_text("{}")

    monkeypatch.setattr(eval_spot, "analyse", fake_analyse)
    eval_spot.run_match(src, clips, tmp_path / "runs", tmp_path / "tmp", keep=False)
    assert seen == {"m_0_20": (0, 19, 20), "m_50_80": (50, 79, 30), "m_70_90": (70, 89, 20)}  # overlap too
    assert not list((tmp_path / "tmp").glob("*.mp4"))  # clips are deleted once analysed
    with pytest.raises(SystemExit, match="ended at frame 130"):
        eval_spot.run_match(src, [{"video": "m_120_140", "fps": fps, "num_frames": 20}], tmp_path / "r2",
                            tmp_path / "tmp", keep=False)
    with pytest.raises(SystemExit, match="not the labelled upload"):
        eval_spot.run_match(src, [{"video": "m_0_10", "fps": 25.0, "num_frames": 10}], tmp_path / "r3",
                            tmp_path / "tmp", keep=False)
