"""Video and audio I/O via OpenCV and the ffmpeg/ffprobe binaries.

PyAV is deliberately not used: its bundled FFmpeg clashes with the one inside
opencv-python on macOS (duplicate AVFoundation classes in one process).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH")
    return path


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    meta = json.loads(out)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(v.get("avg_frame_rate") or v["r_frame_rate"]))
    duration = float(meta["format"].get("duration", v.get("duration", 0.0)))
    n = int(v.get("nb_frames") or round(duration * fps))
    has_audio = any(s["codec_type"] == "audio" for s in meta["streams"])
    return VideoInfo(str(path), int(v["width"]), int(v["height"]), fps, n, duration, has_audio)


def iter_frames(path: str | Path, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) sequentially; decoding dominates, so we never seek per frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while end is None or i < end:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def extract_audio(path: str | Path, sample_rate: int = 22050) -> np.ndarray:
    """Mono float32 audio in [-1, 1]; empty array when the file has no audio."""
    proc = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-f", "f32le", "pipe:1"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


class VideoWriter:
    """H.264 MP4 writer that pipes raw frames into ffmpeg (browser-playable output)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, audio_from: str | None = None):
        cmd = [_bin("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0"]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._size = (width, height)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing video")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
