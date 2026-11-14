"""Video and audio I/O via OpenCV and the ffmpeg/ffprobe binaries.

PyAV is deliberately not used: its bundled FFmpeg clashes with the one inside
opencv-python on macOS (duplicate AVFoundation classes in one process).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from courtvision.errors import UnreadableVideoError


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


_FFMPEG_TAG = re.compile(r"\[[^\]]*@ 0x[0-9a-fA-F]+\]\s*")  # "[mov,mp4,m4a @ 0x7f…] " before a message


def _reason(stderr: str, path: str, name: str) -> str:
    """ffprobe's last message, without the folder or the decoder's memory address, short enough to show."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    why = (lines[-1] if lines else "ffprobe failed").removeprefix(f"{path}: ").replace(path, name)
    why = _FFMPEG_TAG.sub("", why).strip()
    return why if len(why) <= 120 else why[:119] + "…"


def _rotation(stream: dict) -> int:
    """Display rotation of a video stream in degrees (0 when none or unreadable): the display matrix, else the
    old tag. A malformed value must not make an otherwise readable video unreadable."""
    values = [sd["rotation"] for sd in stream.get("side_data_list") or []
              if sd.get("side_data_type") == "Display Matrix" and "rotation" in sd]
    values.append((stream.get("tags") or {}).get("rotate", 0))
    try:
        return int(round(float(values[0])))
    except (TypeError, ValueError, OverflowError):
        return 0


def probe(path: str | Path) -> VideoInfo:
    p, name = str(path), Path(path).name
    res = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", p],
        check=False, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise UnreadableVideoError(f"{name}: video okunamadı ({_reason(res.stderr, p, name)})")

    def rate(s: str | None) -> float | None:  # "0/0" (no average rate) counts as missing
        if not s or s.endswith("/0"):
            return None
        r = float(Fraction(s))
        return r if r > 0 else None

    try:
        meta = json.loads(res.stdout)
        # Cover art in an audio file is a one-picture "video" stream (with a 90000/1 rate): not a video.
        v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
        if v is None:
            raise UnreadableVideoError(f"{name}: dosyada video akışı yok")
        width, height = int(v.get("width") or 0), int(v.get("height") or 0)
        if width <= 0 or height <= 0:
            raise UnreadableVideoError(f"{name}: video akışının boyutu yok")
        if _rotation(v) % 180 == 90:  # stored sideways (phones): OpenCV and ffmpeg decode the frames upright
            width, height = height, width
        fps = rate(v.get("avg_frame_rate")) or rate(v.get("r_frame_rate"))
        if fps is None:
            raise UnreadableVideoError(f"{name}: video akışında kare hızı yok")
        duration = float(meta["format"].get("duration", v.get("duration", 0.0)))
        n = int(v.get("nb_frames") or round(duration * fps))
        has_audio = any(s.get("codec_type") == "audio" for s in meta["streams"])
    except UnreadableVideoError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:  # malformed metadata ("N/A", missing keys)
        raise UnreadableVideoError(f"{name}: video bilgileri okunamadı") from exc
    return VideoInfo(p, width, height, fps, n, duration, has_audio)


def irregular_frame_intervals(path: str | Path) -> int:
    """Frame intervals that differ from the video's median interval by more than 25 % (variable frame rate, e.g.
    a phone dropping frames in low light), from the packets' timestamps. 0 when they cannot be read."""
    try:
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "packet=pts_time", "-of", "csv=p=0", str(path)],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return 0
    times = []
    for field_ in out.split():
        try:
            times.append(float(field_.strip(",")))
        except ValueError:  # "N/A": a packet without a timestamp
            continue
    d = np.diff(np.sort(np.array(times)))
    if len(d) < 2 or not np.median(d) > 0:
        return 0
    r = d / np.median(d)
    return int(((r > 1.25) | (r < 0.75)).sum())


def open_video(path: str | Path) -> cv2.VideoCapture:
    """OpenCV capture that decodes rotated (phone) video upright, as `probe` reports its size. Set explicitly:
    the default depends on the OpenCV build."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def iter_frames(path: str | Path, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) sequentially; decoding dominates, so we never seek per frame."""
    cap = open_video(path)
    if not cap.isOpened():
        raise UnreadableVideoError(f"{Path(path).name}: video açılamadı")
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


def link_source(video: Path, run: Path) -> None:
    """Expose the analysed video inside the run folder (symlink; copy when links are unsupported)."""
    dest = run / f"source{video.suffix.lower()}"
    if dest.is_symlink() and dest.resolve() != video.resolve():
        dest.unlink()  # re-analysed from another video: do not show the old one beside the new report
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(video.resolve())
    except OSError:
        shutil.copy2(video, dest)


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
