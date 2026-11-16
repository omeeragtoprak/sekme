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


MAX_FPS = 1000.0  # an average frame rate above this is metadata noise, not video
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
        avg, stated = rate(v.get("avg_frame_rate")), rate(v.get("r_frame_rate"))
        # Some WebM / Matroska files report a nonsensical average ("30000/1" for 29.97 fps video, seen on a
        # downloaded clip); the stream's stated rate is then the one to trust. A real average is never far above it.
        fps = avg if avg is not None and avg <= MAX_FPS and (stated is None or avg <= 1.5 * stated) else stated
        if fps is None:
            raise UnreadableVideoError(f"{name}: video akışında kare hızı yok")
        duration = float(meta["format"].get("duration", v.get("duration", 0.0)))
        try:  # a stated count of "0" or "N/A" is no count: fall back to duration * fps
            n = int(v.get("nb_frames") or 0)
        except ValueError:
            n = 0
        n = n if n > 0 else round(duration * fps)
        has_audio = any(s.get("codec_type") == "audio" for s in meta["streams"])
    except UnreadableVideoError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:  # malformed metadata ("N/A", missing keys)
        raise UnreadableVideoError(f"{name}: video bilgileri okunamadı") from exc
    return VideoInfo(p, width, height, fps, n, duration, has_audio)


def _frame_intervals(path: str | Path) -> np.ndarray | None:
    """Positive intervals between the video packets' timestamps (in presentation order), or None when they cannot
    be read."""
    try:  # "V": video streams that are not attached pictures (cover art)
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-select_streams", "V:0", "-show_entries",
                              "packet=pts_time", "-of", "csv=p=0", str(path)],
                             check=True, capture_output=True, text=True, timeout=120).stdout
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    times = []
    for field_ in out.split():
        try:
            times.append(float(field_.strip(",")))
        except ValueError:  # "N/A": a packet without a timestamp; the spacing cannot be judged
            return None
    d = np.diff(np.sort(np.array(times)))
    d = d[d > 0]  # duplicated timestamps say nothing about the spacing
    return d if len(d) >= 2 and np.median(d) > 0 else None


def irregular_frame_intervals(path: str | Path) -> int:
    """Frame intervals that differ from the video's median interval by more than 25 % (variable frame rate, e.g.
    a phone dropping frames in low light), from the packets' timestamps. 0 when they cannot be read."""
    d = _frame_intervals(path)
    if d is None:
        return 0
    r = d / np.median(d)
    return int(((r > 1.25) | (r < 0.75)).sum())


def nominal_rate(path: str | Path) -> Fraction | None:
    """The rate a variable-rate video was meant to run at: its stream's `r_frame_rate` when that is within 5 % of
    the rate of its median frame interval, else the median's rate. None when the timestamps cannot be read."""
    d = _frame_intervals(path)
    if d is None:
        return None
    median_rate = 1.0 / float(np.median(d))
    try:
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-select_streams", "V:0", "-show_entries",
                              "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
                             check=True, capture_output=True, text=True, timeout=60).stdout.strip().strip(",")
        stated = Fraction(out)
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError,
            ZeroDivisionError):
        stated = None
    if stated is not None and stated > 0 and abs(float(stated) / median_rate - 1) <= 0.05:
        return stated
    rate = Fraction(median_rate).limit_denominator(1001)
    return rate if rate > 0 else None


_SHOWINFO = re.compile(r"\[Parsed_showinfo_\d+ @ [^\]]*\]\s*n:\s*(\d+)\s.*?\bchecksum:([0-9A-Fa-f]+)")


def repeated_frames(showinfo_log: str) -> tuple[int, list[int]]:
    """(frames shown, indices of frames whose picture checksum equals the previous frame's) from ffmpeg `showinfo`
    lines: after the `fps` filter those are the frames it repeated to fill a gap. A source frame that is itself
    identical to the one before (a frozen picture) counts too; it shows nothing new either."""
    sums = dict((int(m.group(1)), m.group(2)) for m in _SHOWINFO.finditer(showinfo_log))
    n = max(sums, default=-1) + 1
    return n, [i for i in range(1, n) if i in sums and sums[i] == sums.get(i - 1)]


INTERLACED_ORDERS = frozenset({"tt", "bb", "tb", "bt"})  # ffprobe field_order values of interlaced video


def field_order(path: str | Path) -> str | None:
    """The video stream's field order as ffprobe reports it ("progressive", "tt", "bb", ...; "unknown" when the file
    does not say), None when it cannot be read."""
    try:
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-select_streams", "V:0", "-show_entries",
                              "stream=field_order", "-of", "csv=p=0", str(path)],
                             check=True, capture_output=True, text=True, timeout=60).stdout.strip().strip(",")
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out or None


def constant_rate_copy(src: str | Path, dest: str | Path, rate: Fraction) -> tuple[Path, list[int] | None]:
    """Re-encode `src` at a constant `rate` with ffmpeg's `fps` filter: each output frame is the source frame
    nearest its time, so a gap repeats a frame and index / rate is the real time again. Upright (rotation applied),
    H.264 for the browser (an odd last pixel row or column is cropped: 4:2:0 needs even sizes), audio kept as AAC.
    Written to a temporary name first: `dest` is complete or absent. Also returns the indices of the copy's
    repeated frames (same picture as the frame before), None when ffmpeg's frame log could not be read or does not
    have one line per frame of the copy (the indices would not be the copy's). The log is held in memory: about
    300 bytes a frame, 65 MB for an hour at 60 fps."""
    dest = Path(dest)
    tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
    if Path(src).resolve() in (dest.resolve(), tmp.resolve()):
        raise ValueError(f"{dest.name}: the copy would overwrite its source")
    vf = f"fps={rate.numerator}/{rate.denominator},crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0,showinfo"
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostats", "-v", "info", "-y", "-i", str(src), "-map", "0:V:0",
           "-map", "0:a:0?", "-vf", vf, "-c:v", "libx264", "-crf", "17", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        log = "\n".join(ln for ln in res.stderr.splitlines() if "Parsed_showinfo" not in ln)
        raise UnreadableVideoError(f"{Path(src).name}: sabit kare hızlı kopya yapılamadı "
                                   f"({_reason(log, str(src), Path(src).name)})")
    n, repeated = repeated_frames(res.stderr)
    try:
        written = probe(tmp).n_frames
    except UnreadableVideoError:
        written = -1
    tmp.replace(dest)
    return dest, (repeated if n and n == written else None)


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


SOURCE_COPY_RECORD = ".source_copy.json"  # names the copy link_source made where links are unsupported


def link_source(video: Path, run: Path) -> None:
    """Expose the analysed video inside the run folder (symlink; copy when links are unsupported). A link or a
    copy of another video (or of an older state of this one) is replaced; a file it did not make is never removed."""
    dest = run / f"source{video.suffix.lower()}"
    for old in run.glob("source.*"):  # re-analysed from another video (or its constant-rate copy): do not show
        if old.is_symlink() and (old != dest or old.resolve() != video.resolve()):  # the old one beside the new
            old.unlink()  # report
    st = video.stat()
    this = {"of": str(video.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    record = run / SOURCE_COPY_RECORD
    try:
        made = json.loads(record.read_text()) if record.is_file() else None
        made_file = run / Path(str(made["file"])).name if isinstance(made, dict) else None
    except (OSError, ValueError, KeyError):
        made, made_file = None, None
    if made_file is not None and ({**this, "file": dest.name} != made or not made_file.is_file()):
        if made_file.is_file() and not made_file.is_symlink():
            made_file.unlink()  # our copy of something else: it would be offered instead of the new video
        record.unlink(missing_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(video.resolve())
    except OSError:
        shutil.copy2(video, dest)
        record.write_text(json.dumps({**this, "file": dest.name}))


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
