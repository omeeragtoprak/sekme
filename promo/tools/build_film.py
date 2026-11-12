"""Cut the film's shots from the analysis runs and synthesise its sound track.

Every shot is a list of pieces of one source video, each played at a speed factor (slow motion is
motion-interpolated), optionally ending on a held frame. Ball hits and bounces come from the run's
report, are mapped through the same pieces, and drive the sound design: nothing is stock audio.

    .venv/bin/python promo/tools/build_film.py
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "promo" / "assets"
SR = 48_000
OUT_FPS = 30


@dataclass
class Shot:
    name: str
    run: str
    source: str  # pro | classic | raw
    pieces: list[tuple[float, float, float]]  # (src_start, src_end, slow factor)
    hold: float = 0.0  # seconds the last frame is held
    events: list[tuple[float, str]] = field(default_factory=list)  # (local time, hit|bounce)

    @property
    def path(self) -> Path:
        if self.source == "raw":
            return ROOT / "data" / "videos" / f"{self.run}.mp4"
        return ROOT / "runs" / self.run / f"{self.source}.mp4"

    @property
    def duration(self) -> float:
        return sum((b - a) * f for a, b, f in self.pieces) + self.hold

    def local(self, t: float) -> float | None:
        """Source time -> time inside the cut shot (None when not shown)."""
        acc = 0.0
        for a, b, f in self.pieces:
            if a <= t < b:
                return acc + (t - a) * f
            acc += (b - a) * f
        return None


SHOTS = [
    Shot("cold", "usopen2023_final_hd", "raw", [(25.4, 26.75, 1.0), (26.75, 27.24, 3.0)], hold=2.4),
    Shot("court", "usopen2023_final_hd", "classic", [(8.0, 12.5, 1.0)]),
    Shot("track", "usopen2023_final_hd", "pro", [(0.6, 1.55, 1.0), (1.55, 2.1, 3.0), (2.1, 3.3, 1.0)]),
    Shot("call", "usopen2023_final_hd", "pro", [(57.45, 57.98, 1.0), (57.98, 58.3, 2.5), (58.3, 60.6, 1.0)]),
    Shot("doubt", "usopen2023_final_hd", "pro", [(52.0, 53.25, 1.0), (53.25, 53.65, 2.5), (53.65, 54.6, 1.0)],
         hold=0.9),
    Shot("other", "usopen2025_final_hd", "pro", [(58.9, 59.3, 1.0), (59.3, 59.62, 2.5), (59.62, 61.5, 1.0)],
         hold=0.85),
]

# Stills: (name, run, source, time) — frames the film pushes into.
STILLS = [("answer", "usopen2023_final_hd", "pro", 27.27), ("cold_raw", "usopen2023_final_hd", "raw", 27.23),
          ("score", "usopen2023_final_hd", "pro", 59.9)]

# Where each shot sits in the film (seconds). Must match promo/index.html.
PLACE = {"cold": 0.0, "court": 10.0, "track": 14.5, "call": 18.3, "doubt": 21.95, "other": 30.05}
FILM = 43.0
BEATS = {  # sound cues placed in the film, mirrored by the visuals
    "riser_to": 4.5, "impact": [4.5, 25.2], "whoosh": [7.3, 10.0, 14.5, 18.3, 21.95, 26.05, 30.05, 33.95, 36.95],
    "ping": [11.2, 12.2, 13.2, 15.0, 16.4, 17.3, 19.0, 19.4, 20.0, 23.8, 24.45, 27.0, 27.6, 28.2, 30.4, 31.0, 31.6,
             34.6, 37.0, 37.6],
    "logo_land": [7.85, 40.5], "outro_boom": 39.95,
}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def cut(shot: Shot) -> Path:
    parts, labels = [], []
    for k, (a, b, f) in enumerate(shot.pieces):
        chain = f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS"
        if f > 1:
            chain += (f",minterpolate=fps={OUT_FPS * f:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                      f",setpts={f:g}*PTS")
        chain += f",fps={OUT_FPS},scale=1920:1080:flags=lanczos,setsar=1[p{k}]"
        parts.append(chain)
        labels.append(f"[p{k}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0"
    if shot.hold:
        graph += f",tpad=stop_mode=clone:stop_duration={shot.hold}"
    graph += "[v]"
    out = ASSETS / f"f_{shot.name}.mp4"
    ffmpeg("-i", str(shot.path), "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "15",
           "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))
    return out


def still(name: str, run: str, source: str, t: float) -> None:
    src = ROOT / "data" / "videos" / f"{run}.mp4" if source == "raw" else ROOT / "runs" / run / f"{source}.mp4"
    ffmpeg("-ss", f"{t}", "-i", str(src), "-frames:v", "1", "-vf", "scale=1920:1080:flags=lanczos", "-q:v", "2",
           str(ASSETS / f"f_{name}.jpg"))


def shot_events(shot: Shot) -> None:
    report = json.loads((ROOT / "runs" / shot.run / "report.json").read_text())
    fps = report["video"]["fps"]
    for f in report["flights"]:
        for t, kind in ((f["start_frame"] / fps, "hit"), ((f.get("bounce") or {}).get("frame"), "bounce")):
            if t is None:
                continue
            t = t / fps if kind == "bounce" else t
            loc = shot.local(t)
            if loc is not None:
                shot.events.append((loc, kind))


# ---------------------------------------------------------------- sound

def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def pock(rng: np.random.Generator, kind: str, slow: float) -> np.ndarray:
    """Ball on strings (hit) or on hard court (bounce); slowed = pitched down with a longer body."""
    dur = 0.18 * slow
    n = int(SR * dur)
    t = np.arange(n) / SR
    base = (430 if kind == "hit" else 980) / slow
    body = np.sin(2 * np.pi * base * t + 2.5 * np.exp(-t * 60)) * env(n, 0.0008, 0.035 * slow)
    over = 0.45 * np.sin(2 * np.pi * base * 2.7 * t) * env(n, 0.0005, 0.012 * slow)
    click = rng.standard_normal(n) * env(n, 0.0002, 0.003 * slow) * (0.9 if kind == "hit" else 0.5)
    return (body + over + click) * (1.0 if kind == "hit" else 0.65)


def whoosh(rng: np.random.Generator, dur: float = 0.55) -> np.ndarray:
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    shaped = np.diff(np.concatenate([[0], np.cumsum(noise) * 0.02]))  # brown-ish
    shaped = noise * 0.25 + shaped * 3
    return shaped * np.sin(np.pi * t) ** 2 * 0.35


def riser(rng: np.random.Generator, dur: float) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 120 * (1 + 7 * (t / dur) ** 2)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    noise = rng.standard_normal(n) * 0.18
    return (tone + noise) * (t / dur) ** 2.2


def impact(rng: np.random.Generator) -> np.ndarray:
    n = int(SR * 2.2)
    t = np.arange(n) / SR
    f = 34 + 70 * np.exp(-t * 9)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.55)
    crack = rng.standard_normal(n) * env(n, 0.0005, 0.03) * 0.5
    return (sub * 0.95 + crack)


def ping(freq: float = 1760) -> np.ndarray:
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 1.5 * t)) * env(n, 0.002, 0.09) * 0.22


def drone(n: int) -> np.ndarray:
    """Two detuned low voices on A and E with a slow breathing filter — a bed, not a tune."""
    t = np.arange(n) / SR
    v = np.zeros(n)
    for f, a in ((55.0, 0.5), (55.4, 0.4), (82.4, 0.28), (110.2, 0.12), (164.8, 0.05)):
        v += a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.07 * t))
    return v * (0.75 + 0.25 * np.sin(2 * np.pi * 0.11 * t))


def reverb(x: np.ndarray, rng: np.random.Generator, seconds: float = 1.6, mix: float = 0.22) -> np.ndarray:
    n = int(SR * seconds)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 4.2)
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x + mix * wet


def add(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += gain * sig[: j - i]


def sound(shots: list[Shot]) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(FILM * SR)
    fx = np.zeros(n)
    bed = drone(n)
    # Bed level over the film: tension in the cold open, breath at the freeze, lift after the logo.
    pts = [(0, 0.0), (0.4, 0.35), (2.8, 0.6), (2.9, 0.12), (4.45, 0.12), (4.5, 0.75), (7.2, 0.45), (7.5, 0.2),
           (10.0, 0.4), (21.95, 0.45), (23.0, 0.3), (25.2, 0.55), (30.0, 0.45), (39.9, 0.5), (40.5, 0.75),
           (FILM - 0.3, 0.0)]
    tt = np.arange(n) / SR
    bed *= np.interp(tt, [p[0] for p in pts], [p[1] for p in pts]) * 0.32
    for shot in shots:
        for loc, kind in shot.events:
            at = PLACE[shot.name] + loc
            slow = 1.0
            for a, b, f in shot.pieces:  # pieces played slowed get pitched-down sounds
                la, lb = shot.local(a), (shot.local(a) or 0) + (b - a) * f
                if la is not None and la <= loc < lb:
                    slow = f
            add(fx, pock(rng, kind, slow), at, 0.8 if slow > 1 else 0.55)
    add(fx, riser(rng, 1.5), BEATS["riser_to"] - 1.5, 0.6)
    for at in BEATS["impact"]:
        add(fx, impact(rng), at, 0.75 if at == 4.5 else 0.4)
    for at in BEATS["whoosh"]:
        add(fx, whoosh(rng), at - 0.3, 0.6)
    for k, at in enumerate(BEATS["ping"]):
        add(fx, ping(1760 if k % 2 == 0 else 2093), at, 0.5)
    for at in BEATS["logo_land"]:
        add(fx, pock(rng, "bounce", 1.6), at, 0.7)
    add(fx, impact(rng), BEATS["outro_boom"], 0.6)
    fx = reverb(fx, rng)
    mix = bed + fx
    # Fade the ends, gentle soft clip, normalise to -1 dBFS; slight stereo width from a 9 ms offset.
    fade = int(0.3 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))
    d = int(0.009 * SR)
    right = np.concatenate([np.zeros(d), mix[:-d]]) * 0.5 + mix * 0.5
    return np.stack([mix, right], axis=1)


def write_wav(path: Path, stereo: np.ndarray) -> None:
    import wave

    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    for s in SHOTS:
        cut(s)
        shot_events(s)
        print(f"{s.name:7s} at {PLACE[s.name]:5.1f}s  {s.duration:4.2f}s  events "
              + ", ".join(f"{k[0]}{PLACE[s.name] + t:.2f}" for t, k in s.events))
    for spec in STILLS:
        still(*spec)
    raw = ASSETS / "score_raw.wav"
    write_wav(raw, sound(SHOTS))
    # Loudness for social feeds (about -17 LUFS); the low-pass and limiter keep AAC true peaks under 0 dB.
    chain = ("loudnorm=I=-16:TP=-1.5:LRA=11,volume=3dB,lowpass=f=15000,"
             "alimiter=limit=0.56:attack=1:release=50:level=false")
    ffmpeg("-i", str(raw), "-af", chain, "-ar", str(SR), str(ASSETS / "score.wav"))
    raw.unlink()
    print("ok")


if __name__ == "__main__":
    main()
