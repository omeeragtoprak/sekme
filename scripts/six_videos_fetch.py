"""Find (``list``, metadata only) and, after the user's yes, download (``download``) the six videos named in
docs/EVAL_SERVE_SPEED.md: the official US Open channel's highlight video of each match, the longest if several.

    .venv/bin/python scripts/six_videos_fetch.py list                 # ids, titles, durations, 1080p sizes
    .venv/bin/python scripts/six_videos_fetch.py download --ids ...   # only after the user's explicit yes

``download`` writes data/six/videos/<name>.mp4 and data/six/manifest.json (video id, title, sha256), which
the plan asks to record before any labelling or analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YTDLP = str(ROOT / ".venv" / "bin" / "yt-dlp")
OUT = ROOT / "data" / "six"
CHANNEL = "US Open Tennis Championships"
MATCHES = {  # name -> (search query, words the title must contain)
    "w2024_final_sabalenka_pegula": ("US Open 2024 women's final Sabalenka Pegula highlights", ["sabalenka", "pegula"]),
    "m2024_sf_sinner_draper": ("US Open 2024 semifinal Sinner Draper highlights", ["sinner", "draper"]),
    "m2024_sf_fritz_tiafoe": ("US Open 2024 semifinal Fritz Tiafoe highlights", ["fritz", "tiafoe"]),
    "w2025_final_sabalenka_anisimova": ("US Open 2025 women's final Sabalenka Anisimova highlights",
                                        ["sabalenka", "anisimova"]),
    "m2025_sf_alcaraz_djokovic": ("US Open 2025 semifinal Alcaraz Djokovic highlights", ["alcaraz", "djokovic"]),
    "m2025_sf_sinner_auger_aliassime": ("US Open 2025 semifinal Sinner Auger-Aliassime highlights",
                                        ["sinner", "auger"]),
}


def search(query: str, n: int = 12) -> list[dict]:
    r = subprocess.run([YTDLP, f"ytsearch{n}:{query}", "--dump-json", "--skip-download", "--no-warnings"],
                       capture_output=True, text=True)
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip().startswith("{")]


def size_1080(info: dict) -> int | None:
    fmts = [f for f in info.get("formats", []) if f.get("height") and f["height"] <= 1080]
    video = max((f for f in fmts if f.get("vcodec") not in (None, "none")), key=lambda f: (f["height"],
                f.get("tbr") or 0), default=None)
    if video is None:
        return None
    s = video.get("filesize") or video.get("filesize_approx")
    if not s and video.get("tbr") and info.get("duration"):
        s = video["tbr"] * 1000 / 8 * info["duration"]  # estimate from the stream's bitrate (kbit/s)
    return int(s) if s else None


def pick(name: str) -> dict | None:
    query, words = MATCHES[name]
    cands = [i for i in search(query) if i.get("channel") == CHANNEL and "highlight" in i.get("title", "").lower()
             and all(w in i.get("title", "").lower() for w in words)]
    return max(cands, key=lambda i: i.get("duration") or 0) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "download"))
    ap.add_argument("--ids", nargs="*", default=[], help="name=video_id pairs approved by the user")
    a = ap.parse_args()
    if a.cmd == "list":
        total = 0
        for name in MATCHES:
            i = pick(name)
            if i is None:
                print(f"{name}: no official highlight video found")
                continue
            s = size_1080(i)
            total += s or 0
            print(f"{name}: {i['id']} | {i['title']} | {i.get('duration')} s | "
                  f"{i.get('fps') or '?'} fps | 1080p video ~{(s or 0) / 1e6:.0f} MB")
        print(f"total ~{total / 1e6:.0f} MB (video streams; audio is not kept)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    for pair in a.ids:
        name, vid = pair.split("=", 1)
        if name not in MATCHES:
            sys.exit(f"unknown name {name}")
        dst = OUT / "videos" / f"{name}.mp4"
        subprocess.run([YTDLP, "-f", "bv*[height<=1080][ext=mp4]", "-o", str(dst), "--no-warnings",
                        f"https://www.youtube.com/watch?v={vid}"], check=True)
        manifest[name] = {"video_id": vid, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"{name}: {dst} sha256 {manifest[name]['sha256'][:16]}")


if __name__ == "__main__":
    main()
