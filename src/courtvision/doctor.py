"""`sekme doctor`: is this machine ready to analyse videos and serve the Review Room?

Each check says what it found. A missing essential (ffmpeg / ffprobe, OpenCV video reading, a network checkpoint)
makes the analysis fail, so the command exits 1; the rest (no GPU, no Apple Vision OCR, an unbuilt web app, little
disk space) only make it slower or poorer, and are warnings.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_FREE_GB = 5.0  # a 10-minute 1080p analysis writes about 1-2 GB (overlay videos, caches)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    essential: bool
    detail: str


def _version(binary: str) -> str | None:
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    first = out.splitlines()[0] if out else ""
    return first.split()[2] if first.startswith(f"{binary} version") and len(first.split()) > 2 else "?"


def _ffmpeg() -> Check:
    found = {b: _version(b) for b in ("ffmpeg", "ffprobe")}
    missing = [b for b, v in found.items() if v is None]
    if missing:
        return Check("ffmpeg", False, True, f"{' ve '.join(missing)} bulunamadı (macOS: brew install ffmpeg)")
    return Check("ffmpeg", True, True, ", ".join(f"{b} {v}" for b, v in found.items()))


def _opencv() -> Check:
    try:
        import cv2
    except ImportError as exc:
        return Check("OpenCV", False, True, f"içe aktarılamadı ({exc})")
    backends = {cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()}
    if not backends & {"FFMPEG", "AVFOUNDATION"}:
        return Check("OpenCV", False, True, f"{cv2.__version__}, video okuyamıyor (arka uçlar: {sorted(backends)})")
    readers = ", ".join(sorted(backends & {"FFMPEG", "AVFOUNDATION"}))
    return Check("OpenCV", True, True, f"{cv2.__version__}, video okuma: {readers}")


def _torch() -> Check:
    try:
        import torch

        from courtvision.perception.court import pick_device
    except ImportError as exc:
        return Check("PyTorch", False, True, f"içe aktarılamadı ({exc})")
    dev = pick_device("auto").type
    if dev == "cpu":
        return Check("PyTorch", False, False, f"{torch.__version__}, hızlandırıcı yok: analiz yalnız işlemcide, çok "
                                              "daha yavaş")
    return Check("PyTorch", True, False, f"{torch.__version__}, hızlandırıcı: {dev}")


def _models(root: Path) -> list[Check]:
    from courtvision.pipeline import MODEL_FILES, _sha256

    out = []
    for key, rel in MODEL_FILES.items():
        path = root / rel
        if not path.is_file():
            out.append(Check(f"ağ {key}", False, True, f"{rel} yok (ağırlıklar repoda değil, docs/MODELS.md)"))
            continue
        detail = f"{rel}, sha256 {_sha256(path)[:12]}"
        if key == "ball":
            from courtvision.perception.ball import ball_arch

            try:
                detail += f", {ball_arch(str(path)) or 'WASB'}"
            except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint is reported, not raised
                out.append(Check(f"ağ {key}", False, True, f"{rel} okunamadı ({type(exc).__name__})"))
                continue
        out.append(Check(f"ağ {key}", True, True, detail))
    return out


def _ocr() -> Check:
    from courtvision.broadcast import ocr

    if ocr.available():
        return Check("Apple Vision OCR", True, False, "skor grafiği ve hız tabelası okunur")
    return Check("Apple Vision OCR", False, False, "yok: skor grafiği, isimler ve hız tabelası okunmaz (yalnız "
                                                   "macOS, pyobjc-framework-Vision)")


def _web(root: Path) -> Check:
    if (root / "web" / "dist" / "index.html").is_file():
        return Check("İnceleme Odası arayüzü", True, False, "web/dist derlenmiş")
    return Check("İnceleme Odası arayüzü", False, False, "derlenmemiş: cd web && npm ci && npm run build")


def _disk(path: Path) -> Check:
    probe = path if path.exists() else path.parent
    free = shutil.disk_usage(probe).free / 1e9
    if free < MIN_FREE_GB:  # rounded down, so a value under the limit never reads as the limit itself
        return Check("disk", False, False, f"{probe}: {math.floor(free * 10) / 10:.1f} GB boş, bir uzun analiz için az")
    return Check("disk", True, False, f"{probe}: {free:.0f} GB boş")


def run_checks(root: Path = ROOT, runs: Path | None = None) -> list[Check]:
    py = sys.version_info
    checks = [Check("Python", py >= (3, 11), True, f"{py.major}.{py.minor}.{py.micro}"),
              _ffmpeg(), _opencv(), _torch(), *_models(root), _ocr(), _web(root), _disk(runs or root / "runs")]
    return checks


def report(checks: list[Check]) -> tuple[str, int]:
    lines = ["SEKME ortam denetimi"]
    for c in checks:
        mark = "✓" if c.ok else ("✗" if c.essential else "!")
        lines.append(f"  {mark} {c.name}: {c.detail}")
    failed = [c for c in checks if not c.ok and c.essential]
    warned = [c for c in checks if not c.ok and not c.essential]
    if failed:
        lines.append(f"Sonuç: analiz çalışmaz ({len(failed)} eksik{f', {len(warned)} uyarı' if warned else ''})")
    else:
        lines.append(f"Sonuç: analize hazır{f' ({len(warned)} uyarı)' if warned else ''}")
    return "\n".join(lines), 1 if failed else 0
