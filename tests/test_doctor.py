"""`sekme doctor`: essentials fail the check, the rest only warn."""

from __future__ import annotations

import shutil
from pathlib import Path

import courtvision.doctor as doctor


def test_a_missing_essential_fails_and_optional_ones_only_warn(monkeypatch, tmp_path: Path) -> None:
    real_which = shutil.which
    monkeypatch.setattr(doctor.shutil, "which", lambda b: None if b == "ffprobe" else real_which(b))
    monkeypatch.setattr("courtvision.broadcast.ocr.available", lambda: False)
    checks = doctor.run_checks(root=tmp_path, runs=tmp_path / "runs")  # an empty folder: no models, no web build
    by = {c.name: c for c in checks}
    assert not by["ffmpeg"].ok and by["ffmpeg"].essential and "ffprobe bulunamadı" in by["ffmpeg"].detail
    assert all(not by[f"ağ {k}"].ok and by[f"ağ {k}"].essential for k in ("court", "ball", "players", "pose"))
    assert not by["Apple Vision OCR"].ok and not by["Apple Vision OCR"].essential
    assert not by["İnceleme Odası arayüzü"].ok and "npm run build" in by["İnceleme Odası arayüzü"].detail
    text, code = doctor.report(checks)
    assert code == 1 and text.splitlines()[-1].startswith("Sonuç: analiz çalışmaz (5 eksik")
    assert "  ✗ ffmpeg:" in text and "  ! Apple Vision OCR:" in text


def test_ready_when_only_warnings_remain() -> None:
    ok = doctor.Check("Python", True, True, "3.13.7")
    warn = doctor.Check("disk", False, False, "1 GB boş")
    assert doctor.report([ok]) == ("SEKME ortam denetimi\n  ✓ Python: 3.13.7\nSonuç: analize hazır", 0)
    text, code = doctor.report([ok, warn])
    assert code == 0 and text.endswith("Sonuç: analize hazır (1 uyarı)") and "  ! disk: 1 GB boş" in text


def test_a_checkpoint_that_cannot_be_read_is_reported(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "sekme_ball.pt").write_bytes(b"not a checkpoint")
    ball = next(c for c in doctor._models(tmp_path) if c.name == "ağ ball")
    assert not ball.ok and ball.essential and "okunamadı" in ball.detail


def test_the_command_line_runs_it(monkeypatch, capsys) -> None:
    import courtvision.cli as cli

    monkeypatch.setattr(doctor, "run_checks", lambda runs=None: [doctor.Check("ffmpeg", False, True, "yok")])
    assert cli.main(["doctor"]) == 1
    assert "✗ ffmpeg: yok" in capsys.readouterr().out
