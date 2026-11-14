"""Video probing and the mini court's uncertainty ellipse."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest


def test_probe_falls_back_when_the_average_frame_rate_is_missing(monkeypatch) -> None:
    import courtvision.io.video as vid

    meta = {"streams": [{"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "0/0",
                         "r_frame_rate": "30000/1001", "nb_frames": "300"}], "format": {"duration": "10.01"}}
    monkeypatch.setattr(vid, "_bin", lambda name: name)  # no ffprobe needed (CI has none)
    monkeypatch.setattr(vid.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(meta), ""))
    assert abs(vid.probe("x.mp4").fps - 29.97) < 0.01
    meta["streams"][0]["r_frame_rate"] = "0/0"
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video akışında kare hızı yok"):
        vid.probe("x.mp4")


def test_probe_reports_the_size_frames_are_decoded_at_for_rotated_video(monkeypatch) -> None:
    # A phone video is often stored sideways with a rotation tag; OpenCV and ffmpeg decode it upright, so the
    # probed size must be the upright one (it scales every ball position).
    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)

    def size(stream_extra: dict) -> tuple[int, int]:
        meta = {"streams": [{"codec_type": "video", "width": 720, "height": 1280, "avg_frame_rate": "30/1",
                             **stream_extra}], "format": {"duration": "4.0"}}
        monkeypatch.setattr(vid.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(meta), ""))
        info = vid.probe("phone.mp4")
        return info.width, info.height

    assert size({}) == (720, 1280)
    assert size({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90}]}) == (1280, 720)
    assert size({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}) == (1280, 720)
    assert size({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -180}]}) == (720, 1280)
    assert size({"tags": {"rotate": "270"}}) == (1280, 720)  # older containers keep it as a tag
    assert size({"tags": {"rotate": "N/A"}}) == (720, 1280)  # unreadable: taken as no rotation
    # Only the display matrix counts, and it comes before the old tag.
    assert size({"side_data_list": [{"side_data_type": "Other", "rotation": 90}]}) == (720, 1280)
    assert size({"side_data_list": [{"side_data_type": "Content light level metadata"},
                                    {"side_data_type": "Display Matrix", "rotation": -90}],
                 "tags": {"rotate": "0"}}) == (1280, 720)


def test_a_rotated_clip_decodes_at_the_probed_size(tmp_path) -> None:
    import shutil

    from courtvision.io.video import iter_frames, probe

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("needs ffmpeg and ffprobe (not on CI)")
    stored, tagged = tmp_path / "stored.mp4", tmp_path / "tagged.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x32:r=10", "-t", "0.3",
                    "-pix_fmt", "yuv420p", str(stored)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(stored), "-c", "copy",
                    str(tagged)], check=True)
    info = probe(tagged)
    _, frame = next(iter_frames(tagged))
    assert (info.width, info.height) == (32, 64)
    assert frame.shape[:2] == (info.height, info.width)


def test_probe_explains_a_file_that_is_not_a_readable_video(monkeypatch, tmp_path) -> None:
    import courtvision.io.video as vid

    path = tmp_path / "clip.mp4"
    monkeypatch.setattr(vid, "_bin", lambda name: name)  # no ffprobe needed (CI has none)
    monkeypatch.setattr(vid.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, "", f"{path}: Invalid data found when processing input\n"))
    with pytest.raises(vid.UnreadableVideoError) as e:
        vid.probe(path)
    msg = str(e.value)
    assert msg == "clip.mp4: video okunamadı (Invalid data found when processing input)"
    assert str(tmp_path) not in msg  # the server's paths never reach the client
    audio_only = {"streams": [{"codec_type": "audio"}], "format": {"duration": "2.0"}}
    monkeypatch.setattr(vid.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(audio_only), ""))
    with pytest.raises(vid.UnreadableVideoError, match="clip.mp4: dosyada video akışı yok"):
        vid.probe(path)


def test_probe_rejects_cover_art_and_broken_metadata_and_hides_folders(monkeypatch, tmp_path) -> None:
    import courtvision.io.video as vid

    path = tmp_path / "song.mp4"
    monkeypatch.setattr(vid, "_bin", lambda name: name)

    def answer(meta: dict | None, rc: int = 0, err: str = ""):
        monkeypatch.setattr(vid.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a, rc, json.dumps(meta) if meta is not None else "", err))

    cover = {"codec_type": "video", "codec_name": "mjpeg", "width": 320, "height": 240, "avg_frame_rate": "0/0",
             "r_frame_rate": "90000/1", "disposition": {"attached_pic": 1}}
    answer({"streams": [{"codec_type": "audio"}, cover], "format": {"duration": "3.0"}})
    with pytest.raises(vid.UnreadableVideoError, match="song.mp4: dosyada video akışı yok"):
        vid.probe(path)
    answer({"streams": [{"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "30/1"}],
            "format": {"duration": "N/A"}})
    with pytest.raises(vid.UnreadableVideoError, match="song.mp4: video bilgileri okunamadı"):
        vid.probe(path)
    answer(None, 1, f"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7f8a1c004200] moov atom not found\n"
                    f"[in#0 @ 0x600003a1c000] Error opening input: {path} is truncated\n")
    with pytest.raises(vid.UnreadableVideoError) as e:
        vid.probe(path)
    assert str(e.value) == "song.mp4: video okunamadı (Error opening input: song.mp4 is truncated)"


def test_cli_reports_an_unreadable_video_without_a_traceback(monkeypatch, tmp_path, capsys) -> None:
    import courtvision.io.video as vid
    from courtvision.cli import main

    monkeypatch.setattr(vid, "_bin", lambda name: name)
    monkeypatch.setattr(vid.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", f"{cmd[-1]}: moov atom not found\n"))
    out = tmp_path / "run"
    assert main(["analyze", str(tmp_path / "x.mp4"), "--out", str(out), "--no-render"]) == 2
    assert "video okunamadı (moov atom not found)" in capsys.readouterr().err
    assert not out.exists()  # no empty run folder is left behind


def test_the_mini_court_ellipse_follows_its_two_scales() -> None:
    from courtvision.geometry.court import CourtModel
    from courtvision.render.hud import MiniCourt, mini_ellipse_px

    m = MiniCourt(1920, 1080, CourtModel())
    assert abs(m.sx - m.sy) > 1  # the two axes really are scaled differently
    k = 2.4477
    for cov, (wx, wy) in ((np.diag([0.01, 1.0]), (0.1, 1.0)), (np.diag([1.0, 0.01]), (1.0, 0.1))):
        poly = mini_ellipse_px(m, np.array([0.0, 5.0]), cov)
        width, height = np.ptp(poly[:, 0]), np.ptp(poly[:, 1])
        assert abs(width - 2 * k * wx * m.sx) <= 2 and abs(height - 2 * k * wy * m.sy) <= 2
