"""Video probing and the mini court's uncertainty ellipse."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    meta["streams"][0]["avg_frame_rate"] = "30000/1"  # a downloaded WebM's average: 30 000 fps
    assert abs(vid.probe("x.mp4").fps - 29.97) < 0.01
    meta["streams"][0]["avg_frame_rate"] = "249/10"  # variable rate: the average, below the stated rate, is kept
    assert abs(vid.probe("x.mp4").fps - 24.9) < 0.01
    meta["streams"][0]["r_frame_rate"] = "0/0"
    meta["streams"][0]["avg_frame_rate"] = "0/0"
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video akışında kare hızı yok"):
        vid.probe("x.mp4")
    meta["streams"][0]["avg_frame_rate"] = "30000/1"  # and no stated rate to fall back on: no rate
    with pytest.raises(vid.UnreadableVideoError, match="kare hızı yok"):
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


def test_irregular_frame_intervals_counts_dropped_or_stretched_frames(monkeypatch) -> None:
    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)

    def count(times: list) -> int:
        out = "\n".join(str(t) for t in times) + "\n"
        monkeypatch.setattr(vid.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
        return vid.irregular_frame_intervals("x.mp4")

    even = [i / 30 for i in range(90)]
    assert count(even) == 0
    assert count(even[:40] + even[41:]) == 1  # one dropped frame: one interval twice as long
    assert count(even[:40] + ["N/A"] + even[41:]) == 0  # a packet without a timestamp: not judged, no warning
    assert count(even[:40] + [even[40], even[40]] + even[41:]) == 0  # a duplicated timestamp is not a gap
    assert count(even[::-1]) == 0  # packets in decode order (B-frames) are sorted first
    assert count([0.0]) == 0

    def missing(name: str) -> str:
        raise RuntimeError("ffprobe not found on PATH")

    monkeypatch.setattr(vid, "_bin", missing)  # CI has no ffprobe: no warning rather than a failure
    assert vid.irregular_frame_intervals("x.mp4") == 0


def test_a_missing_model_file_is_named_before_anything_is_written(monkeypatch, tmp_path, capsys) -> None:
    import courtvision.pipeline as pipeline
    from courtvision.cli import main
    from courtvision.errors import MissingModelError
    from courtvision.io.video import VideoInfo

    monkeypatch.setattr(pipeline, "probe", lambda v: VideoInfo(str(v), 1280, 720, 30.0, 90, 3.0, False))
    cfg = pipeline.PipelineConfig()
    cfg.models["ball"] = str(tmp_path / "gone" / "sekme_ball.pt")
    out = tmp_path / "run"
    with pytest.raises(MissingModelError, match="ball: sekme_ball.pt"):
        pipeline.analyze(tmp_path / "x.mp4", out, cfg)
    assert not out.exists()
    # The command line says it in one line (no traceback) and exits with 2.
    assert main(["analyze", str(tmp_path / "x.mp4"), "--out", str(out), "--model",
                 f"ball={tmp_path / 'gone' / 'sekme_ball.pt'}"]) == 2
    assert "model dosyası bulunamadı" in capsys.readouterr().err


def test_the_command_line_fits_in_parallel_by_default_but_respects_a_setting(monkeypatch, tmp_path) -> None:
    import os

    import courtvision.cli as cli
    import courtvision.pipeline as pipeline

    seen = []

    def fake_analyze(video, out, cfg, progress):
        seen.append(os.environ.get("SEKME_FIT_WORKERS"))
        raise pipeline.MissingModelError("stop here")  # the CLI prints it and exits with 2

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(cli, "default_fit_workers", lambda: 3)
    args = ["analyze", str(tmp_path / "x.mp4"), "--out", str(tmp_path / "run")]
    monkeypatch.delenv("SEKME_FIT_WORKERS", raising=False)
    assert cli.main(args) == 2 and seen[-1] == "3"  # the command line is never patched: in parallel
    monkeypatch.delenv("SEKME_FIT_WORKERS", raising=False)
    assert cli.main([*args, "--fit-workers", "1"]) == 2 and seen[-1] == "1"
    monkeypatch.setenv("SEKME_FIT_WORKERS", "2")
    assert cli.main(args) == 2 and seen[-1] == "2"  # an explicit environment setting wins over the default


def test_video_notes_warn_about_irregular_timing_and_low_frame_rates() -> None:
    from courtvision.pipeline import LOW_FPS_NOTE, VFR_NOTE, video_notes

    assert video_notes(29.97, 0) == [] and video_notes(59.94, 0) == [] and video_notes(25.0, 0) == []
    assert video_notes(30.0, 3) == [VFR_NOTE.format(n=3)]
    assert video_notes(15.0, 0) == [LOW_FPS_NOTE.format(fps=15.0)]
    assert len(video_notes(12.0, 2)) == 2


def test_nominal_rate_prefers_the_stated_rate_when_it_agrees_with_the_timestamps(monkeypatch) -> None:
    from fractions import Fraction

    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)

    def rate(times: list, stated: str) -> Fraction | None:
        packets = "\n".join(str(t) for t in times) + "\n"

        def run(cmd, **k):
            out = stated + "\n" if "stream=r_frame_rate" in cmd else packets
            return subprocess.CompletedProcess(cmd, 0, out, "")

        monkeypatch.setattr(vid.subprocess, "run", run)
        return vid.nominal_rate("x.mp4")

    ntsc = [i * 1001 / 30000 for i in range(120)]
    gappy = [t for i, t in enumerate(ntsc) if not (40 <= i < 100 and i % 3 == 1)]  # a stretch at 20 fps
    assert rate(gappy, "30000/1001") == Fraction(30000, 1001)  # the median interval is still the 29.97 fps one
    assert rate(gappy, "90000/1") == Fraction(30000, 1001)  # a timebase, not a frame rate: the median's rate
    assert rate(gappy, "N/A") == Fraction(30000, 1001)
    assert rate([0.0], "30/1") is None


def test_constant_rate_copy_runs_ffmpegs_fps_filter_and_leaves_no_partial_file(monkeypatch, tmp_path) -> None:
    from fractions import Fraction

    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)
    seen = []

    def run(cmd, **k):
        seen.append(cmd)
        if fail:
            return subprocess.CompletedProcess(cmd, 1, "", f"{tmp_path / 'in.mov'}: Invalid data found")
        Path(cmd[-1]).write_bytes(b"copy")
        return subprocess.CompletedProcess(cmd, 0, "", log)

    log = ""
    written = []
    monkeypatch.setattr(vid, "probe", lambda p: written.append(Path(p).name) or vid.VideoInfo(
        str(p), 160, 90, 30.0, 3, 0.1, False))
    monkeypatch.setattr(vid.subprocess, "run", run)
    fail = False
    dest, repeated = vid.constant_rate_copy(tmp_path / "in.mov", tmp_path / "constant_rate.mp4", Fraction(30000, 1001))
    assert dest.read_bytes() == b"copy" and "fps=30000/1001," in seen[-1][seen[-1].index("-vf") + 1]
    assert seen[-1][seen[-1].index("-vf") + 1].endswith(",showinfo") and repeated is None  # no frame log: unknown
    frame = "[Parsed_showinfo_2 @ 0x6000] n:{} pts:{} pts_time:0 checksum:{} plane_checksum:[1 2 3]"
    log = "\n".join(frame.format(i, i, c) for i, c in enumerate(("AB", "AB", "CD")))
    assert vid.constant_rate_copy(tmp_path / "in.mov", dest, Fraction(30))[1] == [1]
    assert written[-1] == "constant_rate.part.mp4"  # the frames written are counted before the copy is kept
    log = "\n".join(frame.format(i, i, c) for i, c in enumerate(("AB", "AB")))  # a line per frame missing
    assert vid.constant_rate_copy(tmp_path / "in.mov", dest, Fraction(30))[1] is None
    assert seen[-1][seen[-1].index("0:V:0") - 1] == "-map"  # never the cover art
    with pytest.raises(ValueError, match="overwrite its source"):
        vid.constant_rate_copy(dest, dest, Fraction(30))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["constant_rate.mp4"]
    fail = True
    with pytest.raises(vid.UnreadableVideoError, match="in.mov: sabit kare hızlı kopya yapılamadı"):
        vid.constant_rate_copy(tmp_path / "in.mov", tmp_path / "other.mp4", Fraction(30))
    assert not (tmp_path / "other.mp4").exists() and not (tmp_path / "other.part.mp4").exists()


def test_repeated_frames_are_read_from_ffmpegs_showinfo_checksums() -> None:
    from courtvision.io.video import repeated_frames

    def line(n, checksum):
        return (f"[Parsed_showinfo_2 @ 0x600001a4c000] n:{n:4d} pts:{n * 1001:7d} pts_time:{n / 29.97:<8.6f} "
                f"duration:1001 fmt:yuv420p cl:left sar:1/1 s:1280x720 i:P iskey:0 type:P "
                f"checksum:{checksum} plane_checksum:[1A2B3C4D 5E6F7A8B 9C0D1E2F] mean:[87 127 128]")

    log = "\n".join(["Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'phone.mov':",
                      line(0, "AAAA0001"), line(1, "AAAA0002"), line(2, "AAAA0002"),  # 2 repeats 1
                      "[libx264 @ 0x1234] frame I:1 Avg QP:20.00 size: 1000",
                      line(3, "AAAA0003"), line(4, "aaaa0003"), line(5, "AAAA0005"), line(6, "AAAA0005")])
    assert repeated_frames(log) == (7, [2, 6])  # checksums compared as ffmpeg prints them, frame by frame
    assert repeated_frames("no frame log") == (0, [])


def test_link_source_replaces_links_to_another_video_but_never_deletes_a_file(tmp_path) -> None:
    from courtvision.io.video import link_source

    run = tmp_path / "run"
    run.mkdir()
    original, copy = tmp_path / "phone.mov", run / "constant_rate.mp4"
    original.write_bytes(b"mov")
    copy.write_bytes(b"mp4")
    link_source(original, run)
    assert (run / "source.mov").resolve() == original.resolve()
    link_source(copy, run)  # re-analysed through its constant-rate copy: only the copy is offered
    assert sorted(p.name for p in run.glob("source.*")) == ["source.mp4"]
    assert (run / "source.mp4").resolve() == copy.resolve() and original.read_bytes() == b"mov"
    (run / "source.mp4").unlink()
    (run / "source.webm").write_bytes(b"a copied source")  # made where links are unsupported: kept
    link_source(copy, run)
    assert (run / "source.webm").read_bytes() == b"a copied source"


def test_where_links_are_unsupported_link_source_replaces_its_own_copies_only(monkeypatch, tmp_path) -> None:
    import courtvision.io.video as vid

    def no_links(self, target):
        raise OSError("links unsupported")

    monkeypatch.setattr(Path, "symlink_to", no_links)
    run = tmp_path / "run"
    run.mkdir()
    a, b = tmp_path / "a.mov", tmp_path / "b.mp4"
    a.write_bytes(b"video a")
    b.write_bytes(b"video b")
    vid.link_source(a, run)
    assert (run / "source.mov").read_bytes() == b"video a"
    vid.link_source(a, run)  # the same video again: the copy is kept
    assert json.loads((run / vid.SOURCE_COPY_RECORD).read_text())["file"] == "source.mov"
    vid.link_source(b, run)  # another video: the old copy (first in name order) must not be offered instead
    assert sorted(p.name for p in run.glob("source.*")) == ["source.mp4"]
    b.write_bytes(b"video b, analysed again after it changed")  # same name, new content: copied again
    vid.link_source(b, run)
    assert (run / "source.mp4").read_bytes().startswith(b"video b, analysed again")
    (run / "source.webm").write_bytes(b"a file put here by hand")
    vid.link_source(a, run)
    assert (run / "source.webm").read_bytes() == b"a file put here by hand"  # not made by link_source: kept
    assert sorted(p.name for p in run.glob("source.*")) == ["source.mov", "source.webm"]


def test_a_variable_rate_video_is_analysed_from_one_reused_constant_rate_copy(monkeypatch, tmp_path) -> None:
    from fractions import Fraction

    import courtvision.pipeline as pipeline
    from courtvision.io.video import VideoInfo

    made = []

    def fake_copy(src, dest, rate):
        made.append(rate)
        dest.write_bytes(b"copy at " + str(rate).encode())
        return dest, [40, 43]

    monkeypatch.setattr(pipeline, "constant_rate_copy", fake_copy)
    monkeypatch.setattr(pipeline, "nominal_rate", lambda p: Fraction(30000, 1001))
    monkeypatch.setattr(pipeline, "probe", lambda v: VideoInfo(str(v), 1280, 720, 29.97, 119, 3.97, False))
    src = tmp_path / "phone.mov"
    src.write_bytes(b"source")
    info = VideoInfo(str(src), 1280, 720, 24.9, 99, 3.97, False)
    path, rec, repeated = pipeline._constant_rate_source(src, "abc", info, tmp_path)
    assert path == tmp_path / "constant_rate.mp4" and made == [Fraction(30000, 1001)]
    assert rec["frames"] == 119 and rec["source_frames"] == 99 and rec["added_frames"] == 20
    assert repeated == [40, 43] and rec["repeated_frames"] == 2
    assert rec["sha256"] == pipeline._sha256(path)
    assert pipeline._constant_rate_source(src, "abc", info, tmp_path)[2] == [40, 43]  # re-analysis: reused
    assert len(made) == 1
    record = tmp_path / "constant_rate.json"
    old = json.loads(record.read_text())
    record.write_text(json.dumps({k: v for k, v in old.items() if k != "repeated"}))  # a record from before
    pipeline._constant_rate_source(src, "abc", info, tmp_path)
    assert len(made) == 2  # remade, so the repeats are known
    record.write_text(json.dumps({**old, "repeated": [0, 500]}))  # not this copy's frames: none trusted
    assert pipeline._constant_rate_source(src, "abc", info, tmp_path)[1]["repeated_frames"] is None
    record.write_text(json.dumps(old))
    pipeline._constant_rate_source(src, "other source", info, tmp_path)  # another video in the same folder
    assert len(made) == 3
    path.write_bytes(b"changed by hand")  # a copy that no longer matches its record is remade
    pipeline._constant_rate_source(src, "other source", info, tmp_path)
    assert len(made) == 4 and path.read_bytes().startswith(b"copy at")

    def crash(src, dest, rate):
        dest.write_bytes(b"half written")
        raise OSError("killed")

    monkeypatch.setattr(pipeline, "constant_rate_copy", crash)
    with pytest.raises(OSError):
        pipeline._constant_rate_source(src, "abc", info, tmp_path)  # source A again, crashing mid-copy
    assert not (tmp_path / "constant_rate.json").exists()  # so the next run cannot take this copy for A's
    note = pipeline.video_notes(29.97, 20, rec)[0]
    assert "29,97 fps" in note and "net 20 kare" in note and "yok sayıldı" not in note
    note = pipeline.video_notes(29.97, 20, {**rec, "repeated_ball_dropped": True})[0]
    assert note.endswith("Tekrarlanan 2 karede top yok sayıldı; top izi orada kaçırılmış bir kare gibi dolduruldu.")


@pytest.mark.skipif(not all(__import__("shutil").which(b) for b in ("ffmpeg", "ffprobe")),
                    reason="needs ffmpeg and ffprobe (CI has none)")
def test_the_constant_rate_copy_restores_even_timestamps_on_a_real_file(tmp_path) -> None:
    import courtvision.io.video as vid

    src = tmp_path / "gappy.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=2",
                    "-vf", "select='not(between(n\\,20\\,49)*eq(mod(n\\,3)\\,1))'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    assert vid.irregular_frame_intervals(src) == 10 and vid.probe(src).n_frames == 50
    rate = vid.nominal_rate(src)
    assert rate == 30
    copy, repeated = vid.constant_rate_copy(src, tmp_path / "constant_rate.mp4", rate)
    info = vid.probe(copy)
    assert vid.irregular_frame_intervals(copy) == 0 and info.n_frames == 60 and abs(info.fps - 30) < 1e-6
    assert repeated == list(range(22, 50, 3))  # each removed frame's slot holds the frame before it

    odd = tmp_path / "odd.mp4"  # an odd size (x264 4:2:0 needs even ones) and no audio
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", "pad=161:91", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-pix_fmt", "yuv444p", str(odd)], check=True)
    info = vid.probe(vid.constant_rate_copy(odd, tmp_path / "odd_copy.mp4", rate)[0])
    assert (info.width, info.height, info.n_frames, info.has_audio) == (160, 90, 60, False)


def test_the_command_line_says_how_a_variable_rate_video_was_handled(monkeypatch, tmp_path, capsys) -> None:
    import courtvision.cli as cli
    import courtvision.pipeline as pipeline

    def report(copy):
        return {"video": {"fps": 29.97, "irregular_frame_intervals": 20, "constant_rate_copy": copy},
                "calibration": {"valid_frames": 1, "total_frames": 1, "line_rmse_px_median": 0.3}, "flights": []}

    args = ["analyze", str(tmp_path / "x.mp4"), "--out", str(tmp_path / "run"), "--fit-workers", "1"]
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: report({"file": "constant_rate.mp4", "fps": 29.97003,
                                                                      "added_frames": 20}))
    assert cli.main(args) == 0
    assert "sabit 29,97 fps'lik kopyadan analiz edildi (constant_rate.mp4, net 20 kare eklendi)\n" in \
        capsys.readouterr().err
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: report({
        "file": "constant_rate.mp4", "fps": 29.97003, "added_frames": 20, "repeated_frames": 20,
        "repeated_ball_dropped": True}))
    assert cli.main(args) == 0
    assert "(constant_rate.mp4, net 20 kare eklendi); tekrarlanan 20 karede top yok sayıldı" in capsys.readouterr().err
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: report(None))  # the copy failed: the old warning
    assert cli.main(args) == 0
    assert "uyarı: değişken kare hızı (20 düzensiz kare aralığı); hızlar ve işaretler kayabilir" in \
        capsys.readouterr().err


def test_an_interlaced_video_gets_a_limitation(monkeypatch, tmp_path, capsys) -> None:
    import courtvision.cli as cli
    import courtvision.pipeline as pipeline

    notes = pipeline.video_notes(29.97, 0, None, "tt")
    assert notes == [pipeline.INTERLACED_NOTE.format(order="tt", rate="59,94")]
    assert pipeline.video_notes(29.97, 0, None, "progressive") == []
    assert pipeline.video_notes(29.97, 0, None, "unknown") == []  # (MPEG-TS often says so before decoding)
    both = pipeline.video_notes(29.97, 12, None, "bb")
    assert both[0].startswith("Video geçmeli (interlaced, alan sırası bb)")
    assert both[1] == pipeline.VFR_NOTE.format(n=12)
    rep = {"video": {"fps": 29.97, "irregular_frame_intervals": 0, "constant_rate_copy": None, "field_order": "tt"},
           "calibration": {"valid_frames": 1, "total_frames": 1, "line_rmse_px_median": 0.3}, "flights": []}
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: rep)
    assert cli.main(["analyze", str(tmp_path / "x.ts"), "--out", str(tmp_path / "run"), "--fit-workers", "1"]) == 0
    err = capsys.readouterr().err
    assert "uyarı: video geçmeli (interlaced, alan sırası tt); zamanlar yarım kare kayabilir" in err


@pytest.mark.skipif(not all(__import__("shutil").which(b) for b in ("ffmpeg", "ffprobe")),
                    reason="needs ffmpeg and ffprobe (CI has none)")
def test_interlaced_files_are_recognised(tmp_path) -> None:
    import courtvision.io.video as vid

    prog = tmp_path / "p60.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=60:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(prog)], check=True)
    inter = tmp_path / "i30.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(prog), "-vf", "tinterlace=mode=interleave_top,setfield=tff",
                    "-c:v", "libx264", "-flags", "+ilme+ildct", "-x264-params", "tff=1", "-pix_fmt", "yuv420p",
                    str(inter)], check=True)
    assert vid.field_order(prog) == "progressive" and vid.field_order(inter) in vid.INTERLACED_ORDERS
    assert vid.probe(inter).n_frames == 30
    assert vid.field_order(tmp_path / "missing.mp4") is None


def test_a_video_whose_court_cannot_be_calibrated_says_why(monkeypatch, tmp_path, capsys) -> None:
    import courtvision.cli as cli
    import courtvision.pipeline as pipeline

    assert pipeline.video_notes(29.97, 0, calibrated=(0, 600)) == [pipeline.NO_CALIBRATION_NOTE]
    assert pipeline.video_notes(29.97, 0, calibrated=(60, 600)) == [pipeline.LOW_CALIBRATION_NOTE.format(pct=10)]
    assert pipeline.video_notes(29.97, 0, calibrated=(420, 600)) == []  # a usual highlight share
    assert pipeline.video_notes(29.97, 0, calibrated=(0, 0)) == []
    rep = {"video": {"fps": 29.97, "irregular_frame_intervals": 0, "constant_rate_copy": None},
           "calibration": {"valid_frames": 0, "total_frames": 600, "line_rmse_px_median": None}, "flights": []}
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: rep)
    assert cli.main(["analyze", str(tmp_path / "x.webm"), "--out", str(tmp_path / "run"), "--fit-workers", "1"]) == 0
    assert "uyarı: kort hiçbir karede kalibre edilemedi; ölçüm ve karar yok" in capsys.readouterr().err
