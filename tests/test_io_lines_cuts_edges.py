"""Video probing (io/video.py: the frame rate chosen at its edges, rotation, the ffprobe reason text, packet
intervals, audio, opening a capture), sub-pixel court line refinement (calib/line_refine.py: ridge positions,
residual signs, failure cases) and the shot-cut helpers (perception/court.py: histogram, thumbnail and cut
thresholds) at their edges. Values checked by running the code (gaps found by a test review)."""

from __future__ import annotations

import json
import shutil
import subprocess
import types
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="needs ffmpeg and ffprobe (CI has none)")


def _ffmpeg(*args: str) -> None:
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:  # this build lacks an encoder / filter we asked for
        pytest.skip(f"ffmpeg cannot make the test file: {exc.stderr.decode(errors='replace')[:100]}")


def _tiny_video(path: Path, rate: str = "30", seconds: str = "1", extra: tuple[str, ...] = ()) -> Path:
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=size=160x120:rate={rate}", "-t", seconds, "-c:v", "libx264",
            "-pix_fmt", "yuv420p", *extra, str(path))
    return path


# --------------------------------------------------------------------------- io/video.py: pure helpers

def test_the_reason_text_is_the_last_message_without_folders_or_decoder_addresses() -> None:
    from courtvision.io.video import _reason

    p, n = "/srv/up/clip.mp4", "clip.mp4"
    assert _reason("", p, n) == "ffprobe failed"
    assert _reason("\n   \n\t\n", p, n) == "ffprobe failed"  # only blank lines
    assert _reason(f"first\n{p}: second thing\n\n   \n", p, n) == "second thing"  # last non-blank, prefix cut
    assert _reason(f"[mov,mp4 @ 0xAB12cd] {p}: moov atom not found", p, n) == f"{n}: moov atom not found"
    assert _reason(f"{p}: [mov @ 0xAB12] bad", p, n) == "bad"  # prefix first, then the decoder tag
    assert _reason(f"cannot read {p} nor {p}", p, n) == f"cannot read {n} nor {n}"  # every occurrence, not the first
    assert _reason("[a @ 0x1] [b @ 0x2] two tags", p, n) == "two tags"
    assert _reason("[mov @ 0xZZ] not an address", p, n) == "[mov @ 0xZZ] not an address"
    assert _reason("  padded  ", p, n) == "padded"
    assert _reason("x" * 119, p, n) == "x" * 119
    assert _reason("x" * 120, p, n) == "x" * 120  # 120 characters still fit
    assert _reason("x" * 121, p, n) == "x" * 119 + "…" and len(_reason("x" * 500, p, n)) == 120


def test_rotation_reads_the_display_matrix_before_the_old_tag_and_rounds_to_whole_degrees() -> None:
    from courtvision.io.video import _rotation

    matrix = "Display Matrix"
    assert _rotation({}) == 0
    assert _rotation({"side_data_list": None, "tags": None}) == 0
    assert _rotation({"tags": {"rotate": "89.6"}}) == 90 and _rotation({"tags": {"rotate": "90.4"}}) == 90
    assert _rotation({"tags": {"rotate": "-89.6"}}) == -90 and _rotation({"tags": {"rotate": 90}}) == 90
    assert isinstance(_rotation({"tags": {"rotate": "90.4"}}), int)
    for bad in ("inf", "nan", None, "", "N/A"):  # unreadable: no rotation, never an exception
        assert _rotation({"tags": {"rotate": bad}}) == 0
    tag90 = {"rotate": "90"}
    assert _rotation({"side_data_list": [{"side_data_type": matrix, "rotation": 0}], "tags": tag90}) == 0
    # a matrix that is there but unreadable does not fall back to the tag; one without a rotation field does
    assert _rotation({"side_data_list": [{"side_data_type": matrix, "rotation": "N/A"}], "tags": tag90}) == 0
    assert _rotation({"side_data_list": [{"side_data_type": matrix}], "tags": tag90}) == 90
    assert _rotation({"side_data_list": [{"rotation": 90}], "tags": {}}) == 0  # not a display matrix
    assert _rotation({"side_data_list": [{"side_data_type": matrix, "rotation": 90.0},
                                         {"side_data_type": matrix, "rotation": 180}]}) == 90  # the first one
    assert _rotation({"side_data_list": [{"side_data_type": matrix, "rotation": -270}]}) == -270
    assert _rotation({"side_data_list": [{"side_data_type": matrix, "rotation": 450.0}]}) == 450


def _fake_probe(monkeypatch, meta: dict):
    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)
    monkeypatch.setattr(vid.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(meta), ""))
    return vid


def _stream(**kw) -> dict:
    return {"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
            "nb_frames": "90", **kw}


def test_probe_trusts_the_average_rate_up_to_one_and_a_half_times_the_stated_rate_and_a_thousand(monkeypatch) -> None:
    import courtvision.io.video as vid

    assert vid.MAX_FPS == 1000.0

    def fps(avg: str, stated: str) -> float:
        meta = {"streams": [_stream(avg_frame_rate=avg, r_frame_rate=stated)], "format": {"duration": "3.0"}}
        return _fake_probe(monkeypatch, meta).probe("x.mp4").fps

    assert fps("30/1", "20/1") == 30.0  # exactly 1.5 times the stated rate: still the average
    assert fps("31/1", "20/1") == 20.0  # above that: the stated rate
    assert fps("5/1", "30/1") == 5.0  # far below (variable rate): the average
    assert fps("1000/1", "0/0") == 1000.0  # no stated rate: the average up to the limit ...
    assert fps("1001/1", "30/1") == 30.0
    assert fps("0/1", "25/1") == 25.0 and fps("-5/1", "25/1") == 25.0  # no positive average
    assert fps("30/0", "25/1") == 25.0 and fps("", "25/1") == 25.0
    assert fps("0/0", "30000/1001") == pytest.approx(29.97003, abs=1e-5)
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video akışında kare hızı yok"):
        fps("1001/1", "0/0")  # ... and beyond it nothing to trust
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video bilgileri okunamadı"):
        fps("abc", "25/1")  # not a fraction at all


def test_probe_fills_in_frame_count_duration_and_audio_flag_from_what_the_file_gives(monkeypatch) -> None:
    stream = _stream()
    del stream["nb_frames"]
    vid = _fake_probe(monkeypatch, {"streams": [stream, {"codec_type": "audio"}], "format": {"duration": "10.01"}})
    info = vid.probe("a/b/x.mp4")
    assert (info.n_frames, info.duration, info.has_audio, info.path) == (300, 10.01, True, "a/b/x.mp4")
    assert (info.width, info.height) == (1280, 720) and info.fps == 30.0

    stream = _stream(duration="2.5", nb_frames="77")  # no format duration: the stream's
    vid = _fake_probe(monkeypatch, {"streams": [stream], "format": {}})
    info = vid.probe("x.mp4")
    assert (info.n_frames, info.duration, info.has_audio) == (77, 2.5, False)  # the stated count wins over duration*fps
    vid = _fake_probe(monkeypatch, {"streams": [_stream(nb_frames=0)], "format": {"duration": "2.0"}})
    assert vid.probe("x.mp4").n_frames == 60  # a numeric 0 is no count: duration * fps
    vid = _fake_probe(monkeypatch, {"streams": [_stream(nb_frames=None)], "format": {"duration": "2.02"}})
    assert vid.probe("x.mp4").n_frames == 61  # 60.6 rounded, not cut
    vid = _fake_probe(monkeypatch, {"streams": [_stream(nb_frames="0")], "format": {"duration": "2.0"}})
    assert vid.probe("x.mp4").n_frames == 60  # the text "0" is no count either (it used to give 0 frames)
    vid = _fake_probe(monkeypatch, {"streams": [_stream(nb_frames="N/A")], "format": {"duration": "1"}})
    assert vid.probe("x.mp4").n_frames == 30  # nor is "N/A" (it used to make the video unreadable)


def test_probe_refuses_a_stream_without_a_size_or_without_any_frame_count(monkeypatch) -> None:
    vid = _fake_probe(monkeypatch, {"streams": [_stream(width=0)], "format": {"duration": "1"}})
    with pytest.raises(vid.UnreadableVideoError) as e:
        vid.probe("dir/x.mp4")
    assert str(e.value) == "x.mp4: video akışının boyutu yok"
    vid = _fake_probe(monkeypatch, {"streams": [_stream(height=None)], "format": {"duration": "1"}})
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video akışının boyutu yok"):
        vid.probe("x.mp4")
    vid = _fake_probe(monkeypatch, {"streams": [_stream(nb_frames="N/A")], "format": {"duration": "N/A"}})
    with pytest.raises(vid.UnreadableVideoError, match="x.mp4: video bilgileri okunamadı"):
        vid.probe("x.mp4")  # neither a count nor a duration
    vid = _fake_probe(monkeypatch, {"streams": [_stream()]})  # no format section at all
    with pytest.raises(vid.UnreadableVideoError, match="video bilgileri okunamadı"):
        vid.probe("x.mp4")


def test_probe_swaps_the_size_for_a_quarter_turn_but_not_for_a_half_turn_or_a_tilt(monkeypatch) -> None:
    def size(rot) -> tuple[int, int]:
        sd = [{"side_data_type": "Display Matrix", "rotation": rot}]
        info = _fake_probe(monkeypatch, {"streams": [_stream(side_data_list=sd)],
                                         "format": {"duration": "1"}}).probe("x.mp4")
        return info.width, info.height

    assert size(90) == (720, 1280) and size(-270) == (720, 1280) and size(450) == (720, 1280)
    assert size(180) == (1280, 720) and size(0) == (1280, 720) and size(45) == (1280, 720)
    assert size(89.6) == (720, 1280)  # rounded to 90 first


# --------------------------------------------------------------------------- io/video.py: packet intervals

def _fake_packets(monkeypatch, out: str | Exception, calls: list | None = None):
    import courtvision.io.video as vid

    monkeypatch.setattr(vid, "_bin", lambda name: name)

    def run(cmd, **kw):
        if calls is not None:
            calls.append((cmd, kw))
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(vid.subprocess, "run", run)
    return vid


def test_frame_intervals_are_the_sorted_positive_gaps_between_packet_timestamps(monkeypatch) -> None:
    calls: list = []
    vid = _fake_packets(monkeypatch, "0.0\n0.5\n0.25\n0.75,\n1.0\n1.0\n", calls)
    d = vid._frame_intervals("v.mp4")
    assert d is not None and np.allclose(d, [0.25, 0.25, 0.25, 0.25]) and len(d) == 4  # sorted; duplicate dropped
    cmd, kw = calls[0]
    assert cmd[cmd.index("-select_streams") + 1] == "V:0" and "packet=pts_time" in cmd and cmd[-1] == "v.mp4"
    assert kw["check"] is True and kw["timeout"] > 0  # a failing ffprobe must raise inside, to be caught
    assert vid._frame_intervals(Path("v.mp4")) is not None  # a Path is accepted

    assert np.allclose(_fake_packets(monkeypatch, "0\n1\n2\n")._frame_intervals("v"), [1, 1])  # three stamps: enough
    assert _fake_packets(monkeypatch, "0\n1\n")._frame_intervals("v") is None  # one interval cannot be judged
    assert _fake_packets(monkeypatch, "0\n1\n1\n1.0\n")._frame_intervals("v") is None  # one positive interval only
    assert _fake_packets(monkeypatch, "")._frame_intervals("v") is None
    assert _fake_packets(monkeypatch, "0\nN/A\n1\n2\n")._frame_intervals("v") is None  # one unreadable stamp: none


def test_frame_intervals_are_none_when_ffprobe_fails_times_out_or_is_missing(monkeypatch) -> None:
    for exc in (subprocess.CalledProcessError(1, "ffprobe"), subprocess.TimeoutExpired("ffprobe", 120),
                FileNotFoundError("ffprobe"), RuntimeError("ffprobe not found on PATH")):
        assert _fake_packets(monkeypatch, exc)._frame_intervals("v.mp4") is None


def test_a_nan_packet_timestamp_leaves_the_intervals_unusable_instead_of_raising(monkeypatch) -> None:
    # Pinned as it is: float("nan") parses, np.sort puts nan last, np.diff gives nan and `d > 0` drops it, so a
    # file whose packets carry the text "nan" is judged on the other packets (ffprobe writes "N/A", not "nan")
    d = _fake_packets(monkeypatch, "0\n1\n2\nnan\n")._frame_intervals("v")
    assert d is not None and np.allclose(d, [1, 1])


def test_irregular_intervals_are_counted_strictly_beyond_25_percent_above_and_below_the_median(monkeypatch) -> None:
    def count(gap: float) -> int:  # nine intervals of 4, one of `gap`
        times, t = [0.0], 0.0
        for g in [4.0] * 5 + [gap] + [4.0] * 4:
            t += g
            times.append(t)
        return _fake_packets(monkeypatch, "\n".join(map(str, times)) + "\n").irregular_frame_intervals("v.mp4")

    assert count(4.0) == 0
    assert count(5.0) == 0 and count(5.001) == 1  # 1.25 times the median is still regular
    assert count(3.0) == 0 and count(2.999) == 1  # 0.75 times too
    assert count(8.0) == 1 and count(1.0) == 1 and count(100.0) == 1
    assert _fake_packets(monkeypatch, "0\n1\n")._frame_intervals("v") is None
    assert _fake_packets(monkeypatch, "0\n1\n").irregular_frame_intervals("v") == 0


def test_nominal_rate_keeps_the_stated_rate_within_five_percent_of_the_median_rate(monkeypatch) -> None:
    import courtvision.io.video as vid

    even = [i * 0.05 for i in range(40)]  # a 20 fps median

    def rate(stated: str, times=even) -> Fraction | None:
        packets = "\n".join(map(str, times)) + "\n"

        def run(cmd, **k):
            return subprocess.CompletedProcess(cmd, 0, stated + "\n" if "stream=r_frame_rate" in cmd else packets, "")

        monkeypatch.setattr(vid, "_bin", lambda name: name)
        monkeypatch.setattr(vid.subprocess, "run", run)
        return vid.nominal_rate("x.mp4")

    assert rate("2098/100") == Fraction(2098, 100)  # 4.9 % above: the stated rate
    assert rate("2102/100") == 20  # 5.1 % above: the median's rate
    assert rate("1902/100") == Fraction(1902, 100)  # 4.9 % below
    assert rate("1898/100") == 20  # 5.1 % below
    assert rate("20/1,") == 20 and rate("20/1,") == Fraction(20, 1)  # a trailing csv comma is stripped
    assert rate("0/1") == 20 and rate("-20/1") == 20 and rate("0/0") == 20  # no usable stated rate
    assert rate("30000/1001", [i * 1001 / 30000 for i in range(40)]) == Fraction(30000, 1001)
    ntsc_median = [i * 1001 / 30000 for i in range(40)]
    assert rate("N/A", ntsc_median) == Fraction(30000, 1001)  # limit_denominator(1001) recovers NTSC exactly
    assert rate("N/A", [i * 0.04 for i in range(40)]) == 25
    assert rate("N/A", [0.0, 1.0]) is None

    def broken(cmd, **k):
        if "stream=r_frame_rate" in cmd:
            raise subprocess.CalledProcessError(1, "ffprobe")
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(str, even)), "")

    monkeypatch.setattr(vid.subprocess, "run", broken)
    assert vid.nominal_rate("x.mp4") == 20  # the stated rate could not be read: the median's


# --------------------------------------------------------------------------- io/video.py: audio and capture

def test_extract_audio_asks_ffmpeg_for_mono_float32_at_the_requested_rate(monkeypatch) -> None:
    import courtvision.io.video as vid

    calls = []
    samples = np.array([0.0, 0.5, -1.0, 0.25], dtype="<f4")

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, samples.tobytes(), b"")

    monkeypatch.setattr(vid, "_bin", lambda name: name)
    monkeypatch.setattr(vid.subprocess, "run", run)
    out = vid.extract_audio(Path("some/clip.mp4"), 16000)
    assert out.dtype == np.float32 and list(out) == [0.0, 0.5, -1.0, 0.25]
    assert out.flags.writeable  # a copy, not a read-only view of the pipe's bytes
    cmd = calls[0][0]
    assert cmd[cmd.index("-i") + 1] == "some/clip.mp4" and "-vn" in cmd
    assert cmd[cmd.index("-ac") + 1] == "1" and cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-f") + 1] == "f32le" and cmd[-1] == "pipe:1"
    vid.extract_audio("clip.mp4")
    assert calls[1][0][calls[1][0].index("-ar") + 1] == "22050"  # the default rate

    monkeypatch.setattr(vid.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, samples.tobytes(), b"boom"))
    failed = vid.extract_audio("clip.mp4")  # a failed run gives no audio, whatever it printed
    assert failed.dtype == np.float32 and failed.shape == (0,)
    monkeypatch.setattr(vid.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""))
    silent = vid.extract_audio("clip.mp4")
    assert silent.dtype == np.float32 and silent.shape == (0,)  # a video without audio


def test_extract_audio_without_ffmpeg_says_so_rather_than_returning_silence(monkeypatch) -> None:
    import courtvision.io.video as vid

    monkeypatch.setattr(vid.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found on PATH"):
        vid.extract_audio("clip.mp4")


def test_open_video_asks_opencv_to_decode_upright_and_accepts_paths(monkeypatch) -> None:
    import courtvision.io.video as vid

    seen: dict = {}

    class FakeCapture:
        def __init__(self, target):
            seen["target"] = target
            seen["set"] = []

        def set(self, prop, value):
            seen["set"].append((prop, value))
            return True

    monkeypatch.setattr(vid.cv2, "VideoCapture", FakeCapture)
    cap = vid.open_video(Path("a") / "clip.mp4")
    assert isinstance(cap, FakeCapture) and seen["target"] == str(Path("a") / "clip.mp4")
    assert seen["set"] == [(cv2.CAP_PROP_ORIENTATION_AUTO, 1)]


# --------------------------------------------------------------------------- io/video.py: real files

@needs_ffmpeg
def test_probe_reads_a_real_clip_and_its_audio_and_names_unreadable_files(tmp_path) -> None:
    import courtvision.io.video as vid

    plain = _tiny_video(tmp_path / "plain.mp4")
    info = vid.probe(plain)
    assert (info.width, info.height, info.n_frames, info.has_audio) == (160, 120, 30, False)
    assert info.fps == pytest.approx(30.0) and info.duration == pytest.approx(1.0, abs=0.05)
    assert info.path == str(plain)

    ntsc = tmp_path / "ntsc.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=30000/1001", "-f", "lavfi", "-i", "sine=duration=1",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(ntsc))
    info = vid.probe(ntsc)
    assert info.has_audio and info.fps == pytest.approx(30000 / 1001, abs=1e-3)  # not rounded to 30

    sound = tmp_path / "sound.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=duration=1", str(sound))
    with pytest.raises(vid.UnreadableVideoError) as e:
        vid.probe(sound)
    assert str(e.value) == "sound.wav: dosyada video akışı yok"

    text = tmp_path / "notes.mp4"
    text.write_text("this is not a video\n")
    with pytest.raises(vid.UnreadableVideoError) as e:
        vid.probe(text)
    msg = str(e.value)
    assert msg.startswith("notes.mp4: video okunamadı (") and msg.endswith(")")
    assert str(tmp_path) not in msg and "@ 0x" not in msg and len(msg) <= len("notes.mp4: video okunamadı ()") + 120
    with pytest.raises(vid.UnreadableVideoError, match="missing.mp4: video okunamadı"):
        vid.probe(tmp_path / "missing.mp4")


@needs_ffmpeg
def test_probe_and_open_video_agree_on_the_upright_size_of_quarter_turn_clips(tmp_path) -> None:
    import courtvision.io.video as vid

    src = _tiny_video(tmp_path / "src.mp4")
    for turn, swapped in ((90, True), (-90, True), (180, False)):
        out = tmp_path / f"rot{turn}.mp4"
        _ffmpeg("-display_rotation", str(turn), "-i", str(src), "-c", "copy", str(out))
        info = vid.probe(out)
        cap = vid.open_video(out)
        try:
            assert cap.isOpened()
            ok, frame = cap.read()
        finally:
            cap.release()
        assert ok and (info.width, info.height) == ((120, 160) if swapped else (160, 120))
        assert frame.shape[:2] == (info.height, info.width)


@needs_ffmpeg
def test_open_video_gives_a_closed_capture_for_a_missing_or_unreadable_file(tmp_path) -> None:
    import courtvision.io.video as vid

    text = tmp_path / "notes.mp4"
    text.write_text("not a video\n")
    for path in (text, tmp_path / "missing.mp4"):
        cap = vid.open_video(path)
        assert not cap.isOpened()
        cap.release()


@needs_ffmpeg
def test_packet_intervals_of_real_files_are_sorted_even_when_frames_are_stored_out_of_order(tmp_path) -> None:
    import courtvision.io.video as vid

    ordered = _tiny_video(tmp_path / "b.mp4", seconds="2", extra=("-bf", "3", "-g", "30"))  # B-frames: decode order
    d = vid._frame_intervals(ordered)
    assert d is not None and len(d) == 59 and np.allclose(d, 1 / 30, atol=1e-4)
    assert vid.irregular_frame_intervals(ordered) == 0
    assert vid.nominal_rate(ordered) == 30

    ntsc = _tiny_video(tmp_path / "n.mp4", rate="30000/1001")
    assert vid.nominal_rate(ntsc) == Fraction(30000, 1001)

    gappy = tmp_path / "gappy.mp4"  # frames 10 and 11 missing: one interval of three frames
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1", "-vf", "select='not(between(n\\,10\\,11))'",
            "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(gappy))
    d = vid._frame_intervals(gappy)
    assert d is not None and len(d) == 27 and np.isclose(d.max(), 3 / 30, atol=1e-3)
    assert int((d > 0.09).sum()) == 1 and vid.irregular_frame_intervals(gappy) == 1
    assert vid.nominal_rate(gappy) == 30  # the median is still 30 fps


@needs_ffmpeg
def test_extract_audio_of_a_real_file_is_mono_at_the_requested_rate(tmp_path) -> None:
    import courtvision.io.video as vid

    stereo = tmp_path / "stereo.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ac", "2", "-c:a", "pcm_s16le", str(stereo))
    for rate in (8000, 16000):
        a = vid.extract_audio(stereo, rate)
        assert a.dtype == np.float32 and abs(len(a) - rate) <= rate // 100  # one channel: `rate` samples a second
        assert 0.05 < float(np.abs(a).max()) <= 1.0 and a.flags.writeable
    assert vid.extract_audio(stereo).shape[0] == pytest.approx(22050, abs=221)  # the default rate

    video_only = _tiny_video(tmp_path / "silent.mp4")
    assert vid.extract_audio(video_only, 8000).shape == (0,)
    junk = tmp_path / "junk.mp4"
    junk.write_text("not media")
    assert vid.extract_audio(junk).shape == (0,) and vid.extract_audio(tmp_path / "missing.mp4").shape == (0,)
    with_video = tmp_path / "av.mp4"  # audio inside a video container, with the picture ignored
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=30", "-i", str(stereo), "-t", "1", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(with_video))
    assert abs(len(vid.extract_audio(with_video, 8000)) - 8000) <= 8000 * 0.05


# --------------------------------------------------------------------------- calib/line_refine.py

def _row_profile(values: dict[int, float], size: int = 41, background: float = 0.0) -> np.ndarray:
    """A 1-D brightness profile, centre at index 20, as a 5-row image (rows identical)."""
    prof = np.full(size, background, dtype=np.float32)
    for k, v in values.items():
        prof[20 + k] = v
    return np.tile(prof, (5, 1))


def _peak(img: np.ndarray, at: float = 20.0, n=(1.0, 0.0), reach: int = 10, min_contrast: float = 18.0):
    from courtvision.calib.line_refine import _line_profile_peak

    return _line_profile_peak(img, np.array([at, 2.0]), np.array(n), reach, min_contrast)


def _vline(x: float, h: int = 101, w: int = 40, thick: int = 3) -> np.ndarray:
    """A white anti-aliased vertical line centred on the (sub-pixel) column x, pixel centres at integers."""
    im = np.zeros((h, w), np.uint8)
    xi = int(round(x * 16))
    cv2.line(im, (xi, 0), (xi, h * 16), 255, thick, cv2.LINE_AA, 4)
    return im.astype(np.float32)


def test_the_ridge_offset_follows_a_sub_pixel_line_in_both_normal_directions() -> None:
    for x in (30.0, 30.25, 30.5, 30.75, 31.0):
        img = _vline(x)
        right = _peak(img, 25.0, (1.0, 0.0))  # offset along +x from the sample at x=25
        left = _peak(img, 35.0, (-1.0, 0.0))  # the same line seen along -x from x=35
        assert right == pytest.approx(x - 25.0, abs=0.08)
        assert left == pytest.approx(35.0 - x, abs=0.08)
    assert _peak(_vline(30.0), 30.0) == pytest.approx(0.0, abs=0.05)  # on the line: nothing to correct


def test_the_ridge_offset_is_measured_along_a_slanted_normal() -> None:
    img = np.zeros((80, 80), np.uint8)
    cv2.line(img, (0, 0), (80 * 16, 80 * 16), 255, 3, cv2.LINE_AA, 4)  # the diagonal y = x
    n = np.array([1.0, -1.0]) / np.sqrt(2.0)  # unit normal of the diagonal
    from courtvision.calib.line_refine import _line_profile_peak

    p = np.array([36.0, 44.0])  # 8 / sqrt(2) away from the diagonal, on the +n side
    off = _line_profile_peak(img.astype(np.float32), p, n, 10, 18.0)
    assert off == pytest.approx(8.0 / np.sqrt(2.0), abs=0.15)
    off_back = _line_profile_peak(img.astype(np.float32), p, -n, 10, 18.0)
    assert off_back == pytest.approx(-8.0 / np.sqrt(2.0), abs=0.15)  # reversing the normal flips the sign


def test_the_ridge_centre_is_the_half_height_midpoint_not_the_brightest_pixel() -> None:
    assert _peak(_row_profile({-1: 100, 0: 200, 1: 100})) == 0.0  # symmetric triangle
    assert _peak(_row_profile({-1: 200, 0: 200, 1: 200})) == 0.0  # a flat top: argmax alone would say -1
    assert _peak(_row_profile({0: 200, 1: 200, 2: 200})) == 1.0
    assert _peak(_row_profile({-1: 200, 0: 200, 1: 200, 2: 200})) == 0.5  # an even width: between two pixels
    assert _peak(_row_profile({1: 100, 2: 200, 3: 100})) == 2.0
    assert _peak(_row_profile({1: 100, 2: 200, 3: 100}), n=(-1.0, 0.0)) == -2.0  # the normal's sign is the offset's
    assert _peak(_row_profile({-3: 200, -2: 200, 4: 150})) == -2.5  # the weaker second ridge does not count
    assert _peak(_row_profile({-2: 200, 2: 200})) == -2.0  # equal ridges: the first (argmax)
    # a vertical normal samples down the image: the ridge is a horizontal line at row 22
    img = np.tile(_row_profile({-1: 100, 0: 200, 1: 100})[0][:, None], (1, 5))
    from courtvision.calib.line_refine import _line_profile_peak

    assert _line_profile_peak(img, np.array([2.0, 17.0]), np.array([0.0, 1.0]), 10, 18.0) == 3.0


def test_the_ridge_must_stand_out_from_the_darker_side_by_the_minimum_contrast() -> None:
    assert _peak(_row_profile({0: 18})) == 0.0  # exactly the contrast: accepted
    assert _peak(_row_profile({0: 17.5})) is None
    assert _peak(_row_profile({0: 50}), min_contrast=50.0) == 0.0
    assert _peak(_row_profile({0: 50}), min_contrast=51.0) is None
    assert _peak(_row_profile({0: 100})) == 0.0
    # each side has its own background and the *brighter* one decides
    left_bg = _row_profile({0: 100})
    left_bg[:, :20] = 82.0
    assert _peak(left_bg) == 0.0
    left_bg[:, :20] = 83.0
    assert _peak(left_bg) is None
    right_bg = _row_profile({0: 100})
    right_bg[:, 21:] = 83.0
    assert _peak(right_bg) is None  # and the same from the other side
    right_bg[:, 21:] = 82.0
    assert _peak(right_bg) == 0.0
    # a step edge (dark court, lighter run-off) with a line on it: the background is each side's own minimum
    step = _row_profile({0: 140}, background=0.0)
    step[:, :20] = 40.0
    assert _peak(step) == 0.0
    assert _peak(np.full((5, 41), 100.0, np.float32)) is None  # no ridge at all


def test_a_ridge_that_touches_the_search_window_is_refused() -> None:
    assert _peak(_row_profile({-10: 200})) is None and _peak(_row_profile({10: 200})) is None  # brightest at an end
    assert _peak(_row_profile({-9: 200})) == -9.0 and _peak(_row_profile({9: 200})) == 9.0  # one pixel inside is fine
    wide = {k: 200.0 for k in range(-10, 3)}  # the paint runs off the left end of the window
    assert _peak(_row_profile(wide)) is None
    assert _peak(_row_profile({k: 200.0 for k in range(-2, 11)})) is None  # ... or the right end
    flat_to_the_end = _row_profile({k: 100.0 for k in range(5, 21)})  # zero contrast allowed, paint to the end
    assert _peak(flat_to_the_end, min_contrast=0.0) is None
    assert _peak(_row_profile({4: 200}), reach=3) is None  # outside a window of +-3
    assert _peak(_row_profile({4: 200}), reach=5) == 4.0
    assert _peak(_row_profile({-1: 100, 0: 200, 1: 100}), reach=2) == 0.0
    # Pinned as it is: a ridge shoulder on the window's end counts as the left background (line_refine.py:46), so a
    # ridge cut by the window is returned, biased, instead of refused
    assert _peak(_row_profile({-10: 100, -9: 200, -8: 100})) == pytest.approx(-8.75)


def test_a_nan_in_the_profile_or_a_nan_position_does_not_become_a_ridge_position() -> None:
    # Pinned as it is: a NaN pixel comes back as NaN, not None (every comparison with NaN is False); frames are
    # uint8, so this only matters if a caller passes NaN pixels.
    nan_img = _row_profile({0: 200})
    nan_img[:, 20] = np.nan
    assert np.isnan(_peak(nan_img))
    img = _vline(20.0)
    assert _peak(img, float("nan"), (1.0, 0.0)) is None  # NaN sample position: remap gives border zeros
    assert _peak(img, 20.0, (float("nan"), 0.0)) is None


def _court(*segments):
    """A stand-in court with the given (start, end) world segments."""
    return types.SimpleNamespace(line_segments=lambda: [(f"l{i}", np.array(a, float), np.array(b, float))
                                                        for i, (a, b) in enumerate(segments)])


def test_ridge_samples_sit_at_four_and_ninety_six_percent_and_lie_on_the_found_ridge() -> None:
    from courtvision.calib.line_refine import _sample_ridges

    court = _court(((20, 0), (20, 100)))
    img = _vline(20.0)
    pts, ids, tried = _sample_ridges(img, np.eye(3), court, 2, 4, 18.0)
    assert (tried, ids) == (2, [0, 0]) and isinstance(pts, list)
    assert [round(float(p[1])) for p in pts] == [4, 96]  # ts = 0.04 and 0.96 of the segment
    assert all(abs(p[0] - 20.0) < 0.05 for p in pts)
    pts, ids, tried, offs = _sample_ridges(img, np.eye(3), court, 3, 4, 18.0, return_offsets=True)
    assert tried == 3 and [round(float(p[1])) for p in pts] == [4, 50, 96]
    assert isinstance(offs, np.ndarray) and offs.shape == (3,) and np.abs(offs).max() < 0.05
    assert len(_sample_ridges(img, np.eye(3), court, 3, 4, 18.0)) == 3  # no offsets unless asked for


def test_ridge_offsets_are_signed_along_the_line_normal_and_the_points_land_on_the_paint() -> None:
    from courtvision.calib.line_refine import _sample_ridges

    img = _vline(20.0, h=220)
    up, down = _court(((20, 0), (20, 200))), _court(((20, 200), (20, 0)))
    for dx in (-2.0, 2.0):  # the projection sits dx away from the paint
        H = np.array([[1, 0, -dx], [0, 1, 0], [0, 0, 1]])
        for court, sign in ((up, 1.0), (down, -1.0)):
            pts, ids, tried, offs = _sample_ridges(img, H, court, 4, 8, 18.0, return_offsets=True)
            assert tried == 4 and len(pts) == 4
            # normal of a segment running +y is -x, of one running -y it is +x
            assert offs == pytest.approx(np.full(4, -sign * dx), abs=0.1)
            assert all(abs(p[0] - 20.0) < 0.1 for p in pts)


def test_ridge_samples_must_be_a_full_reach_away_from_every_image_edge() -> None:
    from courtvision.calib.line_refine import _sample_ridges

    def tried(x0: float, h: int = 101, w: int = 40) -> int:
        return _sample_ridges(_vline(x0, h, w), np.eye(3), _court(((x0, 0), (x0, 100))), 2, 4, 18.0)[2]

    assert tried(3.9) == 0 and tried(4.0) == 2  # left edge: reach <= x
    assert tried(35.0) == 2 and tried(36.0) == 0  # right edge: x < w - reach
    assert tried(20.0, h=100) == 1  # the sample at y=96 is not < 100 - 4; the one at y=4 is (reach <= y)
    assert tried(20.0, h=101) == 2

    # the top edge: a horizontal segment sampled at y = 3.9 / 4.0
    def tried_y(y0: float) -> int:
        return _sample_ridges(np.zeros((40, 101), np.float32), np.eye(3), _court(((0, y0), (100, y0))), 2, 4, 18.0)[2]

    assert tried_y(3.9) == 0 and tried_y(4.0) == 2 and tried_y(35.0) == 2 and tried_y(36.0) == 0


def test_ridge_sampling_skips_degenerate_lines_and_counts_samples_without_a_ridge() -> None:
    from courtvision.calib.line_refine import _sample_ridges

    img = _vline(20.0)
    assert _sample_ridges(img, np.eye(3), _court(((20, 50), (20, 50))), 3, 4, 18.0) == ([], [], 0)  # a point
    pts, ids, tried = _sample_ridges(img, np.eye(3), _court(((20, 0), (20, 100)), ((10, 0), (10, 100))), 2, 4, 18.0)
    assert (tried, ids) == (4, [0, 0]) and len(pts) == 2  # the second line has no paint: tried, not found
    # a line ending at infinity (homography sending an end to w=0) is skipped rather than raising
    pts, ids, tried = _sample_ridges(np.zeros((50, 50), np.float32), np.eye(3), _court(((5, 5), (5, 5.0))), 2, 4, 18.0)
    assert tried == 0
    blank = np.zeros((101, 40), np.float32)
    assert _sample_ridges(blank, np.eye(3), _court(((20, 0), (20, 100))), 2, 4, 18.0) == ([], [], 2)
    nan_img = np.full((101, 40), np.nan, np.float32)
    assert _sample_ridges(nan_img, np.eye(3), _court(((20, 0), (20, 100))), 2, 4, 18.0) == ([], [], 2)
    nan_h = np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert _sample_ridges(img, nan_h, _court(((20, 0), (20, 100))), 2, 4, 18.0) == ([], [], 0)  # nothing to try


def test_perpendicular_residuals_are_pixel_distances_positive_to_the_left_of_the_projected_line() -> None:
    from courtvision.calib.line_refine import _perp_residuals

    ident = np.array([1, 0, 0, 0, 1, 0, 0, 0], float)
    ends = np.array([[[0, 0], [0, 10]], [[0, 0], [10, 0]]], float)  # one line along +y, one along +x
    pts = np.array([[2, 5], [3, 4], [3, 4], [0, 7]], float)
    r = _perp_residuals(ident, pts, np.array([0, 1, 1, 0]), ends)
    # the unit normal of (dx, dy) is (-dy, dx): -x for the first line, +y for the second
    assert r == pytest.approx([-2.0, 4.0, 4.0, 0.0])
    assert _perp_residuals(ident, pts[:1], np.array([1]), ends) == pytest.approx([5.0])  # line_ids pick the line
    # reversing a line's ends flips the sign; its length does not matter
    flipped = ends[:, ::-1]
    assert _perp_residuals(ident, pts, np.array([0, 1, 1, 0]), flipped) == pytest.approx([2.0, -4.0, -4.0, 0.0])
    long_ends = ends * 7.0
    assert _perp_residuals(ident, pts, np.array([0, 1, 1, 0]), long_ends) == pytest.approx(r)
    # the homography scales world metres to pixels: residuals come out in pixels
    scale3 = np.array([3, 0, 0, 0, 3, 0, 0, 0], float)
    assert _perp_residuals(scale3, np.array([[6.0, 15.0]]), np.array([0]), ends) == pytest.approx([-6.0])
    # a point on the projected line has residual 0, also under perspective and translation
    H = np.array([[2.0, 0.3, 40.0], [0.1, 1.5, 25.0], [0.001, 0.002, 1.0]])
    from courtvision.calib.camera import apply_h

    a, b = apply_h(H, ends[0, 0]), apply_h(H, ends[0, 1])
    on_line = a + 0.37 * (b - a)
    assert _perp_residuals(H.ravel()[:8], on_line[None], np.array([0]), ends) == pytest.approx([0.0], abs=1e-9)
    d = (b - a) / np.linalg.norm(b - a)
    left_of_line = on_line + 2.5 * np.array([-d[1], d[0]])
    assert _perp_residuals(H.ravel()[:8], left_of_line[None], np.array([0]), ends) == pytest.approx([2.5])


def test_perpendicular_residuals_of_a_collapsed_line_are_not_finite() -> None:
    from courtvision.calib.line_refine import _perp_residuals

    ends = np.array([[[0, 0], [0, 10]]], float)
    with np.errstate(all="ignore"):  # the refinement tests for this and keeps the last good homography
        r = _perp_residuals(np.zeros(8), np.array([[1.0, 1.0]]), np.array([0]), ends)
    assert not np.isfinite(r).any()


def test_line_support_is_the_fraction_of_samples_with_a_ridge_within_tolerance() -> None:
    from courtvision.calib.line_refine import line_support

    img = _vline(20.0, h=200)
    both = _court(((20, 0), (20, 200)), ((8, 0), (8, 200)))  # the second line has no paint
    assert line_support(img, np.eye(3), _court(((20, 0), (20, 200))), per_line=4, reach=8) == 1.0
    assert line_support(img, np.eye(3), both, per_line=4, reach=8) == 0.5
    assert line_support(np.zeros((200, 40), np.float32), np.eye(3), both, per_line=4, reach=8) == 0.0
    assert line_support(img, np.eye(3), _court(((20, 0), (20, 100))), per_line=4, reach=30) == 0.0  # nothing tried
    off_by_2 = np.array([[1, 0, -2.0], [0, 1, 0], [0, 0, 1]])  # projection 2 px from the paint
    one = _court(((20, 0), (20, 200)))
    assert line_support(img, off_by_2, one, per_line=4, reach=8, tol_px=3.0) == 1.0
    assert line_support(img, off_by_2, one, per_line=4, reach=8, tol_px=1.5) == 0.0
    assert line_support(img, off_by_2, one, per_line=4, reach=8, tol_px=2.5) == 1.0
    assert line_support(img, off_by_2, one, per_line=4, reach=8, tol_px=3.0, min_contrast=300.0) == 0.0
    exact = np.zeros((120, 40), np.float32)  # ridge exactly 2.0 px from the projection (integer pixel profile)
    exact[:, 21], exact[:, 22], exact[:, 23] = 100, 200, 100
    on_the_edge = _court(((20, 0), (20, 120)))
    assert line_support(exact, np.eye(3), on_the_edge, per_line=3, reach=8, tol_px=2.0) == 1.0  # <= tol_px
    assert line_support(exact, np.eye(3), on_the_edge, per_line=3, reach=8, tol_px=1.99) == 0.0
    # the offset is judged in either direction: a projection 2 px to the other side counts too
    assert line_support(img, np.array([[1, 0, 2.0], [0, 1, 0], [0, 0, 1]]), one, per_line=4, reach=8) == 1.0


def test_refinement_without_enough_ridge_points_returns_the_normalised_input_and_says_so() -> None:
    from courtvision.calib.line_refine import refine_homography
    from courtvision.geometry.court import CourtModel

    H = 2.0 * np.array([[30.0, 0, 300], [0, -30.0, 400], [0, 0, 1]])  # a court 660 px wide on a black frame
    ref = refine_homography(np.zeros((800, 600, 3), np.uint8), H, CourtModel())
    assert ref.H[2, 2] == 1.0 and np.allclose(ref.H, H / H[2, 2])  # divided by its own corner, not refitted
    assert ref.n_points == 0 and np.isnan(ref.rmse_px) and ref.support == 0.0


# --------------------------------------------------------------------------- perception/court.py: cut helpers

def _solid(bgr, w: int = 320, h: int = 180) -> np.ndarray:
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = bgr
    return frame


def _correl(a, b) -> float:
    return cv2.compareHist(np.asarray(a, np.float32), np.asarray(b, np.float32), cv2.HISTCMP_CORREL)


def test_the_colour_histogram_is_a_unit_length_hue_by_saturation_count_that_ignores_brightness_and_size() -> None:
    from courtvision.perception.court import frame_hist

    def bins(frame) -> list[list[int]]:
        return np.argwhere(frame_hist(frame) > 0).tolist()

    red = frame_hist(_solid((0, 0, 255)))
    assert red.shape == (30, 16) and red.dtype == np.float32  # 30 hue bins (rows), 16 saturation bins (columns)
    assert float(np.linalg.norm(red)) == pytest.approx(1.0)  # normalised, so frame size never matters
    assert bins(_solid((0, 0, 255))) == [[0, 15]]  # OpenCV hue 0, full saturation
    assert bins(_solid((0, 255, 0))) == [[10, 15]]  # hue 60 of 180: bin 10
    assert bins(_solid((255, 0, 0))) == [[20, 15]]  # hue 120: bin 20
    assert bins(_solid((255, 128, 128))) == [[20, 7]]  # half saturation: bin 7
    assert bins(_solid((50, 50, 50))) == [[0, 0]] == bins(_solid((200, 200, 200)))  # greys: hue and saturation 0
    assert bins(_solid((0, 0, 100))) == [[0, 15]]  # brightness (V) is not in the histogram
    assert np.array_equal(frame_hist(_solid((50, 50, 50))), frame_hist(_solid((200, 200, 200))))
    assert np.allclose(frame_hist(_solid((0, 0, 255), 640, 360)), red)  # resized to 160 x 90 first
    half = np.zeros((90, 160, 3), np.uint8)  # half the pixels red, half blue: two bins, equal weight
    half[:, :80, 2] = 255
    half[:, 80:, 0] = 255
    h = frame_hist(half)
    assert np.argwhere(h > 0).tolist() == [[0, 15], [20, 15]] and h[0, 15] == pytest.approx(h[20, 15])
    assert h[0, 15] == pytest.approx(np.sqrt(0.5))


def test_a_shot_cut_is_a_correlation_strictly_below_the_threshold_against_the_frame_before() -> None:
    from courtvision.perception.court import detect_shot_cuts

    a = np.array([1, 0, 0, 0], np.float32)
    b = np.array([1, 1, 0, 0], np.float32)  # correlation with a: 0.577
    c = np.array([3, 1, 0, 0], np.float32)
    d = np.array([1, 2, 0, 0], np.float32)  # correlation of c and d: 0.492
    assert _correl(a, b) == pytest.approx(0.577, abs=1e-3) and _correl(c, d) == pytest.approx(0.492, abs=1e-3)
    assert detect_shot_cuts([a, b]) == [0]  # above the default 0.55
    assert detect_shot_cuts([c, d]) == [0, 1]  # below it
    corr = _correl(a, b)
    assert detect_shot_cuts([a, b], threshold=corr) == [0]  # equal is not a cut
    assert detect_shot_cuts([a, b], threshold=float(np.nextafter(corr, 2.0))) == [0, 1]
    assert detect_shot_cuts([a, b], threshold=0.0) == [0] and detect_shot_cuts([c, d], threshold=0.49) == [0]
    assert detect_shot_cuts([a, a, a], threshold=2.0) == [0, 1, 2]  # everything below 2: every frame starts a shot


def test_cuts_are_indexed_by_the_first_frame_of_the_new_shot_and_compare_neighbours_only() -> None:
    from courtvision.perception.court import detect_shot_cuts, frame_hist

    assert detect_shot_cuts([]) == [0]
    assert detect_shot_cuts([np.ones((4, 1), np.float32)]) == [0]
    red, blue, green = (frame_hist(_solid(c)) for c in ((0, 0, 255), (255, 0, 0), (0, 255, 0)))
    assert detect_shot_cuts([red, red, red, blue, blue, green]) == [0, 3, 5]
    # a drift that never drops below the threshold from one frame to the next is no cut, though first and last differ
    chain = [np.array(v, np.float32) for v in ([5, 2, 0, 0], [3, 3, 1, 0], [2, 3, 2, 0], [1, 3, 3, 0], [0, 2, 4, 1])]
    assert min(_correl(x, y) for x, y in zip(chain, chain[1:], strict=False)) > 0.55 > _correl(chain[0], chain[-1])
    assert detect_shot_cuts(chain) == [0]
    # real frames: the same noisy scene (even a little brighter) is no cut, another scene is
    court = np.zeros((90, 160, 3), np.uint8)
    court[:] = (120, 90, 40)
    court[40:50, :] = (235, 235, 235)

    def shot(seed: int, brighter: int = 0) -> np.ndarray:
        noise = np.random.default_rng(seed).normal(0, 6, court.shape)
        return np.clip(court.astype(int) + brighter + noise, 0, 255).astype(np.uint8)

    crowd = np.random.default_rng(0).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    frames = (shot(1), shot(2), shot(3, 10), crowd, crowd)
    assert detect_shot_cuts([frame_hist(f) for f in frames]) == [0, 3]


def test_the_grey_thumbnail_is_160_by_90_signed_luma_averaged_over_each_block() -> None:
    from courtvision.perception.court import grey_thumb

    for colour, luma in (((0, 0, 255), 76), ((0, 255, 0), 150), ((255, 0, 0), 29), ((255, 255, 255), 255),
                         ((0, 0, 0), 0)):  # BGR input: 0.299 R + 0.587 G + 0.114 B
        t = grey_thumb(_solid(colour))
        assert t.shape == (90, 160) and t.dtype == np.int16 and set(np.unique(t)) == {luma}
    assert grey_thumb(_solid((255, 255, 255), 1920, 1080)).shape == (90, 160)
    split = np.zeros((180, 320, 3), np.uint8)
    split[:, 160:] = 255
    t = grey_thumb(split)
    assert t[0, 79] == 0 and t[0, 80] == 255 and t.mean() == 127.5
    checker = np.zeros((180, 320, 3), np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    assert np.unique(grey_thumb(checker)).tolist() == [128]  # INTER_AREA averages, it does not sample one pixel
    third = np.zeros((270, 480, 3), np.uint8)  # 3 x 3 source pixels per thumbnail pixel
    third[::2, ::2] = 255
    third[1::2, 1::2] = 255
    assert np.unique(grey_thumb(third)).tolist() == [113, 142]  # block means (4 or 5 of 9 white), not single samples
    # the thumbnail is signed, so a difference of two of them cannot wrap around like uint8
    assert int((grey_thumb(_solid((0, 0, 0))) - grey_thumb(_solid((255, 255, 255)))).min()) == -255


def test_thumb_difference_is_the_mean_absolute_grey_gap_and_zero_without_a_frame_before() -> None:
    from courtvision.perception.court import grey_thumb, thumb_difference

    dark, light = grey_thumb(_solid((50, 50, 50))), grey_thumb(_solid((200, 200, 200)))
    assert thumb_difference(None, light) == 0.0 and isinstance(thumb_difference(None, light), float)
    assert thumb_difference(dark, light) == 150.0 and thumb_difference(light, dark) == 150.0  # symmetric
    assert thumb_difference(light, light) == 0.0
    assert type(thumb_difference(dark, light)) is float
    quarter = dark.copy()
    quarter[:45, :80] = 250  # a quarter of the picture changes by 200
    assert thumb_difference(dark, quarter) == 50.0
    assert thumb_difference(grey_thumb(_solid((0, 0, 0))), grey_thumb(_solid((255, 255, 255)))) == 255.0


def test_pick_device_prefers_an_explicit_name_then_mps_then_cuda_then_cpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from courtvision.perception.court import pick_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device().type == "mps" and pick_device("auto").type == "mps"  # Apple first, as the project is run
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("cpu") == torch.device("cpu")  # an explicit choice is never second-guessed
    assert pick_device("cuda:1").type == "cuda" and pick_device("cuda:1").index == 1
    with pytest.raises(RuntimeError):
        pick_device("not-a-device")


def test_same_angle_edits_sit_exactly_on_their_difference_ratio_and_history_limits() -> None:
    from courtvision.perception.court import STILL_DIFF, same_angle_edits

    ok = [True] * 8

    def edits(diffs, valid=None, **kw) -> list[int]:
        return same_angle_edits(np.array(diffs, np.float32), valid or [True] * len(diffs), **kw)

    assert STILL_DIFF == 0.05
    assert edits([0, .5, .5, .5, 2.5]) == [4]  # the threshold itself (>=): 2.5 grey levels
    assert edits([0, .5, .5, .5, 2.49]) == []
    assert edits([0, 1, 1, 1, 3.0]) == [4]  # 3 times the median of the past
    assert edits([0, 1, 1, 1, 2.99]) == []
    assert edits([0, 1, 1, 1, 3.0], threshold=3.5) == []  # both conditions are needed
    assert edits([0, .05, .05, .05, 2.5], ratio=3.0, threshold=0.0) == [4]
    assert edits([0, 0, 0, 0, .16], threshold=0.1) == [4]  # a still picture: the past counts as STILL_DIFF
    assert edits([0, 0, 0, 0, .14], threshold=0.1) == []  # 3 * 0.05 = 0.15 is the floor, not 0
    assert edits([0, 1, 3, 3, 7.0]) == []  # frame 0's placeholder 0 is not part of the past (median 3, not 2)
    assert edits([0, 1, 3, 3, 9.0]) == [4]
    assert edits([0, .5, .5, 2.6]) == []  # i=3: only 2 differences before it
    # only the last `past` differences count: calm for 6 frames after a lively stretch
    lively_then_calm = [0] + [9.0] * 6 + [.3] * 6 + [2.6]
    assert edits(lively_then_calm)[-1] == 13 and edits(lively_then_calm, past=12) == []
    assert edits([0, .3, .3, .3, .3, .3, .3, 2.6], past=3) == [7]
    # both frames must be calibrated, the new one as well as the one before
    diffs = [0, .5, .5, .5, 5.0]
    assert edits(diffs, [True, True, True, True, False]) == []
    assert edits(diffs, [True, True, True, False, True]) == []
    assert edits(diffs, [False, True, True, True, True]) == [4]  # frame 0 is never looked at
    assert edits(diffs, ok[:5]) == [4] and edits([], []) == [] and edits([0.0], [True]) == []
