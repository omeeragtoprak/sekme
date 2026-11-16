"""Gaps found by a test review in the command line (cli.py), the environment check (doctor.py), the Review Room
helpers host_name / media_path / read_jsonl (api/app.py) and the Turkish number format (i18n.num).

Nothing real runs: the pipeline, uvicorn, the benchmark, ffmpeg, OpenCV, torch and the disk are replaced; the
expected values were read off the code by running it. Behaviour that looks wrong but is pinned as it is carries
a ``# pinned:`` comment."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import courtvision.cli as cli
import courtvision.doctor as doctor
import courtvision.pipeline as pipeline
from courtvision.api.app import RUN_ID, create_app, host_name, media_path, read_jsonl
from courtvision.errors import MissingModelError, UnreadableVideoError
from courtvision.i18n import num

LOCAL = "http://127.0.0.1"


@pytest.fixture(autouse=True)
def _fit_workers_env(monkeypatch) -> None:
    """The command line writes SEKME_FIT_WORKERS into the real environment: start clean, restore afterwards."""
    monkeypatch.setenv("SEKME_FIT_WORKERS", "x")
    monkeypatch.delenv("SEKME_FIT_WORKERS")


# --- 1. command line: analyze ----------------------------------------------------------------------------------

def _report(**video) -> dict:
    return {"video": {"fps": 30.0, **video},
            "calibration": {"valid_frames": 3, "total_frames": 5, "line_rmse_px_median": 0.42}, "flights": []}


def _analyze(monkeypatch, argv: list[str], report: dict | None = None):
    """Run ``sekme analyze`` with the pipeline replaced; returns (exit code, what the pipeline was called with)."""
    seen: dict = {}

    def fake(video, out, cfg, progress):
        seen.update(video=video, out=out, cfg=cfg, progress=progress)
        return report if report is not None else _report()

    monkeypatch.setattr(pipeline, "analyze", fake)
    code = cli.main(["analyze", *argv])
    return code, seen


def test_analyze_without_flags_hands_the_pipeline_its_defaults(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "default_fit_workers", lambda: 3)
    code, seen = _analyze(monkeypatch, ["some/dir/My Clip.v2.mp4"])
    assert code == 0
    assert seen["video"] == "some/dir/My Clip.v2.mp4" and isinstance(seen["video"], str)  # not resolved, not a Path
    assert seen["out"] == Path("runs") / "My Clip.v2"  # the file name without its last suffix
    assert seen["cfg"] == pipeline.PipelineConfig()  # every default of the command line is the library default
    assert seen["cfg"].device == "auto" and seen["cfg"].ball_stride == 0 and seen["cfg"].serve_toss_bridge_s == 0.0
    assert callable(seen["progress"])
    assert cli.os.environ["SEKME_FIT_WORKERS"] == "3"
    assert capsys.readouterr().out.splitlines()[-1] == f"report: {Path('runs') / 'My Clip.v2' / 'report.json'}"


@pytest.mark.parametrize("flag,field,value", [
    ("--no-render", "render", False), ("--no-broadcast", "broadcast", False),
    ("--reuse-perception", "reuse_perception", True), ("--pin-serve-height", "pin_serve_height", True),
    ("--fps-scaled-thresholds", "fps_scaled_thresholds", True),
    ("--player-size-from-camera", "player_size_from_camera", True),
    ("--toss-bridge", "serve_toss_bridge_s", 0.3)])
def test_each_switch_changes_exactly_its_own_setting(monkeypatch, flag: str, field: str, value) -> None:
    _, seen = _analyze(monkeypatch, ["v.mp4", flag])
    assert seen["cfg"] == dataclasses.replace(pipeline.PipelineConfig(), **{field: value})


def test_value_flags_reach_the_pipeline_and_a_blank_out_falls_back_to_the_run_name(monkeypatch) -> None:
    _, seen = _analyze(monkeypatch, ["clip.mp4", "--device", "cpu", "--ball-stride", "2", "--out", "elsewhere"])
    assert (seen["cfg"].device, seen["cfg"].ball_stride, seen["out"]) == ("cpu", 2, Path("elsewhere"))
    _, seen = _analyze(monkeypatch, ["dir/clip", "--out", ""])  # an empty --out is no --out; no suffix to cut
    assert seen["out"] == Path("runs") / "clip"


def test_model_paths_are_resolved_and_a_model_flag_beats_ball_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    default = pipeline.PipelineConfig().models
    _, seen = _analyze(monkeypatch, ["v.mp4", "--ball-model", "b.pt", "--model", "court=sub/c.pt",
                                     "--model", "pose=p=q.pt"])
    models = seen["cfg"].models
    assert models["ball"] == str((tmp_path / "b.pt").resolve())
    assert models["court"] == str((tmp_path / "sub" / "c.pt").resolve())
    assert models["pose"] == str((tmp_path / "p=q.pt").resolve())  # only the first "=" splits key from path
    assert models["players"] == default["players"]  # untouched
    _, seen = _analyze(monkeypatch, ["v.mp4", "--model", "ball=m1.pt", "--ball-model", "b.pt", "--model",
                                     "ball=m2.pt"])
    assert seen["cfg"].models["ball"] == str((tmp_path / "m2.pt").resolve())  # --model is applied last, last wins


@pytest.mark.parametrize("item", ["ball", "ball=", "=x.pt", "nope=x.pt", "Ball=x.pt"])
def test_a_malformed_model_option_stops_before_anything_runs(monkeypatch, item: str) -> None:
    called: list = []
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: called.append(a))
    with pytest.raises(SystemExit) as e:
        cli.main(["analyze", "v.mp4", "--model", item])
    assert e.value.code == (f"--model expects KEY=PATH with KEY in ['ball', 'court', 'players', 'pose'], "
                            f"got {item!r}")
    assert not called and "SEKME_FIT_WORKERS" not in cli.os.environ


@pytest.mark.parametrize("given,expect", [(0, "1"), (-5, "1"), (1, "1"), (7, "7"), (99, "99")])
def test_an_explicit_worker_count_is_at_least_one(monkeypatch, given: int, expect: str) -> None:
    monkeypatch.setenv("SEKME_FIT_WORKERS", "2")
    cli._set_fit_workers(given)
    assert cli.os.environ["SEKME_FIT_WORKERS"] == expect  # the flag beats the environment setting


def test_without_a_worker_count_an_empty_setting_counts_as_unset_and_a_set_one_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(cli, "default_fit_workers", lambda: 4)
    monkeypatch.setenv("SEKME_FIT_WORKERS", "")
    cli._set_fit_workers(None)
    assert cli.os.environ["SEKME_FIT_WORKERS"] == "4"
    monkeypatch.setenv("SEKME_FIT_WORKERS", "junk")
    cli._set_fit_workers(None)
    assert cli.os.environ["SEKME_FIT_WORKERS"] == "junk"  # not validated here; the fitting code reads it


@pytest.mark.parametrize("cores,expect", [(None, 1), (1, 1), (2, 1), (3, 1), (4, 2), (8, 4), (10, 4), (64, 4)])
def test_default_fit_workers_is_half_the_cores_between_one_and_four(monkeypatch, cores, expect: int) -> None:
    monkeypatch.setattr(cli.os, "cpu_count", lambda: cores)
    assert cli.default_fit_workers() == expect


def test_progress_is_written_to_stderr_over_the_same_line(monkeypatch, capsys) -> None:
    clock = iter([100.0, 112.34, 112.34])
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=lambda: next(clock)))
    seen: dict = {}

    def fake(video, out, cfg, progress):
        progress("calibration", 1.0)
        progress("rally", 0.0625)
        seen["ok"] = True
        return _report()

    monkeypatch.setattr(pipeline, "analyze", fake)
    assert cli.main(["analyze", "v.mp4"]) == 0 and seen["ok"]
    cap = capsys.readouterr()
    assert cap.err == "\r[  12.3s] calibration 100.0%\r[  12.3s] rally        6.2%\n"
    assert "%" not in cap.out


@pytest.mark.parametrize("fps,shown", [(23.99, "24"), (15.0, "15"), (1.0, "1")])
def test_a_frame_rate_below_24_warns_and_24_or_unknown_does_not(monkeypatch, capsys, fps: float, shown: str) -> None:
    _analyze(monkeypatch, ["v.mp4"], _report(fps=fps))
    assert f"uyarı: kare hızı düşük ({shown} fps); sekmelerin derinliği güvenilmez\n" in capsys.readouterr().err
    # pinned: 23.99 fps is announced as "24 fps" (one decimal would not read as the limit it is below)


@pytest.mark.parametrize("fps", [24.0, 25.0, 59.94, 0.0])
def test_no_low_frame_rate_warning_from_24_up_or_when_the_rate_is_unknown(monkeypatch, capsys, fps: float) -> None:
    _analyze(monkeypatch, ["v.mp4"], _report(fps=fps))
    assert "kare hızı düşük" not in capsys.readouterr().err


@pytest.mark.parametrize("order", ["tt", "bb", "tb", "bt"])
def test_an_interlaced_field_order_warns_with_the_order(monkeypatch, capsys, order: str) -> None:
    _analyze(monkeypatch, ["v.mp4"], _report(field_order=order))
    assert (f"uyarı: video geçmeli (interlaced, alan sırası {order}); zamanlar yarım kare kayabilir, "
            "hızlar şaşabilir\n") in capsys.readouterr().err


@pytest.mark.parametrize("order", ["progressive", "unknown", "", None, "TT"])
def test_other_field_orders_do_not_warn(monkeypatch, capsys, order) -> None:
    _analyze(monkeypatch, ["v.mp4"], _report(field_order=order))
    assert "geçmeli" not in capsys.readouterr().err


@pytest.mark.parametrize("valid,total,warns", [(0, 5, True), (1, 5, False), (5, 5, False), (0, 0, False)])
def test_the_calibration_warning_needs_frames_and_not_one_calibrated(monkeypatch, capsys, valid, total, warns) -> None:
    rep = _report()
    rep["calibration"] = {"valid_frames": valid, "total_frames": total, "line_rmse_px_median": 0.5}
    _analyze(monkeypatch, ["v.mp4"], rep)
    cap = capsys.readouterr()
    assert ("uyarı: kort hiçbir karede kalibre edilemedi; ölçüm ve karar yok" in cap.err) is warns
    assert f"calibrated frames {valid}/{total}, line rmse 0.5 px\n" in cap.out


def test_the_variable_rate_notes_depend_on_the_copy_and_the_count(monkeypatch, capsys) -> None:
    copy = {"file": "c.mp4", "fps": 25.0, "added_frames": 2}
    _analyze(monkeypatch, ["v.mp4"], _report(irregular_frame_intervals=4, constant_rate_copy=copy))
    err = capsys.readouterr().err
    assert err == ("\nnot: değişken kare hızı (4 düzensiz kare aralığı); sabit 25,00 fps'lik kopyadan analiz edildi "
                   "(c.mp4, net 2 kare eklendi)\n")  # the old warning is not repeated next to it
    _analyze(monkeypatch, ["v.mp4"], _report(irregular_frame_intervals=0, constant_rate_copy=copy))
    assert capsys.readouterr().err == "\n"  # nothing irregular: a copy on record says nothing
    _analyze(monkeypatch, ["v.mp4"], _report(constant_rate_copy=copy))
    assert capsys.readouterr().err == "\n"  # only the end of the progress line
    _analyze(monkeypatch, ["v.mp4"], _report(irregular_frame_intervals=0, constant_rate_copy=None))
    assert capsys.readouterr().err == "\n"  # only the end of the progress line
    _analyze(monkeypatch, ["v.mp4"], _report(irregular_frame_intervals=1))  # no copy key at all
    assert capsys.readouterr().err == ("\nuyarı: değişken kare hızı (1 düzensiz kare aralığı); hızlar ve işaretler "
                                       "kayabilir\n")


def test_the_summary_lists_every_flight_with_its_call_or_its_notes(monkeypatch, capsys) -> None:
    rep = _report()
    rep["calibration"]["line_rmse_px_median"] = None
    rep["flights"] = [
        {"id": 0, "hitter": "near", "receiver": "far", "notes": [],
         "call": {"verdict": "OUT", "p_in": 0.1234, "margin_m": -0.0456, "sigma_m": 0.031, "line": "far baseline"}},
        {"id": 1, "hitter": "far", "receiver": "near", "call": None, "notes": ["kısa", "gürültülü"]},
        {"id": 2, "hitter": "far", "receiver": "near", "notes": [],
         "call": {"verdict": "IN", "p_in": 1.0, "margin_m": 0.004, "sigma_m": 0.0004, "line": "net"}},
        {"id": 3, "hitter": "near", "receiver": "far", "notes": [],
         "call": {"verdict": "REVIEW", "p_in": 0.5, "margin_m": -0.004, "sigma_m": 0.02, "line": "x"}}]
    assert _analyze(monkeypatch, ["v.mp4", "--out", "o"], rep)[0] == 0
    assert capsys.readouterr().out.splitlines() == [
        "calibrated frames 3/5, line rmse None px",  # pinned: a missing median is printed as "None px"
        "  flight 0: near -> far  OUT    P(in)=0.123 margin -5±3 cm (far baseline)",
        "  flight 1:  far -> near no call: kısa; gürültülü",
        "  flight 2:  far -> near IN     P(in)=1.000 margin +0±0 cm (net)",
        "  flight 3: near -> far  REVIEW P(in)=0.500 margin -0±2 cm (x)",  # pinned: a "-0" margin is printed
        f"report: {Path('o') / 'report.json'}"]


@pytest.mark.parametrize("exc", [UnreadableVideoError("x.mp4: video okunamadı (moov atom not found)"),
                                 MissingModelError("model dosyası bulunamadı: ball: sekme_ball.pt")])
def test_an_unreadable_video_or_a_missing_model_is_one_line_and_exit_2(monkeypatch, capsys, exc) -> None:
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(pipeline, "analyze", boom)
    assert cli.main(["analyze", "x.mp4"]) == 2
    cap = capsys.readouterr()
    assert cap.err == f"hata: {exc}\n" and cap.out == ""


@pytest.mark.parametrize("exc", [RuntimeError("bug"), ValueError("not the video error"), FileNotFoundError("x"),
                                 KeyError("k")])
def test_any_other_error_is_not_swallowed(monkeypatch, exc) -> None:
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(pipeline, "analyze", boom)
    with pytest.raises(type(exc)):
        cli.main(["analyze", "x.mp4"])


def test_the_exit_code_is_the_commands_return_value(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_doctor", lambda args: 7)
    assert cli.main(["doctor"]) == 7


# --- 2. command line: argument parser -------------------------------------------------------------------------

def _namespace(monkeypatch, cmd: str, argv: list[str]) -> dict:
    got: dict = {}
    monkeypatch.setattr(cli, f"_{cmd}", lambda args: got.update(vars(args)) or 0)
    assert cli.main([cmd, *argv]) == 0
    got.pop("fn")
    return got


def test_parser_defaults_of_every_subcommand(monkeypatch) -> None:
    assert _namespace(monkeypatch, "analyze", ["v.mp4"]) == {
        "cmd": "analyze", "video": "v.mp4", "out": None, "device": "auto", "no_render": False, "no_broadcast": False,
        "reuse_perception": False, "ball_model": None, "ball_stride": 0, "toss_bridge": False,
        "pin_serve_height": False, "player_size_from_camera": False, "fps_scaled_thresholds": False,
        "fit_workers": None, "model": []}
    assert _namespace(monkeypatch, "serve", []) == {"cmd": "serve", "runs": "runs", "host": "127.0.0.1", "port": 8765,
                                                    "device": "auto"}
    assert _namespace(monkeypatch, "doctor", []) == {"cmd": "doctor", "runs": "runs"}
    assert _namespace(monkeypatch, "benchmark", []) == {
        "cmd": "benchmark", "fps": 30.0, "noise": 1.5, "distance": 28.0, "height": 11.0, "focal": 1750.0,
        "width": 1920, "shots": 40}


def test_parser_converts_values_to_their_types(monkeypatch) -> None:
    got = _namespace(monkeypatch, "analyze", ["v.mp4", "--ball-stride", "3", "--fit-workers", "2", "--model", "a=b",
                                              "--model", "c=d"])
    assert (got["ball_stride"], got["fit_workers"], got["model"]) == (3, 2, ["a=b", "c=d"])
    got = _namespace(monkeypatch, "serve", ["--port", "9000", "--runs", "r2", "--host", "0.0.0.0", "--device", "cpu"])
    assert (got["port"], got["runs"], got["host"], got["device"]) == (9000, "r2", "0.0.0.0", "cpu")
    got = _namespace(monkeypatch, "benchmark", ["--fps", "59.94", "--noise", "0", "--distance", "20", "--height", "9",
                                                "--focal", "1000", "--width", "1280", "--shots", "5"])
    assert got["fps"] == 59.94 and got["noise"] == 0.0 and isinstance(got["noise"], float)
    assert (got["distance"], got["height"], got["focal"], got["width"], got["shots"]) == (20.0, 9.0, 1000.0, 1280, 5)
    assert isinstance(got["width"], int) and isinstance(got["shots"], int)


@pytest.mark.parametrize("argv", [
    [], ["nope"], ["analyze"], ["analyze", "v.mp4", "--ball-stride", "x"], ["analyze", "v.mp4", "--fit-workers", "1.5"],
    ["analyze", "v.mp4", "--bogus"], ["analyze", "a.mp4", "b.mp4"], ["serve", "--port", "x"],
    ["benchmark", "--shots", "1.5"], ["benchmark", "--width", "1e3"], ["benchmark", "--fps", "fast"],
    ["doctor", "--host", "x"], ["--runs", "r", "doctor"]])
def test_a_bad_command_line_exits_with_2_and_runs_nothing(monkeypatch, capsys, argv: list[str]) -> None:
    ran: list = []
    for cmd in ("analyze", "serve", "doctor", "benchmark"):
        monkeypatch.setattr(cli, f"_{cmd}", lambda args: ran.append(args) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and not ran
    assert capsys.readouterr().err.startswith("usage: sekme")


# --- 3. command line: serve, doctor, benchmark ----------------------------------------------------------------

def _serve(monkeypatch, argv: list[str]) -> dict:
    seen: dict = {}
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kw: seen.update(uvicorn=(app, kw))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    import courtvision.api.app as app_mod

    def fake_create_app(runs, analyze_fn=None, allowed_hosts=None, **kw):
        seen.update(runs=runs, analyze_fn=analyze_fn, allowed_hosts=allowed_hosts, kw=kw)
        return "the-app"

    monkeypatch.setattr(app_mod, "create_app", fake_create_app)
    seen["code"] = cli.main(["serve", *argv])
    return seen


def test_serve_on_this_machine_keeps_the_host_check_and_stays_quiet(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "default_fit_workers", lambda: 3)
    seen = _serve(monkeypatch, [])
    assert seen["code"] == 0 and seen["runs"] == Path("runs") and seen["allowed_hosts"] is None
    assert seen["uvicorn"] == ("the-app", {"host": "127.0.0.1", "port": 8765, "log_level": "info"})
    assert capsys.readouterr().err == ""
    assert cli.os.environ["SEKME_FIT_WORKERS"] == "3"  # uploaded videos are fitted in parallel


@pytest.mark.parametrize("host", ["localhost", "::1", "127.0.0.1"])
def test_serve_treats_the_three_loopback_names_as_local(monkeypatch, capsys, host: str) -> None:
    seen = _serve(monkeypatch, ["--host", host, "--port", "1234", "--runs", "elsewhere"])
    assert seen["allowed_hosts"] is None and seen["runs"] == Path("elsewhere")
    assert seen["uvicorn"][1]["host"] == host and seen["uvicorn"][1]["port"] == 1234
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "LOCALHOST", "::"])
def test_serve_on_any_other_address_warns_and_answers_to_every_host_name(monkeypatch, capsys, host: str) -> None:
    seen = _serve(monkeypatch, ["--host", host])  # pinned: "LOCALHOST" is not recognised as local (case-sensitive)
    assert seen["allowed_hosts"] == ["*"] and seen["uvicorn"][1]["host"] == host
    assert capsys.readouterr().err == (
        f"uyarı: İnceleme Odası {host} adresinde, kimlik doğrulaması olmadan açılıyor; ağdaki herkes analizleri "
        "görebilir ve video yükleyebilir\n")


def test_serve_keeps_an_explicit_worker_setting_and_analyses_uploads_with_its_device(monkeypatch) -> None:
    monkeypatch.setenv("SEKME_FIT_WORKERS", "1")
    calls: list = []
    monkeypatch.setattr(pipeline, "analyze", lambda *a: calls.append(a))
    seen = _serve(monkeypatch, ["--device", "cpu"])
    assert cli.os.environ["SEKME_FIT_WORKERS"] == "1"
    progress = object()
    seen["analyze_fn"](Path("up.mp4"), Path("out"), progress)
    video, out, cfg, prog = calls[0]
    assert (video, out, prog) == (Path("up.mp4"), Path("out"), progress)
    assert cfg == pipeline.PipelineConfig(device="cpu")


def test_doctor_checks_the_runs_folder_it_is_given_and_returns_the_verdict(monkeypatch, capsys) -> None:
    seen: list = []

    def fake_checks(runs=None):
        seen.append(runs)
        return [doctor.Check("Python", True, True, "3.13.0"), doctor.Check("disk", False, False, "az")]

    monkeypatch.setattr(doctor, "run_checks", fake_checks)
    assert cli.main(["doctor"]) == 0 and cli.main(["doctor", "--runs", "other"]) == 0
    assert seen == [Path("runs"), Path("other")]
    out = capsys.readouterr().out
    assert out.count("SEKME ortam denetimi\n  ✓ Python: 3.13.0\n  ! disk: az\nSonuç: analize hazır (1 uyarı)\n") == 2


def _benchmark(monkeypatch, capsys, argv: list[str]):
    seen: dict = {}

    class Rep:
        def to_json(self) -> str:
            return '{"fake": true}'

    def fake_run(camera, shots, fps, noise):
        seen.update(camera=camera, shots=shots, fps=fps, noise=noise)
        return Rep()

    import courtvision.eval.sim_benchmark as sb

    monkeypatch.setattr(sb, "run_benchmark", fake_run)
    code = cli.main(["benchmark", *argv])
    return code, seen, capsys.readouterr().out


def test_benchmark_passes_its_numbers_on_and_prints_the_report_json(monkeypatch, capsys) -> None:
    code, seen, out = _benchmark(monkeypatch, capsys, ["--shots", "7", "--fps", "50", "--noise", "2.5"])
    assert code == 0 and out == '{"fake": true}\n'
    assert (seen["shots"], seen["fps"], seen["noise"]) == (7, 50.0, 2.5)
    _, seen, _ = _benchmark(monkeypatch, capsys, [])
    assert (seen["shots"], seen["fps"], seen["noise"]) == (40, 30.0, 1.5)


@pytest.mark.parametrize("argv,dist,height,focal,w,h", [
    ([], 28.0, 11.0, 1750.0, 1920, 1080), (["--width", "1000"], 28.0, 11.0, 1750.0, 1000, 562),
    (["--distance", "20", "--height", "5", "--focal", "900", "--width", "1280"], 20.0, 5.0, 900.0, 1280, 720)])
def test_benchmark_camera_sits_behind_the_net_and_looks_at_the_court(monkeypatch, capsys, argv, dist, height, focal,
                                                                     w, h) -> None:
    _, seen, _ = _benchmark(monkeypatch, capsys, argv)
    cam = seen["camera"]
    assert np.allclose(cam.center, [0.0, -dist, height])
    assert cam.focal == focal and cam.K[1, 1] == focal
    assert (cam.K[0, 2], cam.K[1, 2]) == (w / 2, h / 2)  # the picture is 16:9, whole pixels of height
    assert np.allclose(cam.R @ cam.R.T, np.eye(3)) and np.isclose(np.linalg.det(cam.R), 1.0)  # a proper rotation
    aim = cam.project(np.array([0.0, 1.5, 0.0]))
    assert np.allclose(aim, [w / 2, h / 2])  # the middle of the court is the middle of the picture
    assert cam.project(np.array([0.0, 1.5, 1.0]))[1] < aim[1]  # up is up in the picture
    assert cam.project(np.array([1.0, 1.5, 0.0]))[0] > aim[0]  # world +x is to the right
    assert cam.depth(np.array([0.0, 1.5, 0.0])) > 0


# --- 4. doctor ---------------------------------------------------------------------------------------------------

def _stdout(monkeypatch, which, run) -> list:
    calls: list = []
    monkeypatch.setattr(doctor.shutil, "which", which)

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return run(cmd, **kw)

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    return calls


def test_version_runs_the_binary_it_found_with_a_time_limit(monkeypatch) -> None:
    calls = _stdout(monkeypatch, lambda b: f"/opt/bin/{b}",
                    lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "ffmpeg version 6.1.1 Copyright (c)\n", ""))
    assert doctor._version("ffmpeg") == "6.1.1"
    assert calls == [(["/opt/bin/ffmpeg", "-version"], {"capture_output": True, "text": True, "timeout": 20})]


@pytest.mark.parametrize("binary,out,expect", [
    ("ffprobe", "ffprobe version n7.0-static https://x\nbuilt with gcc\n", "n7.0-static"),
    ("ffprobe", "ffmpeg version 6.1 Copyright\n", "?"),  # another program's banner
    ("ffmpeg", "ffmpeg version\n", "?"), ("ffmpeg", "ffmpeg version 7\n", "7"), ("ffmpeg", "", "?"),
    ("ffmpeg", "\nffmpeg version 6.1\n", "?"),  # only the first line is read
    ("ffmpeg", "usage: ffmpeg [options]\n", "?")])
def test_version_reads_the_third_word_of_the_first_line_or_says_unknown(monkeypatch, binary, out, expect) -> None:
    _stdout(monkeypatch, lambda b: "/x/" + b, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, out, ""))
    assert doctor._version(binary) == expect


@pytest.mark.parametrize("exc", [OSError("exec format error"), subprocess.TimeoutExpired("ffmpeg", 20)])
def test_version_of_a_binary_that_cannot_run_is_unknown_not_missing(monkeypatch, exc) -> None:
    def boom(cmd, **kw):
        raise exc

    _stdout(monkeypatch, lambda b: "/x/" + b, boom)
    assert doctor._version("ffmpeg") == "?"


def test_version_of_a_missing_binary_is_none_and_nothing_is_run(monkeypatch) -> None:
    calls = _stdout(monkeypatch, lambda b: None, lambda cmd, **kw: pytest.fail("must not run"))
    assert doctor._version("ffmpeg") is None and not calls


@pytest.mark.parametrize("versions,expect", [
    ({"ffmpeg": None, "ffprobe": None}, (False, "ffmpeg ve ffprobe bulunamadı (macOS: brew install ffmpeg)")),
    ({"ffmpeg": "6.1", "ffprobe": None}, (False, "ffprobe bulunamadı (macOS: brew install ffmpeg)")),
    ({"ffmpeg": None, "ffprobe": "6.1"}, (False, "ffmpeg bulunamadı (macOS: brew install ffmpeg)")),
    ({"ffmpeg": "6.1", "ffprobe": "6.0"}, (True, "ffmpeg 6.1, ffprobe 6.0")),
    ({"ffmpeg": "?", "ffprobe": "?"}, (True, "ffmpeg ?, ffprobe ?"))])  # found but unreadable is still found
def test_ffmpeg_check_names_what_is_missing(monkeypatch, versions, expect) -> None:
    monkeypatch.setattr(doctor, "_version", lambda b: versions[b])
    check = doctor._ffmpeg()
    assert (check.name, check.essential, check.ok, check.detail) == ("ffmpeg", True, *expect)


def _fake_cv2(backends: dict[int, str], version: str = "4.9.0") -> SimpleNamespace:
    reg = SimpleNamespace(getBackends=lambda: list(backends), getBackendName=lambda b: backends[b])
    return SimpleNamespace(__version__=version, videoio_registry=reg)


@pytest.mark.parametrize("backends,ok,detail", [
    ({1: "FFMPEG", 2: "CAP_IMAGES"}, True, "4.9.0, video okuma: FFMPEG"),
    ({1: "AVFOUNDATION", 2: "FFMPEG"}, True, "4.9.0, video okuma: AVFOUNDATION, FFMPEG"),
    ({1: "AVFOUNDATION"}, True, "4.9.0, video okuma: AVFOUNDATION"),
    ({1: "GSTREAMER", 2: "CAP_IMAGES"}, False, "4.9.0, video okuyamıyor (arka uçlar: ['CAP_IMAGES', 'GSTREAMER'])"),
    ({}, False, "4.9.0, video okuyamıyor (arka uçlar: [])")])
def test_opencv_check_needs_a_video_reading_backend(monkeypatch, backends, ok: bool, detail: str) -> None:
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(backends))
    check = doctor._opencv()
    assert (check.name, check.ok, check.essential, check.detail) == ("OpenCV", ok, True, detail)


def test_opencv_that_cannot_be_imported_is_an_essential_failure(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)  # makes "import cv2" raise ImportError
    check = doctor._opencv()
    assert (check.name, check.ok, check.essential) == ("OpenCV", False, True)
    assert check.detail.startswith("içe aktarılamadı (") and check.detail.endswith(")")


@pytest.mark.parametrize("device,ok,detail", [
    ("cpu", False, "9.9.9, hızlandırıcı yok: analiz yalnız işlemcide, çok daha yavaş"),
    ("mps", True, "9.9.9, hızlandırıcı: mps"), ("cuda", True, "9.9.9, hızlandırıcı: cuda")])
def test_torch_without_an_accelerator_only_warns(monkeypatch, device: str, ok: bool, detail: str) -> None:
    import torch

    import courtvision.perception.court as court

    asked: list = []
    monkeypatch.setattr(torch, "__version__", "9.9.9")
    monkeypatch.setattr(court, "pick_device", lambda pref: asked.append(pref) or SimpleNamespace(type=device))
    check = doctor._torch()
    assert (check.name, check.ok, check.essential, check.detail) == ("PyTorch", ok, False, detail)
    assert asked == ["auto"]  # the device the analysis would choose by itself


def test_torch_that_cannot_be_imported_is_an_essential_failure(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    check = doctor._torch()
    assert (check.name, check.ok, check.essential) == ("PyTorch", False, True)
    assert check.detail.startswith("içe aktarılamadı (")


@pytest.mark.parametrize("available,ok,detail", [
    (True, True, "skor grafiği ve hız tabelası okunur"),
    (False, False, "yok: skor grafiği, isimler ve hız tabelası okunmaz (yalnız macOS, pyobjc-framework-Vision)")])
def test_ocr_check_is_only_a_warning(monkeypatch, available: bool, ok: bool, detail: str) -> None:
    monkeypatch.setattr("courtvision.broadcast.ocr.available", lambda: available)
    check = doctor._ocr()
    assert (check.name, check.ok, check.essential, check.detail) == ("Apple Vision OCR", ok, False, detail)


def test_web_check_wants_the_built_index_file_itself(tmp_path: Path) -> None:
    name = "İnceleme Odası arayüzü"
    check = doctor._web(tmp_path)
    assert (check.name, check.ok, check.essential) == (name, False, False)
    assert check.detail == "derlenmemiş: cd web && npm ci && npm run build"
    (tmp_path / "web" / "dist").mkdir(parents=True)  # a folder without the page is not a build
    assert not doctor._web(tmp_path).ok
    (tmp_path / "web" / "dist" / "index.html").mkdir()  # a folder in the page's place is not one either
    assert not doctor._web(tmp_path).ok
    (tmp_path / "web" / "dist" / "index.html").rmdir()
    (tmp_path / "web" / "dist" / "index.html").write_text("<!doctype html>")
    check = doctor._web(tmp_path)
    assert (check.ok, check.essential, check.detail) == (True, False, "web/dist derlenmiş")
    assert not doctor._web(tmp_path / "web").ok  # the build is looked for under <root>/web/dist only


def _disk_usage(monkeypatch, free_bytes: float) -> list:
    probed: list = []

    def usage(path):
        probed.append(Path(path))
        return SimpleNamespace(free=free_bytes)

    monkeypatch.setattr(doctor.shutil, "disk_usage", usage)
    return probed


def test_disk_check_measures_an_existing_folder_or_else_its_parent(monkeypatch, tmp_path: Path) -> None:
    probed = _disk_usage(monkeypatch, 123.6e9)
    check = doctor._disk(tmp_path)
    assert (check.name, check.ok, check.essential) == ("disk", True, False)
    assert check.detail == f"{tmp_path}: 124 GB boş" and probed == [tmp_path]
    check = doctor._disk(tmp_path / "runs")  # not created yet: the folder it will be created in
    assert check.detail == f"{tmp_path}: 124 GB boş" and probed[-1] == tmp_path


@pytest.mark.parametrize("free,ok,detail", [
    (5.0e9, True, "5 GB boş"), (5.0e9 - 1, False, "5.0 GB boş, bir uzun analiz için az"),
    (4.96e9, False, "5.0 GB boş, bir uzun analiz için az"),  # pinned: "5.0 GB" is called too little (rounded)
    (0.0, False, "0.0 GB boş, bir uzun analiz için az"), (1.23e9, False, "1.2 GB boş, bir uzun analiz için az")])
def test_disk_check_warns_below_five_gigabytes(monkeypatch, tmp_path: Path, free: float, ok: bool, detail) -> None:
    _disk_usage(monkeypatch, free)
    check = doctor._disk(tmp_path)
    assert (check.ok, check.essential, check.detail) == (ok, False, f"{tmp_path}: {detail}")


def test_models_report_their_checksum_and_architecture(monkeypatch, tmp_path: Path) -> None:
    import courtvision.perception.ball as ball
    monkeypatch.setattr(pipeline, "MODEL_FILES", {"court": "m/court.pt", "ball": "m/ball.pt", "pose": "m/pose.pt"})
    monkeypatch.setattr(pipeline, "_sha256", lambda p: "0123456789abcdef" * 4)
    monkeypatch.setattr(ball, "ball_arch", lambda path: None)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "court.pt").write_bytes(b"x")
    (tmp_path / "m" / "ball.pt").write_bytes(b"x")
    (tmp_path / "m" / "pose.pt").mkdir()  # a folder is not a checkpoint
    got = doctor._models(tmp_path)
    assert [(c.name, c.ok, c.essential) for c in got] == [("ağ court", True, True), ("ağ ball", True, True),
                                                          ("ağ pose", False, True)]
    assert got[0].detail == "m/court.pt, sha256 0123456789ab"  # twelve characters; the architecture is the ball's
    assert got[1].detail == "m/ball.pt, sha256 0123456789ab, WASB"  # no known architecture name: WASB
    assert got[2].detail == "m/pose.pt yok (ağırlıklar repoda değil, docs/MODELS.md)"
    monkeypatch.setattr(ball, "ball_arch", lambda path: "unet3")
    assert doctor._models(tmp_path)[1].detail == "m/ball.pt, sha256 0123456789ab, unet3"


def test_run_checks_order_arguments_and_the_python_version(monkeypatch, tmp_path: Path) -> None:
    seen: list = []
    mk = doctor.Check
    monkeypatch.setattr(doctor, "_ffmpeg", lambda: mk("ffmpeg", True, True, "-"))
    monkeypatch.setattr(doctor, "_opencv", lambda: mk("OpenCV", True, True, "-"))
    monkeypatch.setattr(doctor, "_torch", lambda: mk("PyTorch", True, False, "-"))
    monkeypatch.setattr(doctor, "_models", lambda root: seen.append(("models", root)) or [mk("ağ x", True, True, "-")])
    monkeypatch.setattr(doctor, "_ocr", lambda: mk("OCR", True, False, "-"))
    monkeypatch.setattr(doctor, "_web", lambda root: seen.append(("web", root)) or mk("web", True, False, "-"))
    monkeypatch.setattr(doctor, "_disk", lambda path: seen.append(("disk", path)) or mk("disk", True, False, "-"))
    version = namedtuple("version", "major minor micro")
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(version_info=version(3, 11, 0)))
    checks = doctor.run_checks(root=tmp_path)
    assert [c.name for c in checks] == ["Python", "ffmpeg", "OpenCV", "PyTorch", "ağ x", "OCR", "web", "disk"]
    assert checks[0] == mk("Python", True, True, "3.11.0")  # the oldest supported version passes
    assert seen == [("models", tmp_path), ("web", tmp_path), ("disk", tmp_path / "runs")]
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(version_info=version(3, 10, 14)))
    seen.clear()
    checks = doctor.run_checks(root=tmp_path, runs=tmp_path / "elsewhere")
    assert checks[0] == mk("Python", False, True, "3.10.14")
    assert seen[-1] == ("disk", tmp_path / "elsewhere")


def test_report_counts_missing_essentials_and_warnings_separately() -> None:
    ok, mk = doctor.Check("ok", True, True, "a"), doctor.Check
    text, code = doctor.report([ok, mk("a", False, True, "x"), mk("b", False, True, "y"), mk("c", False, False, "z"),
                                mk("d", True, False, "fine")])
    assert code == 1
    assert text.splitlines() == ["SEKME ortam denetimi", "  ✓ ok: a", "  ✗ a: x", "  ✗ b: y", "  ! c: z",
                                 "  ✓ d: fine", "Sonuç: analiz çalışmaz (2 eksik, 1 uyarı)"]
    assert doctor.report([mk("a", False, True, "x")]) == (
        "SEKME ortam denetimi\n  ✗ a: x\nSonuç: analiz çalışmaz (1 eksik)", 1)
    text, code = doctor.report([mk("c", False, False, "z"), mk("e", False, False, "w")])
    assert code == 0 and text.endswith("Sonuç: analize hazır (2 uyarı)")
    assert doctor.report([]) == ("SEKME ortam denetimi\nSonuç: analize hazır", 0)


# --- 5. Review Room helpers ------------------------------------------------------------------------------------

@pytest.mark.parametrize("netloc,expect", [
    ("localhost:8765", "localhost"), ("LOCALHOST", "localhost"), ("LocalHost:1", "localhost"),
    ("127.0.0.1:80", "127.0.0.1"), ("[::1]:8765", "::1"), ("[::1]", "::1"),
    ("[0:0:0:0:0:0:0:1]", "0:0:0:0:0:0:0:1"),  # the same address, but not the name the allow list knows
    ("[::ffff:127.0.0.1]", "::ffff:127.0.0.1"), ("localhost.", "localhost."),  # a trailing dot is another name
    ("localhost@evil.example", "evil.example"), ("evil.example:80", "evil.example"),
    ("localhost.evil.example", "localhost.evil.example"), ("127.0.0.1.evil.example", "127.0.0.1.evil.example"),
    ("", None), ("::1", None), ("[::1", None), ("[::1]x", None), ("[zzz]", None)])
def test_host_name_takes_the_name_out_of_a_host_header(netloc: str, expect: str | None) -> None:
    assert host_name(netloc) == expect


@pytest.mark.parametrize("netloc,expect", [
    ("localhost:abc", "localhost"), ("localhost:99999", "localhost"), ("localhost:", "localhost"),
    ("localhost:-1", "localhost"),  # pinned: an impossible port does not make the name unparsable
    ("evil@localhost", "localhost"), ("user:pw@localhost:80", "localhost"),  # pinned: user info in a Host header
    ("localhost/x", "localhost"), ("localhost?x", "localhost"), ("localhost#x", "localhost"),  # pinned: path chars
    ("local\thost", "localhost"),  # pinned: a tab inside the name is dropped, so it matches the local name
    ("  localhost", "  localhost")])
def test_host_name_is_lenient_about_what_surrounds_the_name(netloc: str, expect: str) -> None:
    assert host_name(netloc) == expect


def test_host_header_decides_who_is_served(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "clip1"
    run.mkdir(parents=True)
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    for host in ("LOCALHOST:8765", "[::1]", "[::1]:8765", "127.0.0.1:1", "LocalHost"):
        assert c.get("/api/health", headers={"host": host}).status_code == 200, host
    for host in ("localhost.", "[0:0:0:0:0:0:0:1]", "::1", "[::1", "localhost.evil.example", "evil.example:8765"):
        assert c.get("/api/health", headers={"host": host}).status_code == 400, host
    # pinned: these pass the check although the name is not a plain local one (a browser never sends them)
    for host in ("localhost:abc", "evil@localhost", "localhost/x"):
        assert c.get("/api/health", headers={"host": host}).status_code == 200, host


def test_run_id_pattern_lengths_and_the_trailing_newline() -> None:
    assert RUN_ID.match("a" * 64) and not RUN_ID.match("a" * 65)
    assert RUN_ID.match("a.b-c_d") and RUN_ID.match("7")
    for bad in ("", ".a", "-a", "_a", "a/b", "a\\b", "a b", "a\x00", "é", "a:b"):
        assert not RUN_ID.match(bad), repr(bad)
    assert not RUN_ID.match("clip1\n") and not RUN_ID.match("clip1\n\n")  # "\Z", not "$": no trailing newline


def test_a_run_folder_whose_name_ends_in_a_newline_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "nl\n"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{"video": {}}')
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get("/api/runs/nl%0A/report").status_code == 400  # it used to be served (the pattern ended in "$")
    assert c.get("/api/runs/nl%0A%0A/report").status_code == 400
    assert c.get("/api/runs/clip1%0A/report").status_code == 400  # an id with a newline is invalid


def _video_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "runs" / "clip1"
    run.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    return run, outside


def test_media_path_source_takes_the_first_usable_name_in_sorted_order(tmp_path: Path) -> None:
    run, outside = _video_run(tmp_path)
    (outside / "v.mp4").write_bytes(b"video")
    (run / "source.mp4").symlink_to(tmp_path / "gone.mp4")  # dangling: skipped, the next one is tried
    (run / "source.mkv").write_bytes(b"v")
    (run / "source.bak").write_bytes(b"v")
    (run / "sourcemp4").write_bytes(b"v")
    (run / "source").write_bytes(b"v")
    assert media_path(run, "source") == run / "source.mkv"
    (run / "source.MOV").write_bytes(b"v")  # upper case names sort first and their suffix counts
    assert media_path(run, "source") == run / "source.MOV"
    (run / "source.MOV").unlink()
    (run / "source.mkv").unlink()
    assert media_path(run, "source") is None
    (run / "source.mp4").unlink()
    (run / "source.mp4").mkdir()  # a folder named like a video is no video
    assert media_path(run, "source") is None
    (run / "source.mp4").rmdir()
    (run / "source.tar.webm").write_bytes(b"v")
    assert media_path(run, "source") == run / "source.tar.webm"  # only the last suffix counts


def test_media_path_links_to_a_video_anywhere_are_followed_and_links_to_non_videos_are_not(tmp_path: Path) -> None:
    run, outside = _video_run(tmp_path)
    (outside / "clip.MP4").write_bytes(b"video")
    (outside / "notes.txt").write_text("secret")
    (outside / "clip.txt").write_bytes(b"video")
    (run / "source.mp4").symlink_to(outside / "notes.txt")  # the target must be a video name too
    assert media_path(run, "source") is None
    (run / "source.mp4").unlink()
    (run / "source.mp4").symlink_to(outside / "clip.txt")
    assert media_path(run, "source") is None
    (run / "source.mp4").unlink()
    (run / "source.mp4").symlink_to(outside / "clip.MP4")
    # pinned: any video file outside the runs folder can be linked in and is then served (by design for the
    # source video, which analyze links to; nothing checks that it lies under the runs folder)
    assert media_path(run, "source") == run / "source.mp4"


def test_media_path_classic_and_pro_are_the_fixed_names_and_must_be_files(tmp_path: Path) -> None:
    run, outside = _video_run(tmp_path)
    assert media_path(run, "classic") is None and media_path(run, "pro") is None
    (run / "classic.mp4").write_bytes(b"c")
    (run / "pro.mp4").mkdir()
    assert media_path(run, "classic") == run / "classic.mp4"
    assert media_path(run, "pro") is None  # a folder is not a file
    assert media_path(run, "classic") != media_path(run, "pro")
    for bad in ("", "../x", "Pro", "pro.mp4", "SOURCE"):
        with pytest.raises(KeyError):  # callers (the media route) check the name against the allow list first
            media_path(run, bad)


def test_a_classic_video_that_is_a_link_is_refused_not_served(tmp_path: Path) -> None:
    # A link in place of one of the run's own files would let the API send any readable file (it used to).
    run, outside = _video_run(tmp_path)
    (run / "report.json").write_text(json.dumps({"video": {"file": "a", "duration_s": 1}, "flights": []}))
    (run / "frames.json").write_text("{}")
    (outside / "secret.txt").write_text("SECRET")
    (run / "classic.mp4").symlink_to(outside / "secret.txt")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    r = c.get("/api/runs/clip1/media/classic")
    assert r.status_code == 403 and "SECRET" not in r.text
    assert c.get("/api/runs/clip1/report").status_code == 403  # the whole run is refused
    assert [x["id"] for x in c.get("/api/runs").json()] == []  # and not listed
    (run / "classic.mp4").unlink()
    (run / "classic.mp4").write_bytes(b"c")  # a real file is fine
    assert c.get("/api/runs/clip1/media/classic").status_code == 200
    assert [x["id"] for x in c.get("/api/runs").json()] == ["clip1"]


def test_report_and_log_links_leaving_the_runs_folder_are_refused(tmp_path: Path) -> None:
    run, outside = _video_run(tmp_path)
    call = {"verdict": "IN", "p_in": 0.9, "margin_m": 0.1, "sigma_m": 0.1, "line": "net"}
    (outside / "r.json").write_text(json.dumps({"video": {}, "flights": [{"id": 1, "call": call}], "secret": "S"}))
    (outside / "log.jsonl").write_text('{"x": 1}\n')
    (run / "report.json").symlink_to(outside / "r.json")
    c0 = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c0.get("/api/runs/clip1/report").status_code == 403  # the report alone as a link is refused
    assert "S" not in c0.get("/api/runs/clip1/report").text
    (run / "reviews.jsonl").symlink_to(outside / "log.jsonl")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    # Both used to be followed: the report link read any JSON file, the log link made the review POST append outside.
    assert c.get("/api/runs/clip1/report").status_code == 403
    assert c.get("/api/runs/clip1/reviews").status_code == 403
    r = c.post("/api/runs/clip1/reviews", json={"flight_id": 1, "decision": "confirm", "reviewer": "r"})
    assert r.status_code == 403
    assert (outside / "log.jsonl").read_text().splitlines() == ['{"x": 1}']  # nothing appended outside
    (run / "report.json").unlink()
    (run / "report.json").write_text((outside / "r.json").read_text())
    assert c.get("/api/runs/clip1/report").status_code == 403  # one linked file is enough to refuse the run
    for name in ("frames.json", "labels.jsonl", "serve_labels.jsonl", "pro.mp4"):
        (run / "reviews.jsonl").unlink(missing_ok=True)
        for other in ("frames.json", "labels.jsonl", "serve_labels.jsonl", "pro.mp4"):
            (run / other).unlink(missing_ok=True)
        (run / name).symlink_to(outside / "log.jsonl")
        assert c.get("/api/runs/clip1/report").status_code == 403, name
    (run / "pro.mp4").unlink()
    (run / "source.mp4").symlink_to(outside / "clip.mp4")  # the source video may lie elsewhere (by design)
    (outside / "clip.mp4").write_bytes(b"v")
    assert c.get("/api/runs/clip1/report").status_code == 200


def _jsonl(tmp_path: Path, raw: bytes) -> list:
    p = tmp_path / "log.jsonl"
    p.write_bytes(raw)
    return read_jsonl(p)


def test_read_jsonl_of_a_missing_file_or_a_folder_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "nope.jsonl") == []
    assert read_jsonl(tmp_path) == []
    assert _jsonl(tmp_path, b"") == [] and _jsonl(tmp_path, b"\n\n") == []


def test_read_jsonl_skips_blank_lines_and_a_torn_last_line_but_keeps_the_order(tmp_path: Path) -> None:
    assert _jsonl(tmp_path, b'{"a": 1}\n\n   \n\t\n{"b": 2}\n') == [{"a": 1}, {"b": 2}]
    assert _jsonl(tmp_path, b'{"a": 1}\n{"b": 2}\n{"c": ') == [{"a": 1}, {"b": 2}]
    assert _jsonl(tmp_path, b'{"a": 1}') == [{"a": 1}]  # the last line needs no newline
    assert _jsonl(tmp_path, b'{"a": 1}\r\n{"b": 2}\r\n') == [{"a": 1}, {"b": 2}]
    assert _jsonl(tmp_path, b'{"a": 1}\n{"b" 2}\n{"c": 3}\n') == [{"a": 1}, {"c": 3}]  # a bad line in the middle too
    assert _jsonl(tmp_path, b'{"a": 1}{"b": 2}\n') == []  # two records glued on one line are lost together
    assert _jsonl(tmp_path, '{"name": "çay"}\n'.encode()) == [{"name": "çay"}]


def test_read_jsonl_ignores_a_byte_order_mark_at_the_start(tmp_path: Path) -> None:
    # A log saved by an editor that adds a BOM used to lose its first record.
    assert _jsonl(tmp_path, b'\xef\xbb\xbf{"a": 1}\n{"b": 2}\n') == [{"a": 1}, {"b": 2}]
    assert _jsonl(tmp_path, b'\xef\xbb\xbf{"a": 1}\n') == [{"a": 1}]
    assert _jsonl(tmp_path, b'{"a": 1}\n\xef\xbb\xbf{"b": 2}\n') == [{"a": 1}]  # only at the start of the file


def test_read_jsonl_splits_only_on_newlines_so_unicode_separators_stay_inside_a_record(tmp_path: Path) -> None:
    # str.splitlines also cuts at U+2028 and U+0085, which JSON allows raw inside a string: such a record used
    # to be lost. Form feed and vertical tab are control characters, which JSON does not allow raw in a string.
    for sep in ("\u2028", "\u0085"):
        assert _jsonl(tmp_path, f'{{"note": "a{sep}b"}}\n{{"id": 2}}\n'.encode()) == [{"note": f"a{sep}b"}, {"id": 2}]
    assert _jsonl(tmp_path, b'{"note": "a\\u2028b"}\n') == [{"note": "a\u2028b"}]  # escaped: kept


def test_read_jsonl_returns_records_only(tmp_path: Path) -> None:
    # Any JSON value used to be returned, so a stray number or list reached callers that expect records.
    assert _jsonl(tmp_path, b'[1, 2]\n"s"\n3\nnull\n{"a": 1}\n') == [{"a": 1}]
    assert _jsonl(tmp_path, b'{"a": 1}\n12') == [{"a": 1}]  # a torn line can be a valid number


def test_read_jsonl_skips_a_line_with_invalid_utf8_and_keeps_the_rest(tmp_path: Path) -> None:
    # One bad byte used to make the whole log unreadable.
    assert _jsonl(tmp_path, b'{"a": 1}\n\xff\xfe\n{"b": 2}\n') == [{"a": 1}, {"b": 2}]
    assert _jsonl(tmp_path, b'{"a": 1}\n{"a": "\xc3') == [{"a": 1}]  # a character cut in half by a crash


def test_a_log_with_invalid_utf8_is_still_read_and_takes_new_serve_labels(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "clip1"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"video": {"file": "a", "duration_s": 1}, "flights": []}))
    (run / "frames.json").write_text(json.dumps({"fps": 30, "n": 100}))
    (run / "reviews.jsonl").write_bytes(b'{"a": 1}\n\xff\n')
    (run / "serve_labels.jsonl").write_bytes(b"\xff\n")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL,
                   raise_server_exceptions=False)
    assert c.get("/api/runs/clip1/reviews").json() == [{"a": 1}]  # it used to be a 500
    assert c.post("/api/runs/clip1/serve-labels",
                  json={"type": "serve", "frame": 3, "labeler": "x"}).status_code == 201  # it used to be blocked
    assert (run / "serve_labels.jsonl").read_bytes().startswith(b"\xff\n{")


def test_a_torn_log_still_serves_the_records_before_it_and_a_new_one_lands_on_a_fresh_line(
        tmp_path: Path) -> None:
    run = tmp_path / "runs" / "clip1"
    run.mkdir(parents=True)
    call = {"verdict": "IN", "p_in": 0.9, "margin_m": 0.1, "sigma_m": 0.1, "line": "net"}
    (run / "report.json").write_text(json.dumps({"video": {}, "flights": [{"id": 1, "call": call}]}))
    (run / "reviews.jsonl").write_text('{"id": "a"}\n{"id": "b", "tr')  # cut mid-append, no newline
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get("/api/runs/clip1/reviews").json() == [{"id": "a"}]
    r = c.post("/api/runs/clip1/reviews", json={"flight_id": 1, "decision": "confirm", "reviewer": "r"})
    assert r.status_code == 201
    # The new record used to be glued to the torn fragment and lost with it; a newline now ends the fragment first.
    assert [x["id"] for x in c.get("/api/runs/clip1/reviews").json()] == ["a", r.json()["id"]]
    lines = (run / "reviews.jsonl").read_text().split("\n")
    assert lines[1] == '{"id": "b", "tr' and lines[-1] == ""  # the fragment stays as it was, on its own line
    r2 = c.post("/api/runs/clip1/reviews", json={"flight_id": 1, "decision": "overturn_out", "reviewer": "r"})
    assert [x["id"] for x in c.get("/api/runs/clip1/reviews").json()] == ["a", r.json()["id"], r2.json()["id"]]
    assert (run / "reviews.jsonl").read_text().count("\n\n") == 0  # a log that ends a line gets no blank line


# --- 6. i18n.num ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,digits,expect", [
    (12.34, 1, "12,3"), (12.34, 0, "12"), (12.345, 3, "12,345"), (5, 1, "5,0"), (0.0, 2, "0,00"),
    (1234567.891, 1, "1234567,9"),  # no thousands separator
    (0.25, 1, "0,2"), (0.35, 1, "0,3"), (2.5, 0, "2"), (3.5, 0, "4"),  # the float's own rounding (half to even)
    (9.95, 1, "9,9"), (0.995, 2, "0,99"), (0.999, 2, "1,00"), (1e-7, 3, "0,000"),
    (-1.25, 1, "-1,2"), (-2.5, 0, "-2"), (-12.34, 1, "-12,3"),
    (np.float64(1.25), 1, "1,2"), (np.float32(0.1), 3, "0,100"), (True, 1, "1,0")])
def test_num_uses_a_decimal_comma_and_the_floats_own_rounding(value, digits: int, expect: str) -> None:
    assert num(value, digits) == expect


def test_num_default_is_one_decimal_and_never_shows_a_dot() -> None:
    assert num(2.0) == "2,0" and num(-0.5) == "-0,5"
    for v in (0.1, 99.99, -3.14159, 1e9 + 0.5, 1e21):
        assert "." not in num(v) and "," in num(v)
    assert "," not in num(7.3, 0)  # no decimals: no comma either


@pytest.mark.parametrize("value,expect", [(-0.04, "-0,0"), (-0.0, "-0,0")])
def test_num_keeps_the_sign_of_a_negative_that_rounds_to_zero(value: float, expect: str) -> None:
    assert num(value) == expect  # pinned: "-0,0" is shown for a tiny negative number


@pytest.mark.parametrize("value,digits,expect", [(float("nan"), 1, "nan"), (float("inf"), 1, "inf"),
                                                 (float("-inf"), 2, "-inf")])
def test_num_passes_non_finite_numbers_through_in_english(value: float, digits: int, expect: str) -> None:
    assert num(value, digits) == expect  # pinned: not a Turkish word and not marked as missing


@pytest.mark.parametrize("bad", [None, "1.5", [1.0]])
def test_num_refuses_what_is_not_a_number(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        num(bad)
