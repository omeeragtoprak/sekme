"""The E2E-Spot runner's safeguards (scripts/eval_spot.py): pinned samples, the candidate ledger, failures that
stop a run, and the checks on every run before scoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_spot as E  # noqa: E402, N812


def _clips(match: str, n: int, fps: float = 25.0) -> list[dict]:
    return [{"video": f"{match}_{1000 * k}_{1000 * k + 50}", "fps": fps, "num_frames": 50,
             "events": [{"label": "near_court_swing", "frame": 10, "comment": ""}]} for k in range(n)]


def test_samples_are_the_plan_s_matches_with_their_clip_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(E, "DATA", tmp_path)
    (tmp_path / "tennis").mkdir()
    clips = [c for m, n in E.SAMPLES["test4"]["matches"].items() for c in _clips(m, n)]
    (tmp_path / "tennis" / "test.json").write_text(json.dumps(clips + _clips("other_match", 3)))
    path = E.make_sample("test4")
    assert len(json.loads(path.read_text())["clips"]) == 303  # 64 + 70 + 91 + 78, the other match left out
    assert E.make_sample("test4") == path  # an equal sample is kept
    (tmp_path / "tennis" / "test.json").write_text(json.dumps(clips[1:]))
    with pytest.raises(SystemExit, match="clip counts"):
        E.make_sample("test4")
    path.write_text("{}")
    (tmp_path / "tennis" / "test.json").write_text(json.dumps(clips))
    with pytest.raises(SystemExit, match="not overwritten"):
        E.make_sample("test4")


def test_the_ledger_allows_two_committed_candidates_with_known_flags(tmp_path: Path, monkeypatch) -> None:
    led = tmp_path / "ledger.json"
    monkeypatch.setattr(E, "LEDGER", led)
    assert E.ledger() == []
    led.write_text(json.dumps({"candidates": [{"slot": 1, "flags": ["toss-bridge"]}]}))
    with pytest.raises(SystemExit, match="uncommitted"):  # outside the repo: not committed
        E.ledger()
    monkeypatch.setattr(E, "tracked_and_clean", lambda files: True)
    assert E.candidate(1) == ["toss-bridge"]
    with pytest.raises(SystemExit, match="not in the committed ledger"):
        E.candidate(2)
    three = [{"slot": i, "flags": ["toss-bridge"]} for i in (1, 2, 3)]
    led.write_text(json.dumps({"candidates": three}))
    with pytest.raises(SystemExit, match="allow 2"):
        E.ledger()
    led.write_text(json.dumps({"candidates": [{"slot": 1, "flags": ["toss-brigde"]}]}))
    with pytest.raises(SystemExit, match="unknown candidate flag"):
        E.ledger()


def test_a_clip_failure_is_recorded_but_an_environment_failure_stops(tmp_path: Path, monkeypatch) -> None:
    real = subprocess.run

    def fake(rc: int):  # only the analysis is faked; git (the commit stamp) runs for real
        def run(cmd, *a, **k):
            if "courtvision.cli" in cmd:
                return subprocess.CompletedProcess(cmd, rc, "", "Traceback\nValueError: boom")
            return real(cmd, *a, **k)
        return run

    monkeypatch.setattr(E.subprocess, "run", fake(1))
    E.analyse(tmp_path / "c.mp4", tmp_path / "r1")
    assert (tmp_path / "r1" / "FAILED.txt").exists() and (tmp_path / "r1" / "code_sha.txt").exists()
    for rc in (2, -9):  # a usage error, a killed process
        monkeypatch.setattr(E.subprocess, "run", fake(rc))
        with pytest.raises(SystemExit, match="the run stops"):
            E.analyse(tmp_path / "c.mp4", tmp_path / f"r{rc}")
        assert not (tmp_path / f"r{rc}" / "FAILED.txt").exists()


def test_every_run_is_checked_before_scoring(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(E, "VIDEO_MANIFEST", tmp_path / "man.json")
    clips = _clips("m", 3)
    sample = {"matches": ["m"], "clips": clips}
    runs = tmp_path / "runs"
    head = E.head_sha()
    for c in clips:
        d = runs / c["video"]
        d.mkdir(parents=True)
        (d / "code_sha.txt").write_text(head)
        (d / "report.json").write_text(json.dumps({"candidates": {}, "video": {"n_frames": 50},
                                                   "models": {"ball": {"sha256": "b"}}}))
    problems, models = E.check_runs(sample, runs, [])
    assert problems == ["m: not in man.json"] and models == {"ball": "b"}
    man = {"m": {"width": 1280, "height": 720, "fps": 25.0, "frames": 9000, "files": {"v": ["runs"]}}}
    (tmp_path / "man.json").write_text(json.dumps(man))
    assert E.check_runs(sample, runs, [])[0] == []
    assert E.check_runs(sample, runs, [], {"ball": "other"})[0] == ["runs: models are not the committed defaults"]
    bad = runs / clips[1]["video"]
    (bad / "report.json").write_text(json.dumps({"candidates": {"serve_toss_bridge_s": 0.3},
                                                 "video": {"n_frames": 49}, "models": {"ball": {"sha256": "x"}}}))
    (runs / clips[2]["video"] / "code_sha.txt").unlink()
    problems = E.check_runs(sample, runs, [])[0]
    assert any("made with ['serve_toss_bridge_s']" in p for p in problems)
    assert any("49 frames" in p for p in problems) and any("other models" in p for p in problems)
    assert any(clips[2]["video"] in p and "other code" in p for p in problems)


def test_the_test_matches_take_no_free_runs_dirs_or_flags(monkeypatch) -> None:
    for argv, msg in ((["score", "--name", "test4", "--runs-dir", "x"], "fixed"),
                      (["run", "--name", "test4", "--candidate-flag", "toss-bridge"], "fixed"),
                      (["run", "--name", "dev3", "--slot", "1"], "for the test matches"),
                      (["run", "--name", "dev3", "--candidate-flag", "toss-brigde"], "unknown candidate flag")):
        monkeypatch.setattr(sys, "argv", ["eval_spot.py", *argv])
        with pytest.raises(SystemExit, match=msg):
            E.main()


def _video(path: Path, n: int, size: str = "1280x720", crf: str = "30") -> None:
    src = f"testsrc=size={size}:rate=25:duration={n / 25}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-c:v", "libx264", "-crf", crf,
                    "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_re_encoded_download_of_the_same_upload_is_accepted_and_another_upload_is_not(tmp_path: Path,
                                                                                       monkeypatch) -> None:
    monkeypatch.setattr(E, "VIDEO_MANIFEST", tmp_path / "man.json")
    a, b, other, small = (tmp_path / f"{x}.mp4" for x in ("a", "b", "other", "small"))
    _video(a, 50)
    _video(b, 50, crf="20")  # the same frames, encoded again: another file
    _video(other, 60)
    _video(small, 50, size="640x360")
    E.record_video("m", a, ["runs_spot"])
    E.record_video("m", b, ["runs_spot_base_slot1", "runs_spot_cand_slot1"])
    files = json.loads((tmp_path / "man.json").read_text())["m"]["files"]
    assert sorted(map(sorted, files.values())) == [["runs_spot"], ["runs_spot_base_slot1", "runs_spot_cand_slot1"]]
    with pytest.raises(SystemExit, match="another upload"):
        E.record_video("m", other, ["runs_spot"])
    with pytest.raises(SystemExit, match="720p"):
        E.record_video("n", small, ["runs_spot"])


def test_a_failed_baseline_fails_its_candidate_and_a_mid_run_code_change_records_nothing(tmp_path: Path,
                                                                                         monkeypatch) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "perception.pkl").write_bytes(b"cache")
    (base / "FAILED.txt").write_text("boom")  # failed after its cache was written
    monkeypatch.setattr(E, "analyse", lambda *a, **k: pytest.fail("the candidate must not run"))
    E.analyse_candidate(tmp_path / "c.mp4", base, tmp_path / "cand", ["toss-bridge"])
    assert (tmp_path / "cand" / "FAILED.txt").exists() and (tmp_path / "cand" / "code_sha.txt").exists()
    monkeypatch.undo()
    real = subprocess.run

    def ok(cmd, *a, **k):
        if "courtvision.cli" in cmd:
            run = Path(cmd[cmd.index("--out") + 1])
            run.mkdir(parents=True, exist_ok=True)
            (run / "report.json").write_text("{}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real(cmd, *a, **k)

    monkeypatch.setattr(E.subprocess, "run", ok)
    monkeypatch.setattr(E, "_CODE", ["src"])
    monkeypatch.setattr(E, "code_unchanged_since", lambda sha, code: False)
    with pytest.raises(SystemExit, match="changed during the run"):
        E.analyse(tmp_path / "c.mp4", tmp_path / "r")
    assert not (tmp_path / "r" / "report.json").exists() and not (tmp_path / "r" / "code_sha.txt").exists()


def test_a_hand_edited_sample_and_a_spent_test_pass_are_refused(tmp_path: Path, monkeypatch) -> None:
    sample = {"matches": sorted(E.SAMPLES["test4"]["matches"]),
              "clips": [c for m, n in E.SAMPLES["test4"]["matches"].items() for c in _clips(m, n)]}
    E.check_sample("test4", sample)
    sample["clips"] = sample["clips"][:-1]
    with pytest.raises(SystemExit, match="not the plan's sample"):
        E.check_sample("test4", sample)
    monkeypatch.setattr(E, "TEST_RESULTS", tmp_path)
    assert E.used(E.test_result(1)) is False
    E.test_result(1).write_text("{}")
    assert E.used(E.test_result(1)) is True
    assert E.used(Path(E.ROOT / "docs" / "EVAL_SPOT.md")) is True  # in the history


def test_a_refused_test_score_shows_no_number(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(E, "DATA", tmp_path)
    monkeypatch.setattr(E, "TEST_RESULTS", tmp_path / "results")
    monkeypatch.setattr(E, "VIDEO_MANIFEST", tmp_path / "man.json")
    monkeypatch.setattr(E, "clean", lambda paths: True)
    monkeypatch.setattr(E, "default_models", lambda: {})
    clips = [c for m, n in E.SAMPLES["test4"]["matches"].items() for c in _clips(m, n)]
    (tmp_path / "sample_test4.json").write_text(json.dumps({"matches": sorted(E.SAMPLES["test4"]["matches"]),
                                                           "clips": clips}))
    head = E.head_sha()
    for i, c in enumerate(clips):
        d = tmp_path / "runs_spot" / c["video"]
        d.mkdir(parents=True)
        (d / "code_sha.txt").write_text(head)
        flags = {"serve_toss_bridge_s": 0.3} if i == 5 else {}  # one clip made with a candidate's flag
        (d / "report.json").write_text(json.dumps({"candidates": flags, "video": {"n_frames": 50}, "models": {},
                                                   "hits": [{"frame": 10, "side": "near", "serve": False}],
                                                   "flights": [], "calibration": {"valid_frames": 50}}))
    monkeypatch.setattr(sys, "argv", ["eval_spot.py", "score", "--name", "test4"])
    with pytest.raises(SystemExit, match="made with"):
        E.main()
    assert "recall" not in capsys.readouterr().out and not (tmp_path / "results").exists()
