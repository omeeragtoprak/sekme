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
    assert problems == ["m: no SHA-256 in man.json"] and models == {"ball": "b"}
    (tmp_path / "man.json").write_text(json.dumps({"m": {"sha256": "v", "height": 720}}))
    assert E.check_runs(sample, runs, [])[0] == []
    bad = runs / clips[1]["video"]
    (bad / "report.json").write_text(json.dumps({"candidates": {"serve_toss_bridge_s": 0.3},
                                                 "video": {"n_frames": 49}, "models": {"ball": {"sha256": "x"}}}))
    (runs / clips[2]["video"] / "code_sha.txt").unlink()
    problems = E.check_runs(sample, runs, [])[0]
    assert any("made with ['serve_toss_bridge_s']" in p for p in problems)
    assert any("49 frames" in p for p in problems) and any("other models" in p for p in problems)
    assert any(clips[2]["video"] in p and "other code" in p for p in problems)


def test_the_test_matches_take_no_free_runs_dirs_or_flags(monkeypatch) -> None:
    for argv in (["score", "--name", "test4", "--runs-dir", "x"], ["run", "--name", "test4", "--candidate-flag",
                                                                    "toss-bridge"], ["run", "--name", "dev3",
                                                                                     "--slot", "1"]):
        monkeypatch.setattr(sys, "argv", ["eval_spot.py", *argv])
        with pytest.raises(SystemExit):
            E.main()
