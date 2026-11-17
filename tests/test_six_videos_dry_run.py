"""Dry run of ``score()`` in scripts/eval_six_videos.py on synthetic runs and labels (no video, model, GPU, network).

The data is built so that every verdict is known beforehand:

* fps rule (default vs scaled): the default run finds 1 of 4 labelled serves per video, the scaled one all 4
  (recall 25 -> 100), same contact accuracy, same calls: accepted.
* pin rule (scaled vs scaled_pin): 24 labelled serves with a board reading, all matched in scaled (error 12 %),
  pinned error 3 %; one pinned serve is rejected (it is "lost"); no verdict changes: accepted. The default run has
  one serve per video only, so a pin judged on default vs scaled_pin would be "inconclusive" (6 < 15 paired).
* player rule (default vs players): the default run tracks the near player only in the first half of the
  frames, the candidate throughout: accepted. Swapped, it would be rejected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from courtvision.eval.six_videos import disagreement_sample, presence_sample


def _script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_six_videos

    return eval_six_videos


FPS = 59.94
N_FRAMES = 2000
HEIGHT = 1080
N_SERVES = 4
BOARD_MPH = 120.0
BOARD_KMH = BOARD_MPH * 1.609344
ON_NEAR = [480.0, 700.0, 520.0, 900.0]  # feet (500, 900), 200 px tall
ON_FAR = [945.0, 130.0, 975.0, 200.0]  # feet (960, 200), 70 px tall


def _flight(k: int, kmh: float, rejected: str | None = None) -> dict:
    return {"id": k, "kind": "serve", "hitter": "near" if k % 2 == 0 else "far", "start_frame": 1000 * k + 99,
            "contact_frame": 1000 * k + 100.5, "hit_speed_kmh": kmh, "rejected": rejected,
            "board_speed_kmh": BOARD_KMH, "call": {"verdict": "IN"}, "bounce": {"frame": 1000 * k + 130}}


def _flights(n: int, err: float, lost: tuple[int, ...] = ()) -> list[dict]:
    return [_flight(k, BOARD_KMH * (1 + err / 100), "low contact" if k in lost else None) for k in range(n)]


N_RALLY = 30  # rally flights per video, for the player-names candidate (serve rules look at serves only)


def _rallies(e, n: str, fixed: bool) -> list[dict]:
    """Flights 1..N_RALLY: the near player is the video's first player on even ids. The default run names every
    third one wrongly; the candidate shows no name there instead (``fixed``)."""
    a, b = e.players_of(n)
    out = []
    for k in range(1, N_RALLY + 1):
        near = a if k % 2 == 0 else b
        shown = None if (fixed and k % 3 == 0) else (b if near == a else a) if k % 3 == 0 else near
        hitter = "near" if k % 2 == 0 else "far"
        out.append({"id": k, "kind": "rally", "hitter": hitter, "start_frame": 5000 + 40 * k,
                    "end_frame": 5030 + 40 * k, "call": {"verdict": "IN"}, "bounce": {"frame": 5020 + 40 * k},
                    "hitter_name": shown if hitter == "near" else None,
                    "receiver_name": shown if hitter == "far" else None})
    return out


def _report(kind: str, e, ball_sha: str, vsha: str, flights: list[dict], detected: int) -> dict:
    return {"video": {"fps": FPS, "n_frames": N_FRAMES, "height": HEIGHT, "sha256": vsha},
            "models": {"ball": {"sha256": ball_sha}, "court": {"sha256": "court-sha"}},
            "candidates": {f: True for f in e.FLAGS[kind]}, "flights": flights, "ball": {"detected_frames": detected}}


def _frames(near_until: int) -> dict:
    return {"calib_valid": [True] * N_FRAMES,
            "players": {"near": {"box": [ON_NEAR if i < near_until else None for i in range(N_FRAMES)]},
                        "far": {"box": [ON_FAR] * N_FRAMES}}}


def _build(tmp_path: Path, e, monkeypatch, with_players: bool) -> tuple[Path, Path]:
    labels_dir, runs = tmp_path / "labels", tmp_path / "runs"
    labels_dir.mkdir()
    ball = tmp_path / "sekme_ball.pt"
    ball.write_bytes(b"synthetic ball weights")
    ball_sha = hashlib.sha256(ball.read_bytes()).hexdigest()
    manifest = {n: {"sha256": f"video-{n}"} for n in e.NAMES}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    serve_labels = [{"lo": 1000 * k + 100, "hi": 1000 * k + 101, "side": "near" if k % 2 == 0 else "far",
                     "radar": {"value": BOARD_MPH, "unit": "mph"}} for k in range(N_SERVES)]
    flights = {"default": _flights(1, 12.0), "scaled": _flights(N_SERVES, 12.0, ()),
               "scaled_pin": _flights(N_SERVES, 3.0, (3,)), "players": _flights(1, 12.0),
               "bridge": _flights(N_SERVES, 12.0), "bridge_pin": _flights(N_SERVES, 3.0),
               "consistency": _flights(1, 12.0)}
    for vi, n in enumerate(e.NAMES):
        for kind in e.KINDS:
            d = runs / n / kind
            d.mkdir(parents=True)
            fl = flights[kind] if not (kind == "scaled_pin" and vi != 0) else _flights(N_SERVES, 3.0)
            if kind in ("default", "consistency"):  # the same flights, the names may differ
                fl = fl + _rallies(e, n, fixed=kind == "consistency")
            (d / "report.json").write_text(json.dumps(
                _report(kind, e, ball_sha, f"video-{n}", fl, 1000 if kind == "default" else 1040)))
        fa, fb = _frames(N_FRAMES // 2), _frames(N_FRAMES)
        (runs / n / "default" / "frames.json").write_text(json.dumps(fa))
        (runs / n / "players" / "frames.json").write_text(json.dumps(fb))
        (labels_dir / f"{n}.json").write_text(json.dumps({"serves": serve_labels}))
        if not with_players:
            continue
        ids = [0, *range(1, N_RALLY + 1)]
        drawn = e.names_sample(n, ids)
        a, b = e.players_of(n)
        (labels_dir / f"{n}_names_flights.json").write_text(json.dumps({"n_flights": len(ids), "flights": drawn}))
        ties = [{"frame": 1000 * k + 100, "server": (a if k % 2 == 0 else b).lower()} for k in range(N_SERVES)]
        (labels_dir / f"{n}_names.json").write_text(json.dumps({"ties": ties, "flights": [
            {"flight": i, "near": None if i == 0 else a if i % 2 == 0 else b.title()} for i in drawn]}))  # 0: unclear
        pres = presence_sample(n, N_FRAMES)  # the drawn lists, not made-up ones
        dis = disagreement_sample(n, fa, fb, fa["calib_valid"], HEIGHT)
        assert len(pres) == 80 and len(dis) == 60
        (labels_dir / f"{n}_presence_frames.json").write_text(json.dumps({"n_frames": N_FRAMES, "frames": pres}))
        (labels_dir / f"{n}_disagree_frames.json").write_text(json.dumps({"frames": dis}))
        for suffix, frames in (("_presence", pres), ("_disagree", dis)):
            (labels_dir / f"{n}{suffix}.json").write_text(json.dumps(
                [{"frame": i, "main": True, "near": [500.0, 900.0], "far": [960.0, 200.0]} for i in frames]))
    # Patches, each only where the real thing needs git, a download or the repo's data folder:
    monkeypatch.setattr(e, "RUNS", runs)  # the run folders live in data/six/runs, which holds no real runs yet
    monkeypatch.setattr(e, "MANIFEST", tmp_path / "manifest.json")  # the download manifest (needs real downloads)
    monkeypatch.setattr(e, "BALL_MODEL", ball)  # a stand-in file; check_provenance compares its real SHA-256
    monkeypatch.setattr(e, "clean", lambda paths: True)  # asks git whether src and the scorer are committed
    monkeypatch.setattr(e, "committed", lambda d: True)  # asks git whether the label files are tracked
    monkeypatch.setattr(e, "same_code", lambda run: True)  # git diff between the run's code_sha.txt and HEAD
    monkeypatch.setattr(e, "head_sha", lambda: "deadbeef")  # git rev-parse HEAD (recorded in the result)
    return labels_dir, tmp_path / "result.json"


def test_score_happy_path_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = _script()
    labels_dir, out = _build(tmp_path, e, monkeypatch, with_players=True)
    e.score(labels_dir, out)
    res = json.loads(out.read_text())
    assert {"fps_scaled_thresholds", "pin_serve_height", "player_size_from_camera", "per_video",
            "reported_not_used", "code_sha", "labels_sha256"} <= res.keys()
    assert res["code_sha"] == "deadbeef" and len(res["per_video"]) == 6 and set(res["per_video"]) == set(e.NAMES)
    assert len(res["labels_sha256"]) == 6 * 7 and all(len(h) == 64 for h in res["labels_sha256"].values())

    fps = res["fps_scaled_thresholds"]  # default vs scaled
    assert fps["decision"] == "accepted" and all(fps["criteria"].values())
    assert fps["numbers"]["recall_default"] == 25.0 and fps["numbers"]["recall_scaled"] == 100.0
    assert fps["numbers"]["labelled"] == 24 and fps["numbers"]["videos_50fps"] == 6
    assert fps["numbers"]["false_serves_default"] == 0 and fps["numbers"]["false_serves_scaled"] == 0

    pin = res["pin_serve_height"]  # scaled vs scaled_pin (default vs scaled_pin would have 6 paired: inconclusive)
    assert pin["decision"] == "accepted" and all(pin["criteria"].values())
    assert pin["numbers"]["paired"] == 24 and pin["numbers"]["both_measured"] == 23
    assert pin["numbers"]["closer"] == 23 and pin["numbers"]["further_or_lost"] == 1
    assert abs(pin["numbers"]["mean_abs_drop_points"] - 9.0) < 1e-6
    assert pin["numbers"]["lost_by_pinned"] == [{"video": 0, "start_frame": 3099, "pinned": "low contact"}]
    assert pin["numbers"]["verdict_changes"] == []

    nm = res["score_consistency"]  # default vs consistency, on the 31 drawn flights of each video (0 is unclear)
    assert nm["decision"] == "accepted" and all(nm["criteria"].values()) and len(nm["criteria"]) == 3
    assert nm["numbers"]["default"] == {"right": 120, "wrong": 60, "none": 0}  # every third rally named wrongly
    assert nm["numbers"]["candidate"] == {"right": 120, "wrong": 0, "none": 60}  # shown without a name instead
    assert nm["numbers"]["videos_with_other_changes"] == []

    pl = res["player_size_from_camera"]  # default vs players
    assert pl["decision"] == "accepted" and len(pl["criteria"]) == 4 and all(pl["criteria"].values())
    assert pl["numbers"]["near_present"] == 480 and pl["numbers"]["near_coverage"][1] == 100.0
    assert pl["numbers"]["near_coverage"][0] < 95.0 and pl["numbers"]["far_coverage"] == [100.0, 100.0]
    assert pl["numbers"]["wrong_share"] == [0.0, 0.0]
    assert pl["numbers"]["disagreements"]["near"] == {"one_right": 360, "b_right": 360}
    assert pl["numbers"]["disagreements"]["far"] == {"one_right": 0, "b_right": 0}

    pv = res["per_video"][e.NAMES[0]]
    assert pv["default"] == {"labelled": 4, "matched": 1, "within2": 1, "false_serves": 0}
    assert pv["scaled"]["matched"] == 4 and pv["fps"] == FPS and set(pv) == {*e.KINDS, "fps"}
    rep = res["reported_not_used"]
    assert rep["fps_scaled_thresholds"]["ball_detected_frames"] == {"default": 6000, "scaled": 6240}
    assert rep["pin_serve_height"]["with_toss_bridge"]["decision"] == "accepted"
    assert rep["player_size_from_camera"]["labelled_frames_outside_main_camera"] == dict.fromkeys(e.NAMES, 0)

    with pytest.raises(SystemExit, match="scored once"):  # the result exists now
        e.score(labels_dir, out)


def test_score_without_player_or_name_labels_refuses_unless_those_tests_are_given_up(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e = _script()
    labels_dir, out = _build(tmp_path, e, monkeypatch, with_players=False)
    with pytest.raises(SystemExit, match="--skip players"):  # scoring is once: it must not lose those tests silently
        e.score(labels_dir, out)
    with pytest.raises(SystemExit, match="--skip names"):
        e.score(labels_dir, out, skip=frozenset({"players"}))
    assert not out.exists()
    e.score(labels_dir, out, skip=frozenset({"players", "names"}))
    res = json.loads(out.read_text())
    assert res["skipped"] == ["names", "players"] and res["score_consistency"] == "no name labels"
    assert res["player_size_from_camera"] == "no presence labels"
    assert res["reported_not_used"]["player_size_from_camera"]["labelled_frames_outside_main_camera"] == (
        "no presence labels")
    assert res["fps_scaled_thresholds"]["decision"] == "accepted"
    assert res["pin_serve_height"]["decision"] == "accepted"
    assert len(res["labels_sha256"]) == 6


@pytest.mark.parametrize("cmd", ["sample", "run", "disagreements", "names"])
def test_once_the_result_exists_every_command_but_score_refuses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cmd: str) -> None:
    e = _script()
    spent = tmp_path / "six_videos_test.json"
    spent.write_text("{}")
    monkeypatch.setattr(e, "SPENT", spent)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    called: list[str] = []
    for name in ("sample", "run", "disagreements", "names_flights", "score"):
        monkeypatch.setattr(e, name, lambda *a, _n=name, **k: called.append(_n))
    monkeypatch.setattr(sys, "argv", ["eval_six_videos.py", cmd, "--videos", "v", "--labels", "l", "--out", "o"])
    with pytest.raises(SystemExit, match="spent"):
        e.main()
    assert called == []
    monkeypatch.setattr(sys, "argv", ["eval_six_videos.py", "score", "--labels", "l", "--out", "o"])
    e.main()  # score is left to its own refusal (an existing result file)
    assert called == ["score"]
