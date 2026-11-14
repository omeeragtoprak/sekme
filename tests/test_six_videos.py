"""The six-video test's scoring (docs/EVAL_SERVE_SPEED.md), on synthetic labels and reports."""

from __future__ import annotations

import sys
from pathlib import Path

from courtvision.eval.six_videos import decisive_flips, fps_rule, match, pin_rule, recall_stats, system_serves

FPS = 59.94


def _label(lo: int, side: str = "near", mph: float | None = 120.0) -> dict:
    return {"lo": lo, "hi": lo + 1, "side": side, "radar": None if mph is None else {"value": mph, "unit": "mph"}}


def _serve(i: int, start: int, contact: float, side: str = "near", kmh: float | None = 190.0,
           verdict: str | None = "IN", rejected: str | None = None, bounce: float | None = None) -> dict:
    f = {"id": i, "kind": "serve", "hitter": side, "start_frame": start, "contact_frame": contact,
         "hit_speed_kmh": kmh, "rejected": rejected, "call": None, "bounce": None}
    if verdict is not None:
        f["call"] = {"verdict": verdict}
        f["bounce"] = {"frame": bounce if bounce is not None else contact + 25}
    return f


def _report(flights: list[dict]) -> dict:
    return {"video": {"fps": FPS}, "flights": flights}


def test_matching_is_the_one_of_the_serve_contact_test() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_serve_contact as ref

    labels = [_label(100), _label(500), _label(900, "far")]
    rep = _report([_serve(0, 98, 101.5), _serve(1, 520, 531.0), _serve(2, 890, 899.0, "far"),
                   _serve(3, 300, 300.0)])
    mine = {i: (s["id"], e) for i, (s, e) in match(labels, system_serves(rep)).items()}
    theirs = {i: (s["id"], e) for i, (s, e) in ref.match(labels, ref.system_serves(rep)).items()}
    assert mine == theirs and set(mine) == {0, 1, 2}


def test_recall_counts_and_a_likely_fault_is_not_a_false_serve() -> None:
    labels = [_label(1000), _label(3000)]
    rep = _report([_serve(0, 999, 1000.5), _serve(1, 2000, 2000.0),  # 2000: 1000 frames (17 s) before 3000: a fault
                   _serve(2, 6000, 6000.0)])  # 6000: nothing labelled after it within 30 s: a false serve
    st = recall_stats(labels, rep)
    assert (st["labelled"], st["matched"], st["within2"], st["false_serves"]) == (2, 1, 1, 1)


def test_decisive_flips_need_two_opposite_decisive_calls() -> None:
    a = _report([_serve(0, 0, 10.0, verdict="IN", bounce=40.0), _serve(1, 100, 110.0, verdict="IN", bounce=140.0)])
    b = _report([_serve(0, 0, 10.0, verdict="OUT", bounce=41.0), _serve(1, 100, 110.0, verdict="REVIEW",
                                                                         bounce=140.0)])
    assert decisive_flips(a, b) == [(40.0, "IN", "OUT")]


def _videos(n_videos: int, fps: float, found_default: int, found_scaled: int, flip: bool = False):
    out = []
    for v in range(n_videos):
        labels = [_label(1000 * k + 100) for k in range(5)]
        d = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, bounce=1000 * k + 130) for k in range(found_default)]
        s = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, bounce=1000 * k + 130) for k in range(found_scaled)]
        if flip and v == 0:
            s[0]["call"]["verdict"] = "OUT"
        out.append((fps, labels, {"video": {"fps": fps}, "flights": d}, {"video": {"fps": fps}, "flights": s}))
    return out


def test_fps_rule_accepts_more_serves_and_rejects_a_confident_flip() -> None:
    v = fps_rule(_videos(6, FPS, 1, 4))
    assert v.decision == "accepted" and v.numbers["recall_default"] == 20.0 and v.numbers["recall_scaled"] == 80.0
    assert fps_rule(_videos(6, FPS, 1, 4, flip=True)).decision == "rejected"
    assert fps_rule(_videos(6, FPS, 2, 2)).criteria["recall +10 points"] is False
    assert fps_rule(_videos(6, 29.97, 1, 4)).decision == "cannot decide"  # nothing to test below 50 fps


def _pin_videos(n: int, err_u: float, err_p: float, lost: int = 0, change: bool = False):
    labels = [_label(1000 * k + 100, mph=120.0) for k in range(n)]
    board = 120.0 * 1.609344
    u = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, kmh=board * (1 + err_u / 100)) for k in range(n)]
    p = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, kmh=board * (1 + err_p / 100),
                rejected="rejected" if k < lost else None) for k in range(n)]
    if change:
        p[-1]["call"]["verdict"] = "REVIEW"
    return [(labels, {"video": {"fps": FPS}, "flights": u}, {"video": {"fps": FPS}, "flights": p})]


def test_pin_rule_paths() -> None:
    assert pin_rule(_pin_videos(10, 12.0, 3.0)).decision == "inconclusive"  # fewer than 15 paired serves
    ok = pin_rule(_pin_videos(20, 12.0, 3.0))
    assert ok.decision == "accepted" and abs(ok.numbers["mean_abs_drop_points"] - 9.0) < 1e-9
    assert pin_rule(_pin_videos(20, 12.0, 3.0, change=True)).decision == "rejected"  # a serve verdict changed
    # Serves the pinned run rejects count as worse: 8 lost of 20 leaves 12 closer < two thirds of 20.
    lost = pin_rule(_pin_videos(20, 12.0, 3.0, lost=8))
    assert lost.criteria["two thirds closer"] is False and lost.numbers["further_or_lost"] == 8


def test_presence_sample_depends_only_on_the_name() -> None:
    from courtvision.eval.six_videos import presence_sample

    a, b = presence_sample("w2024_final", 40000), presence_sample("w2024_final", 40000)
    assert a == b and len(a) == 80 and a == sorted(a) and presence_sample("m2025_sf", 40000) != a


def _frames(n: int, near_box, far_box) -> dict:
    return {"players": {"near": {"box": [near_box(i) for i in range(n)]},
                        "far": {"box": [far_box(i) for i in range(n)]}}}


def test_player_rule_paths() -> None:
    from courtvision.eval.six_videos import disagreement_stats, player_rule, presence_stats

    n = 200
    labels = [{"frame": i, "main": True, "near": [500.0, 900.0], "far": [960.0, 200.0]} for i in range(n)]
    on_near = [480.0, 700.0, 520.0, 900.0]  # feet at (500, 900), 200 px tall
    on_far = [945.0, 130.0, 975.0, 200.0]  # feet at (960, 200), 70 px tall
    beside_far = [985.0, 130.0, 1015.0, 200.0]  # a ball kid 40 px away: more than 0.3 x 70 px
    default = _frames(n, lambda i: on_near if i % 2 else None, lambda i: on_far)  # near found half the time
    better = _frames(n, lambda i: on_near, lambda i: on_far)
    st = presence_stats(labels, default, 1080)
    assert (st["near"]["present"], st["near"]["found"], st["near"]["wrong"]) == (200, 100, 0)
    assert presence_stats(labels, _frames(n, lambda i: None, lambda i: beside_far), 1080)["far"]["wrong"] == 200
    dis = labels[1::2][:30]  # disagreement frames: the even frames where default misses the near player
    dis = [dict(lb, frame=lb["frame"] - 1) for lb in dis]
    ds = disagreement_stats(dis, default, better, 1080)
    assert ds["near"] == {"one_right": 30, "b_right": 30} and ds["far"]["one_right"] == 0
    v = player_rule([(labels, dis, default, better, 1080)])
    assert v.decision == "accepted" and v.numbers["near_coverage"] == [50.0, 100.0]
    wrong_spot = _frames(n, lambda i: [1480.0, 700.0, 1520.0, 900.0], lambda i: on_far)  # someone else
    assert player_rule([(labels, dis, default, wrong_spot, 1080)]).decision == "rejected"
    assert player_rule([(labels[:50], dis, default, better, 1080)]).decision == "inconclusive"  # < 100 frames


def test_score_takes_only_serve_label_files_as_videos(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_six_videos

    for suffix in ("", "_presence", "_presence_frames", "_disagree", "_disagree_frames"):
        (tmp_path / f"m2025_final{suffix}.json").write_text("{}")
    (tmp_path / "w2024_sf.json").write_text("{}")
    assert eval_six_videos.label_names(tmp_path) == ["m2025_final", "w2024_sf"]


def test_inclusive_thresholds_hold_exactly_at_the_boundary() -> None:
    # 30 labelled serves, matched 4 -> 7: exactly +10 points (float subtraction gives 9.999999999999998).
    vids = []
    for v in range(3):
        labels = [_label(1000 * k + 100) for k in range(10)]
        found_d, found_s = (2, 3) if v == 0 else (1, 2)
        d = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, bounce=1000 * k + 130) for k in range(found_d)]
        s = [_serve(k, 1000 * k + 99, 1000 * k + 100.5, bounce=1000 * k + 130) for k in range(found_s)]
        vids.append((FPS, labels, {"video": {"fps": FPS}, "flights": d}, {"video": {"fps": FPS}, "flights": s}))
    v = fps_rule(vids)
    assert v.numbers["recall_default"] * 30 / 100 == 4 and v.criteria["recall +10 points"] is True


def test_player_coverage_boundary_is_exact() -> None:
    from courtvision.eval.six_videos import player_rule

    n = 120  # near present 120, found 5 -> 11: +5 points exactly (float gives 4.999...)
    labels = [{"frame": i, "main": True, "near": [500.0, 900.0], "far": None} for i in range(n)]
    on_near = [480.0, 700.0, 520.0, 900.0]
    a = _frames(n, lambda i: on_near if i < 5 else None, lambda i: None)
    b = _frames(n, lambda i: on_near if i < 11 else None, lambda i: None)
    v = player_rule([(labels, [], a, b, 1080)])
    assert v.criteria["near coverage +5 points"] is True


def test_a_flip_is_not_hidden_by_one_to_one_pairing() -> None:
    a = _report([_serve(0, 0, 10.0, verdict="IN", bounce=40.0)])
    b = _report([_serve(0, 0, 10.0, verdict="IN", bounce=39.0), _serve(1, 0, 10.0, verdict="OUT", bounce=41.0)])
    assert decisive_flips(a, b) == [(40.0, "IN", "OUT")]


def test_labels_are_validated() -> None:
    import pytest

    from courtvision.eval.six_videos import check_player_labels, check_serve_labels

    check_serve_labels([_label(100), _label(200, mph=None)])
    for bad in ({"lo": 5, "hi": 4, "side": "near", "radar": None}, {"lo": 1, "hi": 2, "side": "left", "radar": None},
                {"lo": 1, "hi": 2, "side": "near"}, {"lo": 1, "hi": 2, "side": "near", "radar": {"value": 120}}):
        with pytest.raises(ValueError):
            check_serve_labels([bad])
    check_player_labels([{"frame": 3, "main": True, "near": [1.0, 2.0], "far": None}])
    for bad in ({"frame": 3, "near": None, "far": None}, {"frame": 3, "main": True, "near": [1.0], "far": None},
                {"frame": -1, "main": False, "near": None, "far": None}):
        with pytest.raises(ValueError):
            check_player_labels([bad])
    with pytest.raises(ValueError):  # a frame labelled twice
        check_player_labels([{"frame": 3, "main": False, "near": None, "far": None}] * 2)


def test_reported_call_changes_and_speeds() -> None:
    from courtvision.eval.six_videos import call_changes, speed_report

    def call(v: str, x: float, sig: float, p: float) -> dict:
        return {"verdict": v, "x": x, "y": 10.0, "sigma_m": sig, "p_in": p}

    a = _report([_serve(0, 0, 10.0, bounce=40.0), _serve(1, 200, 210.0, bounce=240.0)])
    b = _report([_serve(0, 0, 10.0, bounce=41.0), _serve(1, 200, 210.0, verdict="REVIEW", bounce=240.0),
                 _serve(2, 400, 410.0, bounce=440.0)])
    for f, c in zip(a["flights"] + b["flights"], [call("IN", 1.0, 0.02, 1.0), call("IN", 2.0, 0.02, 0.9),
                                                  call("IN", 1.1, 0.03, 1.0), call("REVIEW", 2.0, 0.02, 0.5),
                                                  call("IN", 3.0, 0.02, 1.0)], strict=True):
        f["call"] = c
    cc = call_changes([(a, b)])
    assert (cc["calls_a"], cc["calls_b"], cc["only_a"], cc["only_b"], cc["review_changes"]) == (2, 3, 0, 1, 1)
    assert cc["verdict_changes"]["serve"] == [(240.0, "IN", "REVIEW")]
    assert abs(cc["paired"]["serve"]["shift_cm"]["max"] - 10.0) < 1e-9
    sp = speed_report([([_label(8, mph=100.0), _label(208, mph=120.0)], a)])
    assert sp["labelled_board"]["n"] == 2 and sp["board_115mph_up"]["n"] == 1 and sp["board_below_100mph"]["n"] == 0


def _script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_six_videos

    return eval_six_videos


def test_score_refuses_anything_but_the_six_and_an_existing_result(tmp_path: Path) -> None:
    import pytest

    e = _script()
    assert len(e.NAMES) == 6
    for n in e.NAMES[:5]:
        (tmp_path / f"{n}.json").write_text('{"serves": []}')
    with pytest.raises(SystemExit, match="exactly the six"):
        e.score(tmp_path, tmp_path / "out.txt")
    (tmp_path / f"{e.NAMES[5]}.json").write_text('{"serves": []}')
    (tmp_path / "out.txt").write_text("{}")
    with pytest.raises(SystemExit, match="scored once"):
        e.score(tmp_path, tmp_path / "out.txt")


def test_provenance_catches_wrong_flags_video_and_models() -> None:
    e = _script()
    models = {"ball": {"sha256": "b4"}, "court": {"sha256": "c"}}

    def rep(kind: str) -> dict:
        return {"candidates": {f: True for f in e.FLAGS[kind]}, "video": {"sha256": "v"}, "models": models}

    runs = {k: rep(k) for k in e.KINDS}
    assert e.check_provenance("x", runs, {"x": {"sha256": "v"}}, "b4") == []
    runs["scaled"] = rep("default")  # a run made without its flag
    runs["players"] = {**rep("players"), "video": {"sha256": "other"}}
    runs["bridge"] = {**rep("bridge"), "models": {"ball": {"sha256": "b3"}, "court": {"sha256": "c"}}}
    problems = e.check_provenance("x", runs, {"x": {"sha256": "v"}}, "b4")
    assert any("x/scaled: made with []" in p for p in problems)
    assert any("x/players: video" in p for p in problems)
    assert any("x/bridge: models" in p for p in problems)
    unlisted = e.check_provenance("x", {k: rep(k) for k in e.KINDS}, {}, "b4")
    assert unlisted[0] == "x: no SHA-256 in manifest.json" and len(unlisted) == 1 + len(e.KINDS)
