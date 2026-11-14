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
