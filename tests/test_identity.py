"""Telling the two players apart by clothing (analytics/identity.py) and how the pipeline uses it."""

from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.analytics import identity

COURT = (150, 110, 60)  # BGR blue-ish court
SHIRTS = {"red": (40, 40, 200), "blue": (200, 90, 30), "black": (25, 25, 25), "white": (235, 235, 235)}


def player(shirt: str, shorts: str = "white", noise: int = 0, seed: int = 0) -> np.ndarray:
    img = np.zeros((identity.CROP_H, identity.CROP_W, 3), np.uint8)
    img[:] = COURT
    img[4:10, 9:15] = (120, 150, 200)  # head
    img[10:30, 6:18] = SHIRTS[shirt]
    img[30:38, 7:17] = SHIRTS[shorts]
    if noise:
        img = np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def test_clothing_separates_shirt_colours_and_ignores_the_court() -> None:
    red, red2, blue = (identity.clothing(player("red", noise=8, seed=s)) for s in (1, 2, 3))
    blue = identity.clothing(player("blue", noise=8, seed=4))
    assert red is not None and blue is not None and red2 is not None
    assert abs(red.sum() - 1) < 1e-9
    assert identity.hellinger(red, red2) < 0.2 < identity.hellinger(red, blue)
    black, white = identity.clothing(player("black", "black")), identity.clothing(player("white", "white"))
    assert identity.hellinger(black, white) > 0.8  # unsaturated clothing still tells dark from light
    empty = np.zeros((identity.CROP_H, identity.CROP_W, 3), np.uint8)
    empty[:] = COURT
    assert identity.clothing(empty) is None  # nothing but court


def test_play_stretches_break_at_cuts_and_long_pauses() -> None:
    fps = 30.0
    starts = [0, 60, 120, 900, 960, 1000, 1100]  # 900: 26 s after 120; a cut at 980
    assert identity.play_stretches(starts, [980], fps) == [0, 0, 0, 1, 1, 2, 2]
    assert identity.play_stretches(list(reversed(starts)), [980], fps) == [2, 2, 1, 1, 0, 0, 0]  # any order
    assert identity.play_stretches([], [], fps) == []


def test_assign_finds_who_is_near_in_every_stretch() -> None:
    a = [identity.clothing(player("red", noise=6, seed=s)) for s in range(30)]
    b = [identity.clothing(player("black", noise=6, seed=100 + s)) for s in range(30)]
    near_is_a = {0: True, 1: False, 2: True, 3: False}
    near = {k: (a if v else b)[k * 5:k * 5 + 5] for k, v in near_is_a.items()}
    far = {k: (b if v else a)[k * 5:k * 5 + 5] for k, v in near_is_a.items()}
    near[0] = near[0] + a[20:30]  # stretch 0 is the largest: A is its near player
    res = identity.assign(near, far)
    assert res.near_is_a == near_is_a and all(res.decided(k) for k in near_is_a)
    near[4], far[4] = a[:3], []  # a stretch with only one player seen is not compared
    assert 4 not in identity.assign(near, far).near_is_a
    same = identity.assign({0: a[:5], 1: a[5:10]}, {0: a[10:15], 1: a[15:20]})  # both dressed alike
    assert not any(same.decided(k) for k in (0, 1))
    assert identity.assign({}, {}).near_is_a == {}


def test_crops_at_reads_only_the_requested_frames(tmp_path: Path) -> None:
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(12):
        f = np.zeros((48, 64, 3), np.uint8)
        f[:, :] = (0, 0, 20 * i)  # red grows frame by frame
        w.write(f)
    w.release()
    box = np.array([10.0, 5.0, 30.0, 45.0])
    out, complete = identity.crops_at(path, [(7, box), (2, box), (7, np.array([np.nan] * 4))])
    assert complete and out[0].shape == (identity.CROP_H, identity.CROP_W, 3)
    assert abs(int(out[0][..., 2].mean()) - 140) <= 6 and abs(int(out[1][..., 2].mean()) - 40) <= 6
    assert out[2] is None  # no box
    out, complete = identity.crops_at(path, [(2, box), (50, box)])
    assert not complete and out[0] is not None and out[1] is None  # past the end: the caller does not cache it
    assert identity.crops_at(tmp_path / "missing.avi", [(2, box)]) == ([None], False)


def _flight(i: int, start: int, hitter: str, hitter_name: str | None) -> dict:
    other = {"SINNER": "FRITZ", "FRITZ": "SINNER", None: None}[hitter_name]
    return {"id": i, "start_frame": start, "end_frame": start + 20, "hitter": hitter,
            "receiver": "far" if hitter == "near" else "near", "hitter_name": hitter_name, "receiver_name": other}


def test_the_pipeline_names_players_by_clothing_where_the_graphic_flips_them(monkeypatch, tmp_path) -> None:
    import courtvision.pipeline as pipeline

    fps, n = 30.0, 2200
    # Three stretches (a cut at 300, a 40 s pause before 2000). Red SINNER is near in 0 and 2, black FRITZ in 1;
    # the graphic (a misread game count) says SINNER is near in all three.
    flights = []
    for first in (0, 400, 2000):
        for j in range(6):
            hitter = "near" if j % 2 == 0 else "far"
            flights.append({"id": len(flights), "start_frame": first + 40 * j, "end_frame": first + 40 * j + 20,
                            "hitter": hitter, "receiver": "far" if hitter == "near" else "near",
                            "hitter_name": "SINNER" if hitter == "near" else "FRITZ",
                            "receiver_name": "FRITZ" if hitter == "near" else "SINNER"})
    near_shirt = {0: "red", 1: "black", 2: "red"}
    stretch = identity.play_stretches([f["start_frame"] for f in flights], [300], fps)
    assert stretch == [0] * 6 + [1] * 6 + [2] * 6

    def fake_crops(video, reqs):
        out = []
        for fr, box in reqs:
            k = next(s for f, s in zip(flights, stretch, strict=True) if f["start_frame"] <= fr <= f["end_frame"])
            shirt = near_shirt[k] if box[1] > 100 else ("black" if near_shirt[k] == "red" else "red")
            out.append(player(shirt, noise=5, seed=fr))
        return out, True

    monkeypatch.setattr(pipeline.identity, "crops_at", fake_crops)
    players = {"near": types.SimpleNamespace(boxes=np.tile([100.0, 300.0, 140.0, 400.0], (n, 1))),
               "far": types.SimpleNamespace(boxes=np.tile([100.0, 20.0, 120.0, 70.0], (n, 1)))}
    cfg = pipeline.PipelineConfig(reuse_perception=True)
    names = ("SINNER", "FRITZ")
    info = pipeline._annotate_identity("v.mp4", flights, players, [300], fps, n, tmp_path, "sha", cfg, names)
    near_names = [f["hitter_name"] if f["hitter"] == "near" else f["receiver_name"] for f in flights]
    assert info["stretches"] == info["decided"] == 3 and info["names"] == {"A": "SINNER", "B": "FRITZ"}
    assert near_names == ["SINNER"] * 6 + ["FRITZ"] * 6 + ["SINNER"] * 6  # the flipped stretch is put right
    assert info["renamed_flights"] == 6 and {f["name_source"] for f in flights} == {"görünüm"}
    assert [f["hitter_id"] for f in flights[:2]] == ["A", "B"] and flights[6]["hitter_id"] == "B"
    assert (tmp_path / "identity.pkl").is_file()
    monkeypatch.setattr(pipeline.identity, "crops_at", lambda *a: pytest.fail("cached descriptors not reused"))
    unnamed = [dict(f, hitter_name=None, receiver_name=None) for f in flights]
    info = pipeline._annotate_identity("v.mp4", unnamed, players, [300], fps, n, tmp_path, "sha", cfg, None)
    assert info["names"] is None and unnamed[6]["hitter_name"] is None and unnamed[6]["hitter_id"] == "B"
    assert unnamed[0]["name_source"] is None and unnamed[0]["hitter_id"] == "A"
    bare = [{k: v for k, v in f.items() if k not in ("hitter_name", "receiver_name")} for f in flights]
    info = pipeline._annotate_identity("v.mp4", bare, players, [300], fps, n, tmp_path, "sha", cfg, None)
    assert bare[6]["hitter_id"] == "B" and bare[6]["hitter_name"] is None  # no broadcast scan: no names at all
    variants = [dict(f, hitter_name="J. SINNER" if f["hitter_name"] == "SINNER" else f["hitter_name"],
                     receiver_name="J. SINNER" if f["receiver_name"] == "SINNER" else f["receiver_name"])
                for f in flights]
    info = pipeline._annotate_identity("v.mp4", variants, players, [300], fps, n, tmp_path, "sha", cfg, names)
    assert info["names"] == {"A": "SINNER", "B": "FRITZ"}  # FRITZ alone still decides; the variant does not count
    tied = [dict(f, hitter_name="ZZ", receiver_name="YY") for f in flights]
    info = pipeline._annotate_identity("v.mp4", tied, players, [300], fps, n, tmp_path, "sha", cfg, names)
    assert info["names"] is None and tied[0]["hitter_name"] == "ZZ"  # no evidence: the graphic's names stay


def test_red_shirts_stay_together_across_the_hue_wrap() -> None:
    a = player("red")
    b = a.copy()
    b[10:30, 6:18] = (60, 30, 200)  # a red a little towards magenta: hue just below 180
    assert identity.hellinger(identity.clothing(a), identity.clothing(b)) < 0.3
