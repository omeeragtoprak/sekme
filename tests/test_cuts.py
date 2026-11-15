"""Shot cuts beyond the colour histogram: edits from the same camera position, and the frame differences they
are found from (perception/court.py, pipeline.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def test_same_angle_edits_are_sudden_whole_frame_changes_on_calibrated_frames(tmp_path: Path) -> None:
    from courtvision.perception.court import frame_differences, same_angle_edits

    diffs = np.array([0, .3, .3, .3, .3, 6.0, .3, .3, .3, .6, .9, 1.2, 1.6, 2.0, 2.5, 2.9, 3.2, 3.4, 3.4, 3.3,
                      .3, .3, .3, .3, .3, .3, 4.0, .3], dtype=np.float32)
    valid = [True] * len(diffs)
    # 5: an edit; 9-19: a camera move building up over frames (never 3 times its own past); 26: an edit
    assert same_angle_edits(diffs, valid) == [5, 26]
    valid[25] = False  # the frame before is not calibrated (a close-up, say): nothing measured to split there
    assert same_angle_edits(diffs, valid) == [5]
    assert same_angle_edits(diffs[:3], [True] * 3) == []  # too little past to compare with

    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    ramp = np.linspace(0, 255, 160, dtype=np.uint8)
    scene_a = np.repeat(np.tile(ramp, (90, 1))[..., None], 3, axis=2)
    scene_b = scene_a[:, ::-1].copy()  # the same brightness overall, every pixel moved
    for i in range(10):
        w.write(scene_a if i < 6 else scene_b)
    w.release()
    d = frame_differences(path)
    assert len(d) == 10 and d[0] == 0 and d[6] > 20 and max(d[1:6].max(), d[7:].max()) < 3


def test_frame_differences_are_taken_from_perception_else_a_checked_cache_else_a_pass(monkeypatch, tmp_path) -> None:
    import courtvision.pipeline as pipeline

    passes = []

    def one_pass(video):
        passes.append(video)
        return np.arange(5, dtype=np.float32)

    monkeypatch.setattr(pipeline, "frame_differences", one_pass)
    measured = np.ones(5, dtype=np.float32)
    assert pipeline._frame_diffs("v", "sha", 5, tmp_path, True, measured) is measured and not passes
    assert pipeline._frame_diffs("v", "sha", 6, tmp_path, True, measured) is None  # not this video's frames
    assert list(pipeline._frame_diffs("v", "sha", 5, tmp_path, True, None)) == [0, 1, 2, 3, 4] and len(passes) == 1
    pipeline._frame_diffs("v", "sha", 5, tmp_path, True, None)
    assert len(passes) == 1  # cached
    pipeline._frame_diffs("v", "other video", 5, tmp_path, True, None)
    assert len(passes) == 2  # another video's cache is not used
    (tmp_path / "frame_diffs.pkl").write_bytes(b"\x80\x04 half written")
    assert pipeline._frame_diffs("v", "sha", 5, tmp_path, True, None) is not None and len(passes) == 3
    assert pipeline._frame_diffs("v", "sha", 9, tmp_path, True, None) is None  # the pass does not cover n frames
