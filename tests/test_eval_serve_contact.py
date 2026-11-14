"""Scoring of serve contacts against brackets labelled by eye (scripts/eval_serve_contact.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_serve_contact as esc  # noqa: E402


def test_bracket_error_is_zero_inside_and_distance_outside():
    assert esc.bracket_error(8887.5, 8887, 8888) == 0.0
    assert esc.bracket_error(8878.15, 8887, 8888) == 8887 - 8878.15
    assert esc.bracket_error(8889.2, 8887, 8888) == 8889.2 - 8888


def test_match_is_one_to_one_closest_first_same_side_within_half_a_second():
    labels = [{"lo": 100, "hi": 101, "side": "far"}, {"lo": 120, "hi": 121, "side": "far"},
              {"lo": 500, "hi": 501, "side": "near"}]
    serves = [{"id": 1, "side": "far", "est": 110.0, "kmh": None},   # 9 from label 0, 10 from label 1
              {"id": 2, "side": "far", "est": 102.0, "kmh": None},   # 1 from label 0
              {"id": 3, "side": "far", "est": 500.0, "kmh": None},   # wrong side for label 2
              {"id": 4, "side": "near", "est": 540.0, "kmh": None}]  # 39 frames away: no match
    m = esc.match(labels, serves)
    assert m[0][0]["id"] == 2 and m[0][1] == 1.0
    assert m[1][0]["id"] == 1 and m[1][1] == 10.0
    assert 2 not in m
    s = esc.summary(labels, m)
    assert (s["labelled"], s["matched"], s["good"]) == (3, 2, 1)
