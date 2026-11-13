from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from courtvision.api.app import create_app
from courtvision.eval.labels import serve_summary
from test_api import _make_run


def _serve(i: str, frame: int) -> dict:
    return {"id": i, "type": "serve", "frame": frame}


def test_serve_summary_matches_one_to_one_within_tolerance() -> None:
    recs = [_serve("a", 100), _serve("b", 104), _serve("c", 400)]
    s = serve_summary(recs, [102.0, 700.0], fps=30)  # tolerance 15 frames
    # Both a and b are in range of 102, but only one pair is allowed: the closest (a, gap 2) wins.
    assert (s["labelled"], s["system"], s["matched"]) == (3, 2, 1)
    assert s["recall"] == round(1 / 3, 3)
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["a"]["matched_system_frame"] == 102.0 and rows["a"]["dt_frames"] == 2.0
    assert rows["b"]["matched_system_frame"] is None and rows["c"]["dt_frames"] is None
    assert s["unmatched_system"] == [700.0]
    edge = serve_summary([_serve("a", 100)], [115.0], fps=30)  # exactly at tolerance
    assert edge["matched"] == 1
    assert serve_summary([_serve("a", 100)], [116.0], fps=30)["matched"] == 0


def test_serve_summary_replays_removals_and_withholds_precision_until_complete() -> None:
    recs = [_serve("a", 100), _serve("b", 300), {"id": "r", "type": "remove", "target": "b"}]
    s = serve_summary(recs, [100.0, 500.0], fps=30)
    assert s["labelled"] == 1 and s["recall"] == 1.0
    assert s["complete"] is False and s["precision"] is None
    done = serve_summary([*recs, {"id": "c1", "type": "complete", "complete": True}], [100.0, 500.0], fps=30)
    assert done["precision"] == 0.5 and done["unmatched_system"] == [500.0]
    undone = serve_summary([*recs, {"id": "c1", "type": "complete", "complete": True},
                            {"id": "c2", "type": "complete", "complete": False}], [100.0], fps=30)
    assert undone["complete"] is False and undone["precision"] is None  # latest wins
    empty = serve_summary([], [100.0], fps=30)
    assert empty["recall"] is None and empty["labelled"] == 0


def test_serve_label_endpoints(tmp_path: Path) -> None:
    d = _make_run(tmp_path / "runs")
    report = json.loads((d / "report.json").read_text())
    report["flights"] += [{"id": 2, "kind": "serve", "start_frame": 8, "contact_frame": 10.5, "call": None},
                          {"id": 3, "kind": "serve", "start_frame": 40, "contact_frame": None, "call": None},
                          {"id": 4, "kind": "rally", "start_frame": 20, "call": None}]
    (d / "report.json").write_text(json.dumps(report))
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "no-dist"))
    url = "/api/runs/clip1/serve-labels"
    assert c.get(url).json() == [] and c.get(f"{url}/summary").json()["labelled"] == 0

    a = c.post(url, json={"type": "serve", "frame": 12, "labeler": "Ayşe"})
    assert a.status_code == 201 and a.json()["frame"] == 12 and a.json()["note"] == ""
    b = c.post(url, json={"type": "serve", "frame": 56, "labeler": "Ayşe", "note": "fault"}).json()
    s = c.get(f"{url}/summary").json()
    assert (s["labelled"], s["system"], s["matched"]) == (2, 2, 1)  # 12 ~ 10.5; 56 vs 40 is beyond 0.5 s
    assert s["unmatched_system"] == [40.0] and s["precision"] is None

    assert c.post(url, json={"type": "remove", "target": b["id"], "labeler": "Ayşe"}).status_code == 201
    assert c.post(url, json={"type": "complete", "complete": True, "labeler": "Ayşe"}).status_code == 201
    s = c.get(f"{url}/summary").json()
    assert (s["labelled"], s["matched"], s["recall"], s["precision"]) == (1, 1, 1.0, 0.5)
    assert s["rows"][0]["matched_system_frame"] == 10.5 and s["unmatched_system"] == [40.0]
    assert len(c.get(url).json()) == 4  # append-only: nothing is ever dropped from the log

    ok = {"labeler": "Ayşe"}
    assert c.post(url, json={"type": "serve", "frame": 60, **ok}).status_code == 422  # n = 60
    assert c.post(url, json={"type": "serve", "frame": -1, **ok}).status_code == 422
    assert c.post(url, json={"type": "serve", **ok}).status_code == 422  # frame missing
    assert c.post(url, json={"type": "serve", "frame": 12, **ok}).status_code == 409  # duplicate
    assert c.post(url, json={"type": "serve", "frame": 13, **ok}).status_code == 409  # same serve, 1 frame on
    assert c.post(url, json={"type": "remove", "target": "nope", **ok}).status_code == 404
    assert c.post(url, json={"type": "remove", "target": b["id"], **ok}).status_code == 409  # already removed
    complete_id = c.get(url).json()[-1]["id"]
    assert c.post(url, json={"type": "remove", "target": complete_id, **ok}).status_code == 409  # not a serve
    assert c.get("/api/runs/..%2Fx/serve-labels").status_code in (400, 404)

    log = next(p for p in tmp_path.rglob("serve_labels.jsonl"))
    with log.open("a") as f:
        f.write('{"id": "cut-short", "type": "se')  # a crash mid-append leaves half a line
    assert c.get(f"{url}/summary").status_code == 200 and c.get(f"{url}/summary").json()["labelled"] == 1
