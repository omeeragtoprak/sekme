from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from courtvision.api.app import create_app

CALL = {"verdict": "REVIEW", "p_in": 0.62, "margin_m": 0.01, "sigma_m": 0.04, "line": "far baseline",
        "zone": "far-singles", "x": 0.0, "y": 11.9, "cov": [[0.001, 0], [0, 0.0016]]}


def _make_run(root: Path, run_id: str = "clip1") -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    report = {
        "schema": "courtvision.report/1", "version": "3.0.0a1", "created": "2026-10-27T00:00:00+00:00",
        "video": {"file": "clip.mp4", "duration_s": 2.0},
        "calibration": {"valid_frames": 50, "total_frames": 60},
        "models": {"ball": {"sha256": "abc"}},
        "flights": [{"id": 0, "call": None}, {"id": 1, "call": CALL}],
    }
    (d / "report.json").write_text(json.dumps(report))
    (d / "frames.json").write_text(json.dumps({"fps": 30, "n": 60}))
    (d / "pro.mp4").write_bytes(bytes(range(256)) * 40)
    return d


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _make_run(tmp_path / "runs")
    (tmp_path / "secret.json").write_text("{}")
    return TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "no-dist"))


def test_runs_listing_reports_honest_calibration_ratio(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert runs == [{"id": "clip1", "video": "clip.mp4", "created": "2026-10-27T00:00:00+00:00",
                     "duration_s": 2.0, "n_flights": 2, "n_calls": 1, "calibrated_ratio": 50 / 60,
                     "media": ["pro"]}]


@pytest.mark.parametrize("bad", ["..", "..%2Fsecret.json", ".hidden", "a" * 80, "clip1%00"])
def test_run_id_validation_blocks_traversal(client: TestClient, bad: str) -> None:
    r = client.get(f"/api/runs/{bad}/report")
    assert r.status_code in (400, 404)


def test_media_allow_list_and_range_requests(client: TestClient) -> None:
    assert client.get("/api/runs/clip1/media/report.json").status_code == 404
    assert client.get("/api/runs/clip1/media/classic").status_code == 404  # not rendered
    r = client.get("/api/runs/clip1/media/pro", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-range"].startswith("bytes 0-99/")


def test_review_audit_log_snapshots_system_call(client: TestClient, tmp_path: Path) -> None:
    body = {"flight_id": 1, "decision": "overturn_out", "reviewer": "chair umpire", "note": "mark visible"}
    r = client.post("/api/runs/clip1/reviews", json=body)
    assert r.status_code == 201
    rec = r.json()
    assert rec["system_call"]["verdict"] == "REVIEW" and rec["decision"] == "overturn_out"
    assert rec["models_sha256"] == {"ball": "abc"}
    assert client.get("/api/runs/clip1/reviews").json() == [rec]
    log = (tmp_path / "runs" / "clip1" / "reviews.jsonl").read_text().splitlines()
    assert len(log) == 1
    # Flights without a system call cannot be "reviewed"; invalid payloads are rejected.
    assert client.post("/api/runs/clip1/reviews", json={**body, "flight_id": 0}).status_code == 409
    assert client.post("/api/runs/clip1/reviews", json={**body, "decision": "delete"}).status_code == 422
    assert client.post("/api/runs/clip1/reviews", json={**body, "reviewer": ""}).status_code == 422


def test_upload_job_runs_analysis_and_validates_type(tmp_path: Path) -> None:
    seen: dict = {}

    def fake_analyze(video: Path, out: Path, progress) -> None:
        progress("perception", 0.5)
        seen["video"] = video
        _make_run(out.parent, out.name)

    app = create_app(tmp_path / "runs", uploads_dir=tmp_path / "up", web_dist=tmp_path / "x", analyze_fn=fake_analyze)
    c = TestClient(app)
    assert c.post("/api/jobs", files={"file": ("evil.sh", b"#!/bin/sh", "text/plain")}).status_code == 415
    r = c.post("/api/jobs", files={"file": ("../My Match!.mp4", b"\x00" * 1000, "video/mp4")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(50):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["run_id"].startswith("My_Match_-")
    assert seen["video"].parent == (tmp_path / "up").resolve()
    assert c.get(f"/api/runs/{job['run_id']}/report").status_code == 200


def test_web_index_is_revalidated_but_bundles_are_cacheable(tmp_path: Path) -> None:
    # A browser that cached index.html kept loading the previous bundle after a rebuild.
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="/assets/index-abc.js"></script>')
    (dist / "assets" / "index-abc.js").write_text("console.log(1)")
    c = TestClient(create_app(tmp_path / "runs", web_dist=dist))
    assert c.get("/").headers.get("cache-control") == "no-cache"
    assert "cache-control" not in c.get("/assets/index-abc.js").headers


def test_a_ground_truth_label_is_judged_with_the_systems_rule(tmp_path: Path) -> None:
    from courtvision.eval.labels import zone_from_name

    d = _make_run(tmp_path / "runs")
    H = [50.0, 0.0, 640.0, 0.0, -50.0, 700.0, 0.0, 0.0, 1.0]  # 2 cm per pixel, far baseline near the top
    (d / "frames.json").write_text(json.dumps({"fps": 30, "n": 3, "H": [None, H, H]}))
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "no-dist"))
    out = c.post("/api/runs/clip1/labels", json={"flight_id": 1, "frame": 1, "px": [640, 100], "labeler": "Ayşe"})
    assert out.status_code == 201
    rec = out.json()
    assert rec["world_xy"] == [0.0, 12.0] and rec["px_cm"] == 2.0
    assert rec["eval"]["truth"] == "OUT" and rec["eval"]["error_cm"] == 10.0 and rec["eval"]["agrees"] is None
    assert c.post("/api/runs/clip1/labels", json={"flight_id": 1, "frame": 2, "px": [640, 107.5],
                                                  "labeler": "Ayşe"}).json()["eval"]["truth"] == "IN"
    s = c.get("/api/runs/clip1/labels/summary").json()
    assert s["n"] == 1 and s["reviews"] == 1 and s["review_truth"] == {"IN": 1, "OUT": 0}  # latest label wins
    assert len(c.get("/api/runs/clip1/labels").json()) == 2
    bad = {"flight_id": 1, "frame": 0, "px": [640, 100], "labeler": "Ayşe"}
    assert c.post("/api/runs/clip1/labels", json=bad).status_code == 409  # uncalibrated frame
    assert c.post("/api/runs/clip1/labels", json={**bad, "flight_id": 0, "frame": 1}).status_code == 409  # no call
    assert c.post("/api/runs/clip1/labels", json={**bad, "frame": 1, "labeler": ""}).status_code == 422
    left = zone_from_name("service-near-left")
    assert left.x1 > 0 > left.x0 - 0.05 and left.y1 == 0.0  # near box left of centre, open at the net


def test_a_report_without_frames_is_an_evaluation_output_and_is_not_listed(tmp_path: Path) -> None:
    _make_run(tmp_path / "runs")
    d = tmp_path / "runs" / "clip1_variant"  # e.g. a candidate's report kept for a comparison
    d.mkdir()
    (d / "report.json").write_text((tmp_path / "runs" / "clip1" / "report.json").read_text())
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "no-dist"))
    assert [r["id"] for r in c.get("/api/runs").json()] == ["clip1"]
