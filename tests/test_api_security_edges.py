"""Review Room API defences added after a security review (2026-11-16): no framing, no cross-site writes by
Sec-Fetch-Site, declared oversize bodies refused before they are read, no partial upload left on a write error,
and no server paths in a job's error text."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import courtvision.api.app as app_module
from courtvision.api.app import SECURITY_HEADERS, create_app

LOCAL = "http://127.0.0.1"
CALL = {"verdict": "IN", "p_in": 0.9, "margin_m": 0.1, "sigma_m": 0.1, "line": "net"}


def _run(root: Path) -> Path:
    d = root / "clip1"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"video": {"file": "clip.mp4", "duration_s": 2.0},
                                               "flights": [{"id": 1, "call": CALL}]}))
    (d / "frames.json").write_text(json.dumps({"fps": 30, "n": 60}))
    return d


def _client(tmp_path: Path, analyze_fn=None) -> TestClient:
    return TestClient(create_app(tmp_path / "runs", uploads_dir=tmp_path / "up", web_dist=tmp_path / "x",
                                 analyze_fn=analyze_fn), base_url=LOCAL)


def _wait(c: TestClient, job_id: str) -> dict:
    for _ in range(100):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def test_every_page_api_answer_and_error_forbids_framing_and_sniffing(tmp_path: Path) -> None:
    _run(tmp_path / "runs")
    c = _client(tmp_path)
    assert SECURITY_HEADERS == {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
                                "X-Content-Type-Options": "nosniff"}
    for path in ("/", "/api/health", "/api/runs", "/api/runs/clip1/report", "/api/runs/nope/report"):
        r = c.get(path)
        for k, v in SECURITY_HEADERS.items():
            assert r.headers.get(k) == v, (path, k)


def test_a_write_marked_cross_site_by_the_browser_is_refused_and_a_same_origin_one_is_not(tmp_path: Path) -> None:
    run = _run(tmp_path / "runs")
    c = _client(tmp_path)
    body = {"flight_id": 1, "decision": "confirm", "reviewer": "r"}
    r = c.post("/api/runs/clip1/reviews", json=body, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403 and not (run / "reviews.jsonl").exists()
    for site in ("same-origin", "none"):  # "none": typed in the address bar or a bookmark
        assert c.post("/api/runs/clip1/reviews", json=body, headers={"sec-fetch-site": site}).status_code == 201
    assert c.get("/api/runs/clip1/report", headers={"sec-fetch-site": "cross-site"}).status_code == 200  # reads


def test_a_json_write_over_a_megabyte_is_refused_before_the_route_reads_it(tmp_path: Path) -> None:
    run = _run(tmp_path / "runs")
    c = _client(tmp_path)
    big = {"flight_id": 1, "decision": "confirm", "reviewer": "r", "note": "x" * (app_module.MAX_JSON_BYTES + 10)}
    r = c.post("/api/runs/clip1/reviews", json=big)
    assert r.status_code == 413 and r.json()["detail"] == "istek çok büyük"
    assert not (run / "reviews.jsonl").exists()
    ok = {"flight_id": 1, "decision": "confirm", "reviewer": "r", "note": "kısa"}
    assert c.post("/api/runs/clip1/reviews", json=ok).status_code == 201


def test_an_upload_declared_over_the_limit_is_refused_unread_and_leaves_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)  # the early limit is this plus 1 MiB of form overhead
    c = _client(tmp_path, analyze_fn=lambda *a: None)
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"x" * (3 * 1024**2), "video/mp4")})
    assert r.status_code == 413 and r.json()["detail"] == "dosya çok büyük"
    assert not (tmp_path / "up").exists()  # the route never ran
    # Under the early limit but over the real one: still refused by the route, and the file is removed.
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"x" * 5000, "video/mp4")})
    assert r.status_code == 413 and list((tmp_path / "up").iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores folder permissions")
def test_an_upload_that_cannot_be_written_is_a_507_and_leaves_no_file(tmp_path: Path) -> None:
    c = _client(tmp_path, analyze_fn=lambda *a: None)
    up = tmp_path / "up"
    up.mkdir()
    up.chmod(0o500)  # readable, not writable
    try:
        r = c.post("/api/jobs", files={"file": ("a.mp4", b"x" * 100, "video/mp4")})
        assert r.status_code == 507 and "yazılamadı" in r.json()["detail"]
        assert list(up.iterdir()) == []
    finally:
        up.chmod(0o700)


def test_a_failed_analysis_reports_its_error_without_the_servers_folders(tmp_path: Path) -> None:
    def boom(video: Path, out: Path, progress) -> None:
        raise PermissionError(f"[Errno 13] Permission denied: '{out / 'report.json'}' (reading {video})")

    c = _client(tmp_path, analyze_fn=boom)
    job = _wait(c, c.post("/api/jobs", files={"file": ("maç 1.mp4", b"x", "video/mp4")}).json()["job_id"])
    assert job["status"] == "error" and job["error"].startswith("PermissionError: [Errno 13] Permission denied: '")
    assert str(tmp_path) not in job["error"] and "/up/" not in job["error"]
    run_id = job["error"].split("'")[1].split("/")[0]
    assert run_id.startswith("ma__1-") and job["error"].endswith("(reading maç 1.mp4)")

    def oom(video: Path, out: Path, progress) -> None:
        raise RuntimeError("cuda out of memory")

    c = _client(tmp_path / "b", analyze_fn=oom)
    job = _wait(c, c.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}).json()["job_id"])
    assert job["error"] == "RuntimeError: cuda out of memory"  # a message without paths is kept as it is


def test_an_upload_that_fails_mid_write_is_removed(tmp_path: Path, monkeypatch) -> None:
    real_open = open

    class DiskFull:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def write(self, b: bytes) -> int:
            self.f.write(b[:10])  # part of it lands on disk, then the disk is full
            raise OSError(28, "No space left on device")

    def fake_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return DiskFull(f) if mode == "wb" else f

    monkeypatch.setattr(app_module, "open", fake_open, raising=False)  # the module's name lookup finds it first
    c = _client(tmp_path, analyze_fn=lambda *a: None)
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"x" * 100, "video/mp4")})
    assert r.status_code == 507
    assert list((tmp_path / "up").iterdir()) == []  # the partial file is gone
