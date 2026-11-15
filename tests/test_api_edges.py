"""The Review Room API (api/app.py) at its edges: upload limits, links out of the runs folder, Range requests,
reviews, Host and Origin checks, display names and labels. Status codes and bodies checked by running the code
(gaps found by a test review)."""

from __future__ import annotations

import json
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import courtvision.api.app as app_mod
from courtvision.api.app import RUN_ID, create_app
from courtvision.errors import MissingModelError

CALL = {"verdict": "REVIEW", "p_in": 0.62, "margin_m": 0.01, "sigma_m": 0.04, "line": "far baseline",
        "zone": "far-singles", "x": 0.0, "y": 11.9, "cov": [[0.001, 0], [0, 0.0016]]}
LOCAL = "http://127.0.0.1"


def _run(root: Path, run_id: str = "clip1") -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({
        "video": {"file": "clip.mp4", "duration_s": 2.0, "width": 1280, "height": 720},
        "flights": [{"id": 0, "call": None}, {"id": 1, "call": CALL}]}))
    (d / "frames.json").write_text(json.dumps({"fps": 30, "n": 60}))
    (d / "pro.mp4").write_bytes(bytes(range(256)) * 40)  # 10240 bytes
    return d


def _upload_app(tmp_path: Path, analyze_fn=None, **kw) -> TestClient:
    return TestClient(create_app(tmp_path / "runs", uploads_dir=tmp_path / "up", web_dist=tmp_path / "x",
                                 analyze_fn=analyze_fn, **kw), base_url=LOCAL)


def _wait(c: TestClient, job_id: str) -> dict:
    for _ in range(100):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(job)


def _noop(video: Path, out: Path, progress) -> None:
    out.mkdir(parents=True, exist_ok=True)


# --- 1. uploads: size limit, free disk, no analyzer ------------------------------------------------------------

def test_an_oversize_upload_is_refused_with_413_and_nothing_is_kept(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 1000)
    c = _upload_app(tmp_path, _noop)
    ok = c.post("/api/jobs", files={"file": ("a.mp4", b"x" * 1000, "video/mp4")})
    assert ok.status_code == 202  # exactly the limit is allowed
    _wait(c, ok.json()["job_id"])
    kept = sorted(p.name for p in (tmp_path / "up").iterdir())
    r = c.post("/api/jobs", files={"file": ("b.mp4", b"x" * 1001, "video/mp4")})
    assert r.status_code == 413 and r.json() == {"detail": "dosya çok büyük"}
    assert sorted(p.name for p in (tmp_path / "up").iterdir()) == kept  # the partial file was removed


def test_an_upload_on_a_nearly_full_disk_is_refused_with_507_before_anything_is_written(
        tmp_path: Path, monkeypatch) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(app_mod.shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 1, app_mod.MIN_FREE_BYTES - 1))
    called: list[Path] = []
    c = _upload_app(tmp_path, lambda v, o, p: called.append(v))
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 507 and r.json() == {"detail": "diskte yer yok"}
    assert list((tmp_path / "up").iterdir()) == [] and called == []
    monkeypatch.setattr(app_mod.shutil, "disk_usage", lambda p: usage(10**12, 0, app_mod.MIN_FREE_BYTES))
    assert c.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 202  # exactly enough


def test_uploads_are_off_without_an_analyzer_even_for_a_valid_video(tmp_path: Path) -> None:
    c = _upload_app(tmp_path, None)
    r = c.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 503 and r.json() == {"detail": "bu sunucuda analiz başlatma kapalı"}
    assert not (tmp_path / "up").exists()
    assert c.get("/api/jobs/unknown").status_code == 404


@pytest.mark.parametrize("name", ["evil.exe", "movie", ".mp4", "a.mp4.sh", "a.mp4 ", "x.MP4.txt"])
def test_unsupported_or_extensionless_upload_names_get_415_and_leave_no_file(tmp_path: Path, name: str) -> None:
    c = _upload_app(tmp_path, _noop)
    r = c.post("/api/jobs", files={"file": (name, b"x", "video/mp4")})
    assert r.status_code == 415, name
    assert not (tmp_path / "up").exists() or list((tmp_path / "up").iterdir()) == []


@pytest.mark.parametrize("name", ["../../etc/passwd.mp4", "..\\..\\x.MOV", "ma\u00e7 1.WebM", "....mp4",
                                  "a" * 300 + ".mp4", "\x00\x07.mp4"])
def test_upload_names_never_escape_the_upload_dir_and_give_a_valid_run_id(tmp_path: Path, name: str) -> None:
    seen: dict = {}

    def fake(video: Path, out: Path, progress) -> None:
        seen["video"], seen["out"] = video, out
        out.mkdir(parents=True)

    c = _upload_app(tmp_path, fake)
    r = c.post("/api/jobs", files={"file": (name, b"x", "video/mp4")})
    assert r.status_code == 202, r.text
    job = _wait(c, r.json()["job_id"])
    assert job["status"] == "done", job
    assert RUN_ID.match(job["run_id"])
    assert seen["video"].parent == (tmp_path / "up").resolve()
    assert seen["out"].parent == (tmp_path / "runs").resolve() and seen["out"].name == job["run_id"]


def test_job_errors_other_than_unreadable_video_are_reported_through_the_status(tmp_path: Path) -> None:
    def missing(video, out, progress):
        raise MissingModelError("model yok: sekme_ball.pt")

    def boom(video, out, progress):
        raise RuntimeError("cuda out of memory")

    for fn, expect in ((missing, "model yok: sekme_ball.pt"), (boom, "RuntimeError: cuda out of memory")):
        c = _upload_app(tmp_path / fn.__name__, fn)
        job = _wait(c, c.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")}).json()["job_id"])
        assert job["status"] == "error" and job["error"] == expect and job["run_id"] is None
        # the upload is kept after these two errors (only an unreadable video is dropped)
        assert len(list((tmp_path / fn.__name__ / "up").iterdir())) == 1


# --- 2. reviews --------------------------------------------------------------------------------------------------

def test_a_review_for_a_missing_flight_or_run_is_rejected_and_writes_nothing(tmp_path: Path) -> None:
    d = _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    body = {"flight_id": 1, "decision": "confirm", "reviewer": "r"}
    r = c.post("/api/runs/clip1/reviews", json={**body, "flight_id": 99})
    assert r.status_code == 404 and r.json() == {"detail": "vuruş bulunamadı"}
    assert c.post("/api/runs/clip1/reviews", json={**body, "flight_id": -1}).status_code == 422
    assert c.post("/api/runs/clip1/reviews", json={**body, "flight_id": "1"}).status_code == 201  # coerced, see below
    assert c.post("/api/runs/clip1/reviews", json={**body, "note": "n" * 501}).status_code == 422
    assert c.post("/api/runs/clip1/reviews", json={**body, "reviewer": "r" * 65}).status_code == 422
    assert c.post("/api/runs/nope/reviews", json=body).status_code == 404
    assert c.post("/api/runs/..%2Fclip1/reviews", json=body).status_code in (400, 404)
    assert len((d / "reviews.jsonl").read_text().splitlines()) == 1  # only the one accepted review


def test_concurrent_reviews_are_all_kept_as_whole_lines(tmp_path: Path) -> None:
    d = _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    note = "x" * 400  # long enough that interleaved writes would corrupt lines

    def post(i: int) -> int:
        body = {"flight_id": 1, "decision": "confirm", "reviewer": f"r{i}", "note": note}
        return c.post("/api/runs/clip1/reviews", json=body).status_code

    with ThreadPoolExecutor(16) as ex:
        codes = list(ex.map(post, range(64)))
    assert codes == [201] * 64
    lines = (d / "reviews.jsonl").read_text().splitlines()
    recs = [json.loads(line) for line in lines]  # every line is intact JSON
    assert len(recs) == 64 and len({r["id"] for r in recs}) == 64
    assert sorted(r["reviewer"] for r in recs) == sorted(f"r{i}" for i in range(64))


# --- 3. media: run-directory links, Range edge cases -------------------------------------------------------------

def test_a_run_folder_linked_from_outside_is_refused_by_every_route(tmp_path: Path) -> None:
    _run(tmp_path / "runs")
    outside = _run(tmp_path / "other")
    (tmp_path / "runs" / "linked").symlink_to(outside)
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    body = {"flight_id": 1, "decision": "confirm", "reviewer": "r"}
    for method, path in (("get", "report"), ("get", "frames"), ("get", "media/pro"), ("get", "reviews"),
                         ("get", "labels"), ("get", "serve-labels")):
        assert getattr(c, method)(f"/api/runs/linked/{path}").status_code == 404, path
    assert c.post("/api/runs/linked/reviews", json=body).status_code == 404
    assert not (outside / "reviews.jsonl").exists()  # nothing was written through the link


@pytest.mark.parametrize("rng,status,content_range,size", [
    ("bytes=5000-6000", 206, "bytes 5000-6000/10240", 1001),
    ("bytes=10230-", 206, "bytes 10230-10239/10240", 10),
    ("bytes=-10", 206, "bytes 10230-10239/10240", 10),
    ("bytes=0-99999", 206, "bytes 0-10239/10240", 10240),  # end past the file is clamped
    ("bytes=10240-", 416, "*/10240", 0),  # start at the end
    ("bytes=99999-", 416, "*/10240", 0),
])
def test_range_requests_are_clamped_or_refused_without_reading_past_the_file(
        tmp_path: Path, rng: str, status: int, content_range: str, size: int) -> None:
    _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    r = c.get("/api/runs/clip1/media/pro", headers={"Range": rng})
    assert r.status_code == status and r.headers.get("content-range") == content_range and len(r.content) == size
    if status == 206:
        assert r.content == (bytes(range(256)) * 40)[int(r.headers["content-range"].split()[1].split("-")[0]):][:size]


@pytest.mark.parametrize("rng", ["bytes=abc", "bytes=9-3", "items=0-5"])
def test_a_malformed_range_is_a_400_not_a_server_error(tmp_path: Path, rng: str) -> None:
    _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get("/api/runs/clip1/media/pro", headers={"Range": rng}).status_code == 400


@pytest.mark.parametrize("name", ["SOURCE", "Pro", "..", "pro.mp4", "%2e%2e%2freport.json", "source.mp4"])
def test_media_names_outside_the_allow_list_are_refused(tmp_path: Path, name: str) -> None:
    _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get(f"/api/runs/clip1/media/{name}").status_code == 404


def test_a_dangling_source_link_is_neither_served_nor_listed(tmp_path: Path) -> None:
    run = _run(tmp_path / "runs")
    (run / "source.mp4").symlink_to(tmp_path / "gone.mp4")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get("/api/runs/clip1/media/source").status_code == 404
    assert c.get("/api/runs").json()[0]["media"] == ["pro"]


def test_a_source_file_with_a_non_video_name_is_ignored_even_if_the_link_target_is_a_video(tmp_path: Path) -> None:
    run = _run(tmp_path / "runs")
    video = tmp_path / "m.mp4"
    video.write_bytes(b"\x00" * 10)
    (run / "source.txt").symlink_to(video)  # the link's own name must be a video name too
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    assert c.get("/api/runs/clip1/media/source").status_code == 404


# --- 4. Host / Origin --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["", "evil.example", "127.0.0.1.evil.example", "localhost@evil.example",
                                  "[::2]:8765", "127.0.0.2", "0.0.0.0:8765"])
def test_wrong_or_unparsable_hosts_are_refused_with_400_on_every_method(tmp_path: Path, host: str) -> None:
    _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    r = c.get("/api/health", headers={"host": host})
    assert r.status_code == 400 and r.json() == {"detail": "geçersiz sunucu adı"}
    assert c.post("/api/runs/clip1/reviews", headers={"host": host},
                  json={"flight_id": 1, "decision": "confirm", "reviewer": "r"}).status_code == 400


def test_host_matching_ignores_case_and_a_custom_allow_list_replaces_the_local_names(tmp_path: Path) -> None:
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x", allowed_hosts=["sekme.lan"]),
                   base_url="http://sekme.lan:8765")
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/health", headers={"host": "SEKME.LAN"}).status_code == 200
    assert c.get("/api/health", headers={"host": "127.0.0.1"}).status_code == 400
    assert c.get("/api/health", headers={"host": "localhost:8765"}).status_code == 400


def test_every_write_method_checks_the_origin_but_reads_and_option_requests_do_not(tmp_path: Path) -> None:
    _run(tmp_path / "runs")
    c = TestClient(create_app(tmp_path / "runs", uploads_dir=tmp_path / "up", web_dist=tmp_path / "x",
                              analyze_fn=_noop), base_url=LOCAL)
    evil = {"origin": "https://evil.example"}
    up = {"file": ("a.mp4", b"x", "video/mp4")}
    assert c.post("/api/jobs", files=up, headers=evil).status_code == 403  # an upload is a write too
    assert not (tmp_path / "up").exists()  # refused before anything is stored
    for method in ("put", "patch", "delete"):
        assert getattr(c, method)("/api/runs/clip1/reviews", headers=evil).status_code == 403
    assert c.get("/api/runs", headers=evil).status_code == 200  # reads carry no origin check
    assert c.options("/api/runs", headers=evil).status_code != 403
    # the origin must match case-insensitively, including the scheme-less host:port
    assert c.post("/api/runs/clip1/reviews", headers={"origin": "http://127.0.0.1".upper()},
                  json={"flight_id": 1, "decision": "confirm", "reviewer": "r"}).status_code == 201


# --- 5. display_name ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("", ""), ("a/b\\c/d.mp4", "d.mp4"), ("dir/", ""), ("C:\\", ""), ("..\\..\\x.mp4", "x.mp4"),
    ("a\x00b\x1fc\x7fd.mp4", "abcd.mp4"), ("\x1b[31mred\x1b[0m.mp4", "[31mred[0m.mp4"),
    ("x\u202e4pm.exe", "x4pm.exe"),  # a bidi override would show the name reordered: stripped
    ("tab\there.mp4", "tabhere.mp4"),
])
def test_display_name_edge_cases(raw: str, expect: str) -> None:
    assert app_mod.display_name(raw) == expect


def test_display_name_length_boundaries_count_characters() -> None:
    assert app_mod.display_name("a" * 80) == "a" * 80  # exactly 80 is kept whole
    cut = app_mod.display_name("a" * 81)
    assert cut == "a" * 79 + "…" and len(cut) == 80
    assert app_mod.display_name("ç" * 200).count("ç") == 79
    assert app_mod.display_name("x" * 100 + "/" + "y" * 5) == "yyyyy"  # length applies after the segment is taken


# --- 6. labels: image bounds ---------------------------------------------------------------------------------------

def test_a_label_outside_the_image_is_422_and_nothing_is_logged(tmp_path: Path) -> None:
    d = _run(tmp_path / "runs")
    H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    (d / "frames.json").write_text(json.dumps({"fps": 30, "n": 2, "H": [H, H]}))
    c = TestClient(create_app(tmp_path / "runs", web_dist=tmp_path / "x"), base_url=LOCAL)
    ok = {"flight_id": 1, "frame": 1, "px": [10, 10], "labeler": "a"}
    for px in ([-1, 10], [1280, 10], [10, 720], [10, -0.5]):
        r = c.post("/api/runs/clip1/labels", json={**ok, "px": px})
        assert r.status_code == 422, px
    assert not (d / "labels.jsonl").exists()
    assert c.post("/api/runs/clip1/labels", json={**ok, "px": [1279, 719]}).status_code == 201
    assert c.post("/api/runs/clip1/labels", json={**ok, "frame": 2}).status_code == 409  # past the frame table
