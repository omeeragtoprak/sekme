"""Review Room backend: runs, media (with HTTP Range), reviews audit log, analysis jobs.

Security posture: binds to localhost by default; run ids and media names
are validated against a strict pattern / allow-list and resolved paths must
stay inside the runs directory; uploads are size-limited, extension-checked
and stored under generated names. Reviews are an append-only JSONL audit
log that snapshots the system call next to the human decision.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from courtvision import __version__
from courtvision.eval.labels import evaluate, px_scale_cm, summarize, world_from_px

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MEDIA = {"classic": "classic.mp4", "pro": "pro.mp4"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
MAX_UPLOAD_BYTES = 2 * 1024**3

AnalyzeFn = Callable[[Path, Path, Callable[[str, float], None]], object]


class ReviewIn(BaseModel):
    flight_id: int = Field(ge=0)
    decision: Literal["confirm", "overturn_in", "overturn_out", "unsure"]
    reviewer: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


class LabelIn(BaseModel):
    """A reviewer's click on the ball's ground contact at a calibrated frame (image pixels)."""

    flight_id: int = Field(ge=0)
    frame: int = Field(ge=0)
    px: tuple[float, float]
    labeler: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"] = "queued"
    stage: str = "queued"
    progress: float = 0.0
    run_id: str | None = None
    error: str | None = None


def media_path(run: Path, name: str) -> Path | None:
    if name == "source":
        for p in sorted(run.glob("source.*")):
            if p.suffix.lower() in VIDEO_EXT and p.resolve().is_file():
                return p
        return None
    p = run / MEDIA[name]
    return p if p.is_file() else None


def link_source(video: Path, run: Path) -> None:
    """Expose the analysed video inside the run folder (symlink; copy when links are unsupported)."""
    dest = run / f"source{video.suffix.lower()}"
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(video.resolve())
    except OSError:
        shutil.copy2(video, dest)


class _WebFiles(StaticFiles):
    """Built web app. index.html names the hashed bundles, so it must be revalidated after a rebuild."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[no-untyped-def]
        resp = super().file_response(full_path, stat_result, scope, status_code)
        if str(full_path).endswith(".html"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(runs_dir: Path, uploads_dir: Path | None = None, web_dist: Path | None = None,
               analyze_fn: AnalyzeFn | None = None) -> FastAPI:
    runs_root = runs_dir.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    uploads = (uploads_dir or runs_root.parent / "data" / "uploads").resolve()
    jobs: dict[str, Job] = {}
    review_lock = threading.Lock()
    app = FastAPI(title="SEKME Review Room", version=__version__)

    def run_dir(run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise HTTPException(400, "geçersiz analiz kimliği")
        d = (runs_root / run_id).resolve()
        if d.parent != runs_root or not (d / "report.json").is_file():
            raise HTTPException(404, "analiz bulunamadı")
        return d

    def load_json(path: Path) -> dict:
        return json.loads(path.read_text())

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        out = []
        for d in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir() or not RUN_ID.match(d.name) or not (d / "report.json").is_file():
                continue
            try:
                r = load_json(d / "report.json")
            except (OSError, json.JSONDecodeError):
                continue
            cal = r.get("calibration", {})
            out.append({
                "id": d.name, "video": r["video"]["file"], "created": r.get("created", ""),
                "duration_s": r["video"]["duration_s"], "n_flights": len(r.get("flights", [])),
                "n_calls": sum(1 for f in r.get("flights", []) if f.get("call")),
                "calibrated_ratio": cal.get("valid_frames", 0) / max(cal.get("total_frames", 1), 1),
                "media": [m for m in ("source", "classic", "pro") if media_path(d, m) is not None],
            })
        return out

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> JSONResponse:
        return JSONResponse(load_json(run_dir(run_id) / "report.json"))

    @app.get("/api/runs/{run_id}/frames")
    def get_frames(run_id: str) -> JSONResponse:
        p = run_dir(run_id) / "frames.json"
        if not p.is_file():
            raise HTTPException(404, "kare verisi bulunamadı")
        return JSONResponse(load_json(p))

    @app.get("/api/runs/{run_id}/media/{name}")
    def get_media(run_id: str, name: str) -> FileResponse:
        if name not in ("source", *MEDIA):
            raise HTTPException(404, "bilinmeyen video türü")
        p = media_path(run_dir(run_id), name)
        if p is None:
            raise HTTPException(404, "bu video bu analizde yok")
        return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id}/reviews")
    def get_reviews(run_id: str) -> list[dict]:
        p = run_dir(run_id) / "reviews.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    @app.post("/api/runs/{run_id}/reviews", status_code=201)
    def add_review(run_id: str, body: ReviewIn) -> dict:
        d = run_dir(run_id)
        report = load_json(d / "report.json")
        flight = next((f for f in report.get("flights", []) if f.get("id") == body.flight_id), None)
        if flight is None:
            raise HTTPException(404, "vuruş bulunamadı")
        call = flight.get("call")
        if call is None:
            raise HTTPException(409, "bu vuruşun incelenecek bir sistem kararı yok")
        record = {
            **body.model_dump(), "id": uuid.uuid4().hex, "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "system_call": {k: call[k] for k in ("verdict", "p_in", "margin_m", "sigma_m", "line")},
            "report_version": report.get("version"),
            "models_sha256": {k: v.get("sha256") for k, v in report.get("models", {}).items()},
        }
        with review_lock, open(d / "reviews.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def read_labels(d: Path) -> list[dict]:
        p = d / "labels.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    @app.get("/api/runs/{run_id}/labels")
    def get_labels(run_id: str) -> list[dict]:
        return read_labels(run_dir(run_id))

    @app.get("/api/runs/{run_id}/labels/summary")
    def get_label_summary(run_id: str) -> dict:
        return summarize(read_labels(run_dir(run_id)))

    @app.post("/api/runs/{run_id}/labels", status_code=201)
    def add_label(run_id: str, body: LabelIn) -> dict:
        d = run_dir(run_id)
        report = load_json(d / "report.json")
        flight = next((f for f in report.get("flights", []) if f.get("id") == body.flight_id), None)
        if flight is None:
            raise HTTPException(404, "vuruş bulunamadı")
        call = flight.get("call")
        if call is None:
            raise HTTPException(409, "bu vuruşun karşılaştırılacak bir sistem kararı yok")
        frames = d / "frames.json"
        H = load_json(frames).get("H", []) if frames.is_file() else []
        if body.frame >= len(H) or H[body.frame] is None:
            raise HTTPException(409, "bu kare kalibre değil; kalibre bir karede işaretleyin")
        w, h = report["video"].get("width", float("inf")), report["video"].get("height", float("inf"))
        if not (0 <= body.px[0] < w and 0 <= body.px[1] < h):
            raise HTTPException(422, "işaretlenen nokta görüntünün dışında")
        xy = world_from_px(H[body.frame], body.px)
        record = {
            **body.model_dump(), "id": uuid.uuid4().hex, "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "world_xy": [round(float(v), 4) for v in xy], "px_cm": round(px_scale_cm(H[body.frame], body.px), 2),
            "eval": evaluate(xy, call),
            "system_call": {k: call[k] for k in ("verdict", "p_in", "margin_m", "sigma_m", "line", "zone", "x", "y")},
            "report_version": report.get("version"),
            "models_sha256": {k: v.get("sha256") for k, v in report.get("models", {}).items()},
        }
        with review_lock, open(d / "labels.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    @app.post("/api/jobs", status_code=202)
    def create_job(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI parameter idiom
        if analyze_fn is None:
            raise HTTPException(503, "bu sunucuda analiz başlatma kapalı")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(415, f"desteklenmeyen dosya türü; izin verilenler: {', '.join(sorted(VIDEO_EXT))}")
        uploads.mkdir(parents=True, exist_ok=True)
        job = Job(id=uuid.uuid4().hex[:12])
        dest = uploads / f"{job.id}{ext}"
        size = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    break
                out.write(chunk)
        if size > MAX_UPLOAD_BYTES:
            dest.unlink(missing_ok=True)
            raise HTTPException(413, "dosya çok büyük")
        jobs[job.id] = job
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file.filename or "upload").stem)[:40].lstrip("._-") or "upload"
        run_id = f"{stem}-{job.id}"

        def work() -> None:
            job.status = "running"
            try:
                def prog(stage: str, frac: float) -> None:
                    job.stage, job.progress = stage, round(float(frac), 3)

                analyze_fn(dest, runs_root / run_id, prog)
                job.run_id, job.status, job.stage, job.progress = run_id, "done", "done", 1.0
            except Exception as exc:  # reported to the client through the job status
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=work, daemon=True).start()
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "iş bulunamadı")
        return job.model_dump()

    dist = web_dist if web_dist is not None else Path(__file__).resolve().parents[3] / "web" / "dist"
    if dist.is_dir():
        app.mount("/", _WebFiles(directory=dist, html=True), name="web")
    return app
