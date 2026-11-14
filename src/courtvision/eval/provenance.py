"""Which code made an evaluation run: git helpers shared by the evaluation scripts (scripts/eval_six_videos.py,
scripts/eval_spot.py). Not used by the analysis itself, so changing it never makes a run stale."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def head_sha() -> str:
    r = git("rev-parse", "HEAD")
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(f"git rev-parse HEAD failed: {r.stderr.strip()}")
    return r.stdout.strip()


def clean(paths: list[str]) -> bool:
    """No uncommitted change (staged, unstaged or untracked) under these paths; a failing git is not clean."""
    r = git("status", "--porcelain", "--", *paths)
    return r.returncode == 0 and r.stdout.strip() == ""


def tracked_and_clean(files: list[Path]) -> bool:
    """Every file is tracked by git and unchanged since its commit."""
    return bool(files) and git("ls-files", "--error-unmatch", "--", *map(str, files)).returncode == 0 and clean(
        [str(f) for f in files])


def analysis_code() -> list[str]:
    """The source files ``sekme analyze`` can load: cli.py and the import closure of what its analyze command
    imports (every import statement, also those inside functions, and the packages' __init__ files). Only these
    decide whether two runs come from the same code: a commit to the API, the web app or a scorer does not make
    a run stale."""
    src = ROOT / "src"

    def file_of(mod: str) -> Path | None:
        p = src / Path(*mod.split("."))
        return next((c for c in (p.with_suffix(".py"), p / "__init__.py") if c.exists()), None)

    files = {src / "courtvision" / "cli.py"}
    todo = ["courtvision.pipeline", "courtvision.events.serves"]  # what cli._analyze imports
    seen: set[str] = set()
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        f = file_of(mod)
        if f is None:
            continue
        files.add(f)
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]  # parent packages run their __init__
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                raise SystemExit(f"{f}: relative import; analysis_code() resolves absolute imports only")
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names if a.name.startswith("courtvision")]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("courtvision"):
                todo.append(node.module)
                todo += [f"{node.module}.{a.name}" for a in node.names]  # "from pkg import module"
    return sorted(str(f.relative_to(ROOT)) for f in files)


def same_code(run: Path) -> bool:
    """The run was made from a commit whose analysis code equals HEAD's (its code_sha.txt)."""
    marker = run / "code_sha.txt"
    if not marker.exists():
        return False
    return git("diff", "--quiet", marker.read_text().strip(), "HEAD", "--", *analysis_code()).returncode == 0


def code_unchanged_since(sha: str, code: list[str]) -> bool:
    """Still committed and equal to ``sha``: no edit or commit to the analysis code since a run started."""
    return clean(code) and git("diff", "--quiet", sha, "HEAD", "--", *code).returncode == 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
