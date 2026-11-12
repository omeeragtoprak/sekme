"""Score graphic and serve-speed board, read every half second and turned into events.

Parsing works on recognised text boxes only, so it is testable without Vision:

* score rows: two name boxes stacked at the same left edge, each followed by number tokens in its
  band; tokens are aligned into columns across the two rows, the last column is the point score when
  it holds point values (15/30/40/AD) — earlier columns are games per set;
* serve marker: a yellow ball logo or a white arrow next to one row's name (pixel test) or a marker
  glyph recognised in the name text;
* speed board: a number next to a "SERVE SPEED"/"MPH" label; the unit is taken from the text.

Readings only count when the same value is read twice in a row (OCR noise is per frame), and
changes between stable readings become events: a score change (with the point winner when a single
point separates the two states) or a new serve speed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from courtvision.broadcast import ocr
from courtvision.broadcast.ocr import TextBox

POINT_RANK = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
_NUM_TOKEN = re.compile(r"^(?:\d{1,2}|AD|A)$", re.IGNORECASE)
_NAME = re.compile(r"^(?:(\d{1,2})\s+)?([^\W\d_][^\W\d_'\-. ]*(?:[ '\-.][^\W\d_]+)*)(?:\s+([A-Z]{3}))?$")
_MARKERS = "▸►▶•●◀◂"


@dataclass(frozen=True)
class ScoreRow:
    name: str
    games: tuple[int, ...]  # per set, left to right
    points: str | None  # "0".."40", "AD", or None when not shown
    serving: bool | None = None


@dataclass(frozen=True)
class ScoreReading:
    frame: int
    t: float
    rows: tuple[ScoreRow, ScoreRow]

    def key(self) -> tuple:
        return tuple((r.games, r.points) for r in self.rows)


@dataclass(frozen=True)
class SpeedReading:
    frame: int
    t: float
    value: int


@dataclass(frozen=True)
class ScoreChange:
    t_before: float  # last time the old score was seen
    t_after: float  # first time the new score was seen
    before: ScoreReading
    after: ScoreReading
    winner: int | None  # row index (0 top, 1 bottom) of the point winner, if one point separates them
    n_points: int | None  # points played between the two readings, when it can be counted


@dataclass(frozen=True)
class SpeedUpdate:
    t_before: float
    t_after: float
    value: int
    previous: int | None


@dataclass
class BroadcastScan:
    step_frames: int
    frames_read: int
    scores: list[ScoreReading] = field(default_factory=list)
    speeds: list[SpeedReading] = field(default_factory=list)
    speed_unit: str | None = None  # "mph" | "km/h" | None (unknown: no speed checks)
    names: tuple[str, str] | None = None


# ---------------------------------------------------------------------------------------------
# Parsing one frame


def _split_tokens(b: TextBox) -> list[TextBox]:
    """Split a box like "5 40" into per-token boxes with proportional x positions."""
    parts = b.text.split()
    if len(parts) <= 1:
        return [b]
    n = len(b.text)
    out, pos = [], 0
    for p in parts:
        i = b.text.index(p, pos)
        pos = i + len(p)
        out.append(TextBox(p, b.conf, b.x + b.w * i / n, b.y, b.w * len(p) / n, b.h))
    return out


def _name_parts(text: str) -> tuple[bool, str]:
    """(marker glyph present, cleaned name) for a candidate name box; name is "" when not a name."""
    t = text.strip()
    marked = bool(t) and (t[0] in _MARKERS or t[-1] in _MARKERS)
    t = t.strip(_MARKERS + " ")
    m = _NAME.match(t)
    if not m:
        return marked, ""
    name = m.group(2).strip()
    words = name.split()
    if len(words) >= 2 and len(words[-1]) == 3 and words[-1].isupper():
        name = " ".join(words[:-1])  # trailing country code (e.g. "DJOKOVIC SRB")
    letters = sum(ch.isalpha() for ch in name)
    return marked, name.upper() if letters >= 3 else ""


def _marker_pixels(img: np.ndarray | None, row: TextBox, gap_x1: float, boxes: list[TextBox]) -> int:
    """Salient non-text pixels (yellow ball logo or white arrow) beside a row's name."""
    if img is None:
        return 0
    H, W = img.shape[:2]
    y0, y1 = int(row.y * H), int((row.y + row.h) * H) + 1
    regions = [(row.x1, gap_x1), (row.x - 1.6 * row.h * H / W, row.x)]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 110) & (hsv[..., 2] > 140)
    white = (hsv[..., 1] < 50) & (hsv[..., 2] > 205)
    mask = yellow | white
    text = np.zeros(mask.shape, bool)
    for b in boxes:  # recognised glyphs are not markers
        text[int(b.y * H) - 1:int((b.y + b.h) * H) + 2, int(b.x * W) - 1:int(b.x1 * W) + 2] = True
    n = 0
    for xa, xb in regions:
        a, b = max(0, int(xa * W)), min(W, int(xb * W))
        if b > a:
            n += int((mask[y0:y1, a:b] & ~text[y0:y1, a:b]).sum())
    return n


def parse_scoreboard(boxes: list[TextBox], img: np.ndarray | None, frame: int, t: float) -> ScoreReading | None:
    """The two score rows in one frame's text boxes, or None when no score graphic is visible."""
    names = [(b, *_name_parts(b.text)) for b in boxes]
    names = [(b, mk, nm) for b, mk, nm in names if nm]
    nums = [s for b in boxes for s in _split_tokens(b) if _NUM_TOKEN.match(s.text)]
    best: tuple | None = None
    for i, (a, mka, _) in enumerate(names):
        for b, mkb, _ in names[i + 1:]:
            top, bot = (a, b) if a.y < b.y else (b, a)
            if abs(top.x - bot.x) > 0.02 or not 0.6 * top.h < bot.y - top.y < 3.0 * top.h:
                continue
            if abs(top.h - bot.h) > 0.5 * max(top.h, bot.h):
                continue
            rows = []
            for r in (top, bot):
                band = [n for n in nums if abs(n.cy - r.cy) < 0.6 * r.h and r.x1 - 0.005 < n.x < r.x + 0.45]
                rows.append(sorted(band, key=lambda n: n.x))
            if not rows[0] and not rows[1]:
                continue
            score = len(rows[0]) + len(rows[1])
            if best is None or score > best[0]:
                marks = (mka, mkb) if top is a else (mkb, mka)
                best = (score, top, bot, rows, marks)
    if best is None:
        return None
    _, top, bot, rows, marks = best
    # Align number tokens of both rows into columns.
    # A column holds at most one token per row; tokens closer than half a row height share a column.
    tol = 0.5 * min(top.h, bot.h)
    cols: list[list[tuple[int, TextBox]]] = []
    for ri, toks in enumerate(rows):
        for tk in toks:
            for c in cols:
                if all(r != ri for r, _ in c) and abs(c[0][1].cx - tk.cx) < tol:
                    c.append((ri, tk))
                    break
            else:
                cols.append([(ri, tk)])
    cols.sort(key=lambda c: min(tk.cx for _, tk in c))
    vals = [{ri: tk.text.upper() for ri, tk in c} for c in cols]
    last = vals[-1] if vals else {}
    is_points = bool(last) and all(v in POINT_RANK for v in last.values()) and (
        any(v in ("15", "30", "40", "AD", "A") for v in last.values()) or len(vals) >= 2)
    point_col = len(vals) - 1 if is_points else None
    out = []
    for ri, box in enumerate((top, bot)):
        games = tuple(int(v[ri]) for k, v in enumerate(vals) if k != point_col and ri in v and v[ri].isdigit())
        pts = vals[point_col].get(ri) if point_col is not None else None
        if pts == "A":
            pts = "AD"
        if point_col is not None and pts is None and vals[point_col].get(1 - ri) == "AD":
            pts = "40"  # some graphics show only the advantage
        out.append((_name_parts(box.text)[1], games, pts))
    if len(out[0][1]) != len(out[1][1]):
        return None  # a set column was missed in one row: the reading is incomplete
    gap_x1 = [min((tk.x for tk in rows[ri]), default=box.x1 + 0.1) for ri, box in enumerate((top, bot))]
    px = [_marker_pixels(img, box, gap_x1[ri], boxes) for ri, box in enumerate((top, bot))]
    serving: list[bool | None] = [None, None]
    if marks[0] != marks[1]:
        serving = [marks[0], marks[1]]
    elif max(px) >= 12 and max(px) >= 3 * min(px):
        serving = [px[0] > px[1], px[1] > px[0]]
    return ScoreReading(frame, t, (ScoreRow(*out[0], serving[0]), ScoreRow(*out[1], serving[1])))


_SPEED_LABELS = ("SERVESPEED", "SERVE", "SPEED", "MPH", "KMH")


def _is_speed_label(text: str) -> bool:
    t = re.sub(r"[^A-Z]", "", text.upper())
    return len(t) >= 3 and any(SequenceMatcher(None, t, lab).ratio() >= 0.7 for lab in _SPEED_LABELS)


SERVE_SPEED_RANGE = {"mph": (50, 165), "km/h": (80, 265)}  # plausible board values (record ~263 km/h)
# Cyrillic look-alikes Vision sometimes returns on Latin board fonts ("116MРL" for "116MPH").
_HOMOGLYPHS = str.maketrans({"Р": "P", "Н": "H", "М": "M", "К": "K", "А": "A", "Е": "E", "Т": "T", "О": "O"})
# A number with its unit glued on, read as one box; the last letter of MPH is often misread (L, N).
_GLUED = re.compile(r"^\s*(\d{2,3})\s*(MP[HLN]?|KM/?[HLN]?)\s*$")


def parse_speed(boxes: list[TextBox], frame: int, t: float) -> tuple[SpeedReading | None, str | None]:
    """A serve-speed board reading and the unit printed near it (None when not printed/legible)."""
    glued = []
    for b in boxes:
        m = _GLUED.match(b.text.upper().translate(_HOMOGLYPHS))
        if m:
            v, u = int(m.group(1)), "mph" if m.group(2).startswith("MP") else "km/h"
            if SERVE_SPEED_RANGE[u][0] <= v <= SERVE_SPEED_RANGE[u][1]:  # ads and captions carry "20 KM" too
                glued.append((b.conf, v, u))
    if glued:  # the unit printed with the number labels it
        conf, value, unit = max(glued)
        return SpeedReading(frame, t, value), unit
    labels = [b for b in boxes if _is_speed_label(b.text)]
    if not labels:
        return None, None
    unit = None
    near = [b for b in boxes if any(abs(b.cx - lb.cx) < 0.06 and abs(b.cy - lb.cy) < 0.06 for lb in labels)]
    for b in near:
        u = re.sub(r"[^A-Z/]", "", b.text.upper())
        if "MPH" in u:
            unit = "mph"
        elif "KM" in u:
            unit = "km/h"
    cands = []
    for b in near:
        m = re.match(r"^\s*(\d{2,3})(?!\d)", b.text)
        if m and b not in labels:
            cands.append((b.conf, int(m.group(1))))
    if not cands:
        return None, unit
    conf, value = max(cands)
    return SpeedReading(frame, t, value), unit


# ---------------------------------------------------------------------------------------------
# Scanning a video and turning readings into events


def scan_video(path: str | Path, fps: float, step_s: float = 0.5, max_frames: int | None = None) -> BroadcastScan:
    """OCR every ``step_s`` seconds of the video (Vision must be available)."""
    step = max(1, int(round(step_s * fps)))
    scan = BroadcastScan(step_frames=step, frames_read=0)
    if not ocr.available():
        return scan
    cap = cv2.VideoCapture(str(path))
    units: Counter[str] = Counter()
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok or (max_frames is not None and i >= max_frames):
                break
            if i % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    boxes = ocr.recognize(img)
                    scan.frames_read += 1
                    sr = parse_scoreboard(boxes, img, i, i / fps)
                    if sr is not None:
                        scan.scores.append(sr)
                    sp, unit = parse_speed(boxes, i, i / fps)
                    if sp is not None:
                        scan.speeds.append(sp)
                    if unit:
                        units[unit] += 1
            i += 1
    finally:
        cap.release()
    scan.speed_unit = units.most_common(1)[0][0] if units else None
    scan.names = majority_names(scan.scores)
    return scan


def majority_names(scores: list[ScoreReading]) -> tuple[str, str] | None:
    if not scores:
        return None
    top = Counter(s.rows[0].name for s in scores).most_common(1)[0][0]
    bot = Counter(s.rows[1].name for s in scores).most_common(1)[0][0]
    return top, bot


def _stable(seq: list, key, max_gap: float) -> list[tuple[float, float, object]]:
    """Runs of equal keys confirmed by at least two consecutive readings: (t_first, t_last, item)."""
    runs: list[list] = []
    for it in seq:
        k = key(it)
        if runs and runs[-1][2] == k and it.t - runs[-1][1] <= max_gap:
            runs[-1][1] = it.t
            runs[-1][3] += 1
        else:
            runs.append([it.t, it.t, k, 1, it])
    return [(r[0], r[1], r[4]) for r in runs if r[3] >= 2]


def _point_step(a: ScoreReading, b: ScoreReading) -> tuple[int | None, int | None]:
    """(winner row, points played) between two score states, when it can be determined."""
    ga = [r.games[-1] if r.games else 0 for r in a.rows]
    gb = [r.games[-1] if r.games else 0 for r in b.rows]
    if [len(r.games) for r in a.rows] != [len(r.games) for r in b.rows]:
        return None, None  # a set ended in between (or the graphic changed layout)
    pa = [POINT_RANK.get(r.points or "0") for r in a.rows]
    pb = [POINT_RANK.get(r.points or "0") for r in b.rows]
    if None in pa or None in pb:
        return None, None  # tiebreak points or unreadable
    dg = [gb[0] - ga[0], gb[1] - ga[1]]
    if dg == [0, 0]:
        if pb[0] == pa[0] == 4 or pb[1] == pa[1] == 4:
            return None, None
        # Deuce: advantage lost counts as a point for the other player.
        if pa == [4, 3] and pb == [3, 3]:
            return 1, 1
        if pa == [3, 4] and pb == [3, 3]:
            return 0, 1
        d = [pb[0] - pa[0], pb[1] - pa[1]]
        if d in ([1, 0], [0, 1]):
            return (0 if d[0] else 1), 1
        if min(d) >= 0 and sum(d) > 0:
            return None, sum(d)
        return None, None
    if sorted(dg) == [0, 1] and pb == [0, 0]:
        w = 0 if dg[0] == 1 else 1
        # One point won the game if the winner was on game point before.
        if (pa[w] == 3 and pa[1 - w] <= 2) or pa[w] == 4:
            return w, 1
        return None, None
    return None, None


def score_changes(scores: list[ScoreReading], max_gap: float = 1.2) -> list[ScoreChange]:
    stable = _stable(scores, ScoreReading.key, max_gap)
    out = []
    for (_, t1a, a), (t0b, _, b) in zip(stable, stable[1:], strict=False):
        if a.key() == b.key():
            continue
        w, n = _point_step(a, b)
        out.append(ScoreChange(t1a, t0b, a, b, w, n))
    return out


def speed_updates(speeds: list[SpeedReading], max_gap: float = 1.2) -> list[SpeedUpdate]:
    stable = _stable(speeds, lambda s: s.value, max_gap)
    out = []
    prev = None
    for t0, t1, s in stable:
        if prev is None or s.value != prev[2].value:
            out.append(SpeedUpdate(prev[1] if prev else t0, t0, s.value, prev[2].value if prev else None))
        prev = (t0, t1, s)
    return out
