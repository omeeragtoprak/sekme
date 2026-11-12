"""Check the pipeline against what the broadcast itself shows (weak, independent ground truth).

* Serve speed: the stadium radar board shows each serve's speed a few seconds after the serve; each
  board update is matched to the single detected serve that precedes it by 1-6.5 s.
* Serve side: the score graphic marks the server. Players change ends after every odd game of a set
  (ITF Rule 10), so a set's g played games give (g + 1) // 2 changes; with the number of changes since
  the start of the match, every serve by the same player must come from the same side modulo 2.
* Names: one serve with a known server fixes which end each player started at, which names the near
  and far players wherever the score graphic is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.broadcast.scoreboard import BroadcastScan, ScoreReading, score_changes, speed_updates

MPH_TO_KMH = 1.609344


def to_kmh(value: float, unit: str | None) -> float | None:
    if unit == "mph":
        return value * MPH_TO_KMH
    if unit == "km/h":
        return float(value)
    return None


def end_changes(r: ScoreReading) -> int:
    """Ends changes since the start of the match, from the games of every set shown (tiebreaks aside)."""
    sets = list(zip(r.rows[0].games, r.rows[1].games, strict=False))
    return sum((a + b + 1) // 2 for a, b in sets)


def _flip(side: str) -> str:
    return "far" if side == "near" else "near"


@dataclass(frozen=True)
class Serve:
    flight_id: int
    t: float  # contact time, s
    side: str  # "near" | "far" (where the server stood, per the pipeline)
    speed_kmh: float | None


def reading_near(scores: list[ScoreReading], t: float, window: float = 8.0) -> ScoreReading | None:
    """The score graphic closest in time to ``t`` with a marked server (within ``window`` seconds)."""
    cands = [s for s in scores if abs(s.t - t) <= window and any(r.serving for r in s.rows)]
    return min(cands, key=lambda s: abs(s.t - t)) if cands else None


# The board changes ~2.5-3.5 s after the serve (tight pairs on the US Open clips). When the change itself is
# only known to within more than this, the value cannot be tied to one serve (it may follow an edited-out one).
MAX_UPDATE_WINDOW_S = 4.0


def check(scan: BroadcastScan, serves: list[Serve], lag: tuple[float, float] = (1.0, 6.5),
          measurable: list[tuple[float, float]] | None = None) -> dict:
    """JSON-ready summary: score changes, board updates, serve-speed pairs, server/side consistency.

    ``measurable`` are the time spans the pipeline could analyse (no cut, calibrated). A board update
    without a serve is a *missed serve* only when its whole search window lies inside one of them;
    otherwise the serve may have been edited out ("cut").
    """
    names = scan.names
    changes = score_changes(scan.scores)
    updates = speed_updates(scan.speeds)

    # Serve speed vs the radar board ------------------------------------------------------------
    pairs, unmatched_updates = [], []
    matched: set[int] = set()
    for u in updates:
        if u.previous is None:  # first value seen: the serve behind it is not in view
            continue
        lo, hi = u.t_before - lag[1], u.t_after - lag[0]
        cands = [s for s in serves if lo <= s.t <= hi and s.flight_id not in matched]
        board = to_kmh(u.value, scan.speed_unit)
        if u.t_after - u.t_before > MAX_UPDATE_WINDOW_S:
            unmatched_updates.append({"t": [round(u.t_before, 2), round(u.t_after, 2)], "board": u.value,
                                      "candidates": len(cands), "reason": "timing"})
            continue
        if len(cands) == 1 and board is not None and cands[0].speed_kmh is not None:
            s = cands[0]
            matched.add(s.flight_id)
            pairs.append({"flight_id": s.flight_id, "t": round(s.t, 2), "ours_kmh": round(s.speed_kmh, 1),
                          "board": u.value, "board_kmh": round(board, 1),
                          "lag_s": [round(u.t_before - s.t, 2), round(u.t_after - s.t, 2)],
                          "error_pct": round(100 * (s.speed_kmh - board) / board, 1)})
        else:
            inside = measurable is not None and any(a <= lo and hi <= b for a, b in measurable)
            reason = "ambiguous" if len(cands) > 1 else "missed" if inside else "cut"
            unmatched_updates.append({"t": [round(u.t_before, 2), round(u.t_after, 2)], "board": u.value,
                                      "candidates": len(cands), "reason": reason})
    errs = [p["error_pct"] for p in pairs]

    # Server marker vs detected side, normalised by the ends changes ---------------------------
    sides: list[dict] = []
    start_end: dict[str, str] = {}
    consistent = None
    for s in serves:
        r = reading_near(scan.scores, s.t)
        if r is None:
            continue
        name = next(row.name for row in r.rows if row.serving)
        k = end_changes(r)
        start = s.side if k % 2 == 0 else _flip(s.side)  # where this server stood at the start
        sides.append({"flight_id": s.flight_id, "server": name, "side": s.side, "end_changes": k,
                      "start_side": start})
    by_name: dict[str, set[str]] = {}
    for x in sides:
        by_name.setdefault(x["server"], set()).add(x["start_side"])
    if sides:
        consistent = all(len(v) == 1 for v in by_name.values()) and len(
            {next(iter(v)) for v in by_name.values()}) == len(by_name)
        if consistent:
            start_end = {n: next(iter(v)) for n, v in by_name.items()}
            if names and len(start_end) == 1:  # the other player started at the other end
                only = next(iter(start_end))
                start_end[names[1] if only == names[0] else names[0]] = _flip(start_end[only])

    return {
        "ocr": "apple-vision",
        "frames_read": scan.frames_read,
        "step_frames": scan.step_frames,
        "names": list(names) if names else None,
        "score_readings": len(scan.scores),
        "score_changes": [
            {"t": [round(c.t_before, 2), round(c.t_after, 2)],
             "before": [[list(r.games), r.points] for r in c.before.rows],
             "after": [[list(r.games), r.points] for r in c.after.rows],
             "winner": (c.after.rows[c.winner].name if c.winner is not None else None), "n_points": c.n_points}
            for c in changes
        ],
        "speed_unit": scan.speed_unit,
        "speed_updates": [{"t": [round(u.t_before, 2), round(u.t_after, 2)], "value": u.value,
                           "previous": u.previous} for u in updates],
        "serve_speed": {
            "pairs": pairs,
            "n": len(pairs),
            "mean_error_pct": round(sum(errs) / len(errs), 1) if errs else None,
            "mean_abs_error_pct": round(sum(abs(e) for e in errs) / len(errs), 1) if errs else None,
            "unmatched_updates": unmatched_updates,
        },
        "server_sides": {"serves": sides, "consistent": consistent, "start_end": start_end or None},
    }


def side_names(scan: BroadcastScan, start_end: dict[str, str] | None, t: float,
               window: float = 8.0) -> dict[str, str] | None:
    """{"near": name, "far": name} at time ``t`` from the start ends and the nearest score graphic."""
    if not start_end:
        return None
    cands = [s for s in scan.scores if abs(s.t - t) <= window]
    if not cands:
        return None
    r = min(cands, key=lambda s: abs(s.t - t))
    k = end_changes(r)
    out = {}
    for name, start in start_end.items():
        out[start if k % 2 == 0 else _flip(start)] = name
    return out if len(out) == 2 else None
