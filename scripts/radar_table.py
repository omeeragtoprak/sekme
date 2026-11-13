"""Fitted serve speed vs the stadium radar board, pooled over runs (each report's broadcast.serve_speed).

    .venv/bin/python scripts/radar_table.py runs/usopen2023_final_hd runs/usopen2025_final_hd ...

A report keeps a pair only when the board change is timed to within a few seconds of one fitted serve;
the other board changes are listed as unmatched with the reason (cut, timing).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    errors = []
    for run in map(Path, sys.argv[1:]):
        rep = json.loads((run / "report.json").read_text())
        ss = rep["broadcast"]["serve_speed"]
        pairs = [(p["flight_id"], p["ours_kmh"], p["board_kmh"], p["error_pct"]) for p in ss["pairs"]]
        unmatched = dict(Counter(u["reason"] for u in ss["unmatched_updates"]))
        models = " ".join(m["file"] for m in rep["models"].values())
        print(f"{run.name} | models: {models} | pairs: {pairs} | unmatched: {unmatched}")
        errors += [p[3] for p in pairs]
    e = np.array(errors)
    if len(e):
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
        print(f"ALL n={len(e)} mean {e.mean():+.1f} % (SE {se:.1f}), mean |error| {np.abs(e).mean():.1f} %, "
              f"SD {e.std(ddof=1) if len(e) > 1 else float('nan'):.1f} %")


if __name__ == "__main__":
    main()
