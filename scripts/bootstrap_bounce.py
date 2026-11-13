"""Is one model set's median bounce distance really lower than another's? Clip bootstrap on TrackNet games 8-10.

    .venv/bin/python scripts/bootstrap_bounce.py

Resamples the 30 clips with replacement (bounces in one clip share a camera, players and a calibration, so they
are not independent), recomputes each set's median distance to the reference contact and reports the observed
difference with a 95 % percentile interval. Inputs: eval_tracknet.py results files under data/tracknet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "tracknet"
PAIRS = [("test_ball_v3.json", "results.json", "v3 - WASB, raw reference"),
         ("centred_v3.json", "centred_wasb.json", "v3 - WASB, centred reference"),
         ("test_ball_v3.json", "test_ball_v2.json", "v3 - v2, raw reference"),
         ("centred_v3.json", "centred_v2.json", "v3 - v2, centred reference"),
         # v4 appended after v3 so the shared random stream, and v3's published intervals, stay the same
         ("test_ball_v4.json", "results.json", "v4 - WASB, raw reference"),
         ("centred_v4.json", "centred_wasb.json", "v4 - WASB, centred reference"),
         ("test_ball_v4.json", "test_ball_v3.json", "v4 - v3, raw reference"),
         ("centred_v4.json", "centred_v3.json", "v4 - v3, centred reference")]


def errors_cm(name: str) -> dict[str, list[float]]:
    res = json.loads((DATA / name).read_text())
    return {cid: [r["error_m"] * 100 for r in v["refs"] if r["error_m"] is not None]
            for cid, v in res.items() if cid.split("_")[0] in ("game8", "game9", "game10")}


def main(n_boot: int = 5000) -> None:
    rng = np.random.default_rng(0)
    for a_name, b_name, label in PAIRS:
        a, b = errors_cm(a_name), errors_cm(b_name)
        clips = sorted(set(a) & set(b))

        def diff(sample: list[str], a=a, b=b) -> float:
            return float(np.median(np.concatenate([a[c] for c in sample if a[c]]))
                         - np.median(np.concatenate([b[c] for c in sample if b[c]])))

        boot = [diff(list(rng.choice(clips, len(clips)))) for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{label}: median difference {diff(clips):+.1f} cm, 95 % interval {lo:+.1f} to {hi:+.1f} cm "
              f"({len(clips)} clips)")


if __name__ == "__main__":
    main()
