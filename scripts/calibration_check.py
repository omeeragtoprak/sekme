"""Is the reported bounce uncertainty honest? TrackNet games 8-10, system bounces vs reference contacts.

    .venv/bin/python scripts/calibration_check.py RESULTS_JSON RUNS_DIR [--centred-reference]
    (both relative to data/tracknet; --centred-reference: labels minus their game's GAME_LABEL_OFFSET_PX, as
    eval_tracknet.py; pass the results file that eval_tracknet.py --centred-reference wrote)

For every matched bounce, the error between the system bounce and the reference contact is normalised by
the combined covariance: the system's reported one plus the reference's own, obtained by perturbing the
labels with 1 or 2 px of noise and re-deriving the contact (Monte Carlo). A calibrated 2D Gaussian gives
a median normalised error of 1.18 and 95 % inside the 95 % ellipse (normalised error < 2.45).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.eval.labels import world_from_px  # noqa: E402
from courtvision.eval.tracknet import GAME_LABEL_OFFSET_PX, Labels, ball_centre_ground, contacts, load_labels  # noqa: E402

DATA = ROOT / "data" / "tracknet"
TEST_GAMES = ("game8", "game9", "game10")


def read(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def normalised_errors(results: str, runs: str, label_px: float, n_sim: int = 200,
                      centred: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    z = []
    for cid, clip_res in read(DATA / results).items():
        game, clip = cid.split("_")
        if game not in TEST_GAMES:
            continue
        offset = GAME_LABEL_OFFSET_PX[game] if centred else (0.0, 0.0)
        if tuple(clip_res.get("reference_offset_px", offset)) != tuple(offset):
            raise SystemExit(f"{results}: {cid} was scored with reference offset {clip_res['reference_offset_px']}, "
                             f"not {offset}; pass the matching results file or flag")
        lab = load_labels(DATA / "Dataset" / game / clip / "Label.csv", offset)
        report = read(DATA / runs / cid / "report.json")
        homographies = read(DATA / runs / cid / "frames.json")["H"]
        camera = report["calibration"]["camera_center_m"]
        calls = {round(f["bounce"]["frame"], 3): f for f in report["flights"] if f.get("call") and f.get("bounce")}
        refs = sorted(contacts(lab), key=lambda c: c.frame)
        for row, ref in zip(clip_res["refs"], refs, strict=True):
            if row["error_m"] is None:
                continue
            h = homographies[int(round(ref.frame))]
            sims = []
            for _ in range(n_sim):
                noisy = Labels(lab.visible, lab.xy + rng.normal(0, label_px, lab.xy.shape), lab.status)
                near = [k for k in contacts(noisy) if abs(k.frame - ref.frame) < 2]
                if near:
                    sims.append(ball_centre_ground(np.asarray(world_from_px(h, tuple(near[0].px))), camera))
            cov = np.array(calls[round(row["sys_frame"], 3)]["call"]["cov"]) + np.cov(np.array(sims).T)
            d = np.array([row["dx_m"], row["dy_m"]])
            z.append(float(np.sqrt(d @ np.linalg.solve(cov, d))))
    return np.array(z)


def main() -> None:
    results, runs = sys.argv[1], sys.argv[2]
    for label_px in (1.0, 2.0):
        z = normalised_errors(results, runs, label_px, centred="--centred-reference" in sys.argv[3:])
        print(f"{runs}: label noise {label_px} px  n={len(z)}  normalised error median {np.median(z):.2f} "
              f"(calibrated: 1.18)  inside the 95% ellipse {100 * np.mean(z < 2.45):.0f}%")


if __name__ == "__main__":
    main()
