# Independent benchmark: TrackNet tennis labels

`scripts/eval_tracknet.py` runs the full pipeline on every clip of the TrackNet tennis set
(Huang et al., 2019: broadcast 1280×720, 30 fps, ball pixel labelled by hand in every frame, bounce
frames marked) and scores the system's bounces against the labels (`courtvision.eval.tracknet`).
Data stays local under `data/tracknet/` (the set has no stated licence).

**Split.** The ball detector (WASB) was trained on games 1–7 and tested on games 8–10
(`nttcom/WASB-SBDT`, `configs/dataset/tennis.yaml`). Games 8–10 are the honest numbers; games 1–7
were used to tune decisions (below), never games 8–10.

**Reference contact.** The labelled bounce frame is only the frame nearest the contact (the ball
moves decimetres per frame at 30 fps). The reference is the meeting point of the labelled incoming and
outgoing tracks (two labels each side), lifted by the ball radius because labels mark the ball
centre (that alone moved the median error from 19 to 12 cm on games 8–10). Reference and system use
the same homography, so this measures detection, tracking, hit segmentation, the 3D fit and timing,
not the court calibration (bounded separately by the court-line residual, README).

## Results (commit with this file; `data/tracknet/results.json`)

| | games 8–10 (unseen) | all 95 clips |
|---|---|---|
| labelled bounces (on calibrated frames) | 154 | 516 |
| found within ±3 frames (recall) | **112 (73 %)** | 348 (67 %) |
| **first bounce after a stroke found** (the one a line call is about) | **110 / 142 (77 %)** | 330 / 471 (70 %) |
| system bounces with no label (false alarms) | 9 (precision 93 %) | 38 (precision 90 %) |
| ground distance to the reference, median / p90 | **11 / 33 cm** | 13 / 39 cm |
| timing, median absolute | 0.17 frame | 0.14 frame |
| decisive IN/OUT calls agreeing with the verdict at the reference point | **108 / 108** | 336 / 336 |

History on games 8–10: first run 69 % recall / 95 % precision / 12 cm (107 found, 103/103 decisive);
the 2D fallback below brought it to the table above.

The error is almost entirely along the court length (depth, the camera's weak axis): lateral
median |error| 3.6 cm, depth 18 cm before the radius correction; after it the depth bias is −3 to −4 cm
on both halves. With 1 px of label noise propagated into the reference, system error over the combined
uncertainty has median 1.18 (the value for a correct 2D Gaussian) and 90 % fall inside the 95 % ellipse;
with 2 px, 98 %. The reported uncertainty is therefore consistent with these data; the residual 12 cm
is not evidence of an over-confident system, but the reference cannot resolve errors below ~10 cm.

Caveat on the 336/336: agreement is judged at the *reference* point with the system's own zone; few
of these bounces are close to a line, so it shows the absence of gross errors, not line-call
accuracy at the centimetre level.

## Where bounces were missed (games 8–10, 47 misses before the 2D fallback)

| cause | n |
|---|---|
| no clear bounce kink in the flight (often a flight cut short by a false hit) | 15 |
| 3D fit rejected by the plausibility gates (reprojection, contact height, σ) | 23 |
| bounce found elsewhere / two bounces inside one flight (a missed hit) | 7 |
| too few calibrated detections | 2 |

**Accepted: 2D fallback** (`rally.image_bounce`). When the 3D fit fails or is rejected, the bounce is
taken from the image kink (lines through two detections each side), put on the ground with the ball
radius lift, with a Monte-Carlo covariance from 2.5 px detection noise plus the calibration term;
reported as `bounce.method = "2d"` with no trajectory. Ungated it found many false kinks (games 1–7:
precision 89 → 69 %), so it is gated on games 1–7 only: depth σ ≤ 0.5 m, ≥ 3 m from the net, inside
the court + ~1 m. Games 1–7: recall 60.2 → 65.2 %, precision 89.3 → 89.1 %. Held-out games 8–10:
recall 69.5 → 72.7 %, precision 94.7 → 92.6 %, decisive calls 108/108. On the US Open clips it added
two calls (one IN 1.4 m inside, one REVIEW 7 cm from the baseline ± 11 cm) and changed none.

Most remaining misses are *second* bounces (a winner bouncing twice); for line calling the first
bounce after a stroke is what counts, hence the separate row above.

**Rejected: learned stroke detector.** A small MLP over a ±7-frame window of the ball track and the
ball's position relative to both player boxes, trained on TrackNet stroke labels of games 1–5,
threshold on games 6–7. Strokes on games 6–7: precision 58 → 82 %, recall 77 → 90 %, and first-bounce
recall 70.5 → 78.4 %. On held-out games 8–10 first-bounce recall fell 77.5 → 66.2 % and one decisive
call disagreed: it does not generalise from five matches. Not merged (code in the git history of this
session only).

Hit segmentation remains the main lever. Also tried and **rejected**: accepting a reversal as a hit only next
to the player's box (reach 1.0 box height). Games 1–7: recall 60.2 → 61.9 %, precision 89.3 → 90.7 %;
games 8–10: recall 69.5 → 60.4 % and one decisive disagreement — it does not generalise.
