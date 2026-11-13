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

| WASB ball network, TennisCourtDetector court (the original networks; current defaults below) | games 8–10 (unseen) | all 95 clips |
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

**Current defaults** (SEKME ball network v3 + SEKME court network, `data/tracknet/test_ball_v3.json`),
games 8–10: first bounce after a stroke 109 / 142, 112 bounces found with 8 false alarms (precision
93 %), distance to the reference median / p90 **10 / 30 cm**, decisive calls 107 / 107. Ball v2 gave
110 / 142, 7 false alarms, 16 / 38 cm, 107 / 107; v1 108 / 142, 10 false alarms, 15 / 38 cm, 105 / 105.
Every variant is in `results/tracknet_benchmark.txt`; the networks and the rules that chose them are in
[MODELS.md](MODELS.md).

For the WASB set in the first table: the error is almost entirely along the court length (depth, the camera's weak axis): lateral
median |error| 3.6 cm, depth 18 cm before the radius correction; after it the depth bias is −3 to −4 cm
on both halves. With 1 px of label noise propagated into the reference, system error over the combined
uncertainty has median 1.18 (the value for a correct 2D Gaussian) and 90 % fall inside the 95 % ellipse;
with 2 px, 98 %. The reported uncertainty is therefore consistent with these data; the residual 12 cm
is not evidence of an over-confident system, but the reference cannot resolve errors below ~10 cm.

Caveat on the 336/336: agreement is judged at the *reference* point with the system's own zone; few
of these bounces are close to a line, so it shows the absence of gross errors, not line-call
accuracy at the centimetre level.

## Systematic offsets (2026-11-13; evidence in `results/position_bias.txt`)

Three offsets sit inside the numbers above; none of them is random error.

1. **The labels.** TrackNet ball labels lie right of and below the ball's centre in the image
   (`scripts/label_offset.py`, no network involved): a constant of 0.6 to 1.7 px that differs between
   games, plus a lead of 0.1–0.19 frame along the ball's motion on eight of the ten games. Only the
   constant moves a reference contact (a lead shifts each point along its own track, and the contact
   is where two tracks meet). A reference built from the raw labels sits low in the image, i.e.
   towards the camera.
2. **The reference construction.** Extrapolating two labels on each side of the bounce in straight
   lines misses the curve of the track: on simulated shots with exact pixels the reference lands 4–6 cm
   *away* from the camera, while the 3D fit is unbiased (`scripts/reference_bias_sim.py`).
3. **The ball networks.** Constant offsets from the ball's image centre on games 8–10
   (`scripts/detection_offset.py`): WASB +0.5…+0.75 px right and +0.25…+0.64 px below; our U-Nets
   v1/v2, trained with `label × scale`, 1.8–2.2 px below; v3, trained on labels minus 1.5 px, within
   0.3 px. All of them reproduce the labels' lead. The court network is not a cause (v1 with either
   court network: 15.0 cm).

`--centred-reference` removes (1), using each game's constant (`GAME_LABEL_OFFSET_PX`); (2) is not
corrected. Bounce minus reference on games 8–10, median:

| | raw labels: \|error\| / depth | centred per game: \|error\| / depth |
|---|---|---|
| WASB + TCD court | 11.4 / −3.6 cm | 13.3 / −9.8 cm |
| SEKME v1 + SEKME court | 15.1 / −13.1 cm | 19.9 / −19.2 cm |
| SEKME v2 + SEKME court | 16.1 / −14.9 cm | 21.1 / −19.6 cm |
| **SEKME v3 + SEKME court** | **10.0 / +0.2 cm** | **11.4 / −5.9 cm** |

(depth: negative = towards the camera.) v3 against v2: −6.1 cm at the median (95 % interval −10.6 to
−2.9 cm, clip bootstrap, `scripts/bootstrap_bounce.py`); against WASB −1.4 cm (−2.9 to +1.0): on par,
not demonstrably better. Adding (2), the median depth error against the true bounce is roughly −1 cm
for v3, −5 cm for WASB and −15 cm for v1/v2. This is an estimate from two separate measurements, one of
them a simulation with exact pixels, so read it as a few centimetres either way, not as a direct
measurement. Down the image is the camera's weak axis, so this matters most for calls on the
baselines. With the centred reference the reported uncertainty also holds for v3 (94 % inside the 95 %
ellipse with 1 px label noise; WASB 89 %, v2 87 %). The per-game references were introduced after
v3's games 8–10 run had been scored against a fixed 1.5 px one; the ranking is the same under all
three references (`results/position_bias.txt`).

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

## Serve detection: the dead-ball check (rule written before the change, 2026-11-13)

A serve needs a dead ball in the second before the toss: until now, *any* ball detection outside the
server's reach in that second rejected it. With ball network v3 three serves found by v1 were lost on
the US Open clips. Inspected case (`usopen2024_final_hl`, serve at frame 2715): v3 also sees a second,
almost static ball-like object far from the server (~(1446, 521) px, frames 2619–2657), and that
alone failed the check.

**Change.** A detection in that second breaks the dead ball only if it moves faster than a toss
(`TOSS_MAX_SPEED` box heights per frame against the previous detection within 3 frames). Slow or
isolated detections away from the server (a ball held by a ball kid, a spare ball) no longer do.

**Rule.** Accepted only if all hold: (a) TrackNet games 1–7 (WASB detections, events re-run): first
bounce recall and precision not lower, no decisive call changes its verdict; (b) on the five US Open
clips with ball v3, no serve found now is lost; (c) of the two other serves lost with v3, which were
*not* inspected before this rule (`usopen2025_qf_djokovic_fritz_hl` near frame 5737,
`usopen2025_final_hd` near frame 4679), at least one is recovered; (d) no radar pair's |error| grows by
more than 2 points. Games 8–10 are not used.

**Result: rejected.** (a) held: games 1–7 first bounces 220 / 329 before and after, false alarms
29 → 27 (precision 89.1 → 89.7 %), decisive calls 228 / 228 with no verdict changed, serve flights
45 → 47. (b) held, with one serve moved by 20 frames (`qf`, 3689 → 3709). (c) failed: neither of the
two uninspected serves came back, nor did the inspected one; instead seven new serves appeared
elsewhere (two of them 2.4 s apart in the 2024 final), and 12 IN calls changed into other flights.
(d) the radar pairs were unchanged. The dead-ball check is not what keeps these serves out; not merged
(patch kept outside the repository). Serve recall on broadcast clips needs labelled serves before it
can be tuned.
