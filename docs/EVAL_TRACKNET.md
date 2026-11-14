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

**Current defaults** (SEKME ball network v4 + SEKME court network, `data/tracknet/test_ball_v4.json`),
games 8–10: first bounce after a stroke 110 / 142, 113 bounces found with 9 false alarms (precision
93 %), median / p90 10 / 32 cm, decisive calls 108 / 108; on par with v3 (MODELS.md, v4). Ball v3
(`data/tracknet/test_ball_v3.json`), games 8–10: first bounce after a stroke 109 / 142, 112 bounces found with 8 false alarms (precision
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

## Hit detection: a bounce taken for a stroke (rule written before the games 8–10 run, 2026-11-13)

**Finding (games 1–7, WASB detections).** Of the 115 first bounces missed, 40 had a "hit" detected at
the bounce itself (±3 frames): after a bounce on the near half the rising ball moves up the image, the
reversal detector takes that for the near player's stroke, the incoming flight ends there and its
bounce is lost. The real stroke comes later as a second reversal. What separates the two is the
horizontal image velocity: across a bounce it carries on, across a stroke it breaks. On games 1–5,
|Δvx| / incoming speed had a median of 1.83 (near-side strokes, n = 80) against 0.12 (bounces taken
for strokes, n = 27). (TrackNet's hit labels are incomplete, so hit-level precision is not a usable
metric here; decisions use the bounce metrics.)

**Change** (`events/hits.py`, `BOUNCE_VX_CHANGE`). A near-side reversal with |Δvx| below T × incoming
speed (velocities over ±0.2 s) is not a stroke; the detector keeps the incoming state, so the next
reversal becomes the stroke. T was chosen on games 1–5 from {0.15, 0.25, 0.35}: first bounces 158 →
175 / 175 / 172 of 241, precision 85.8 → 87.8 / 87.4 / 88.5 %; **T = 0.15** (most first bounces, then
precision). Check on games 6–7, not used for the choice: first bounces 62 → 67 / 88, precision 97.3 →
96.0 %, decisive calls 69 / 69. Games 1–7 together: 220 → 242 / 329 (66.9 → 73.6 %), precision
89.1 → 90.0 %.

**Rule for the single games 8–10 run** (default networks, ball v3 + SEKME court; baseline
`test_ball_v3.json`: 109 / 142 first bounces, precision 93.3 %, median 10.0 cm, 107 / 107). The change
becomes the default only if all hold: (a) at least 112 first bounces found (baseline + 3); (b)
precision at least 92.0 %; (c) every decisive call consistent with the label; (d) median distance to
the reference at most 11.0 cm.

**Result: rejected.** Games 8–10: (a) 113 / 142 first bounces (baseline 109) ✓; (b) precision 94.3 %
(7 false alarms) ✓; (d) median 10.2 cm ✓; (c) failed: 110 / 111 decisive calls consistent. The
exception is a bounce the gate newly found (`game8_Clip9`, frame 209): called IN at P(in) = 0.997,
5.9 cm inside the right singles sideline with a lateral σ of 2.1 cm, while the reference contact lies
6.6 cm further out (OUT). A confident wrong call is the error the rule exists to stop, so the gate is
not the default and was removed from the code (it is in the git history, commit e46775d). What it
leaves open: the recall gain is real on all three game sets, and the failure is the call's
uncertainty on a near-sideline bounce at the end of a point ("next stroke not seen"), not the gate's
logic; a fix there would have to be checked on new data, since games 8–10 have now been used again.

## Bounces in the frame of the lines (rule written before the games 8–10 run, 2026-11-13)

**Finding (games 1–7, WASB detections).** The reported uncertainty was over-confident across the court:
75 % of bounces inside their 95 % ellipse against the per-game centred reference (1 px label noise,
which a three-cornered-hat estimate on games 8–10 confirms: label noise 0.9 / 1.1 px in x / y, our
ball network's 0.34 / 0.39 px; that estimate pairs each label with the detection nearest to it and drops pairs
over 10 px apart, which correlates the two errors and trims the tails, so the split between label and
detection noise is approximate — noted after a later code review, not re-measured). Lateral |z| had a median of 1.11 (0.67 if calibrated). Cause: the 3D fit
works with the pinhole camera recovered from the homography (square pixels, principal point at the
image centre), and on six of the seven games that camera puts ground points 2–6 cm (median, centre
line) away from where the homography — which locates the painted lines — puts them. The reference uses
the homography, and so do the lines.

**Change** (`events/flights.py`, `in_line_frame`, flag `LINE_FRAME_BOUNCE`). After a successful 3D fit
the bounce is read where the fit's camera images it, back through the homography, and the fitted track
is moved by that offset. No parameter. Games 1–7 end to end: bounces found unchanged (220 / 329 first,
29 false alarms); median lateral error 5.32 → 3.85 cm; median distance 14.0 → 13.2 cm (raw reference),
16.0 → 15.0 cm (centred); inside the 95 % ellipse 75 → 85 % (near half 89 %, far half 81 %); decisive
calls 227 / 227 (raw reference; one call became low-confidence).

**Rule for the single games 8–10 run** (ball v3 + SEKME court, events re-run on the cached detections).
The flag becomes the default only if all hold: (a) median lateral |error| against the centred
reference lower than with the flag off; (b) median distance to the reference at most 10.5 cm (raw;
baseline 10.0) and 11.9 cm (centred; baseline 11.4); (c) every decisive call consistent with the label
(raw reference); (d) at least 108 first bounces found and precision at least 92 %.

**Result: rejected.** Games 8–10: (a) failed: median lateral error 3.17 → 3.21 cm (centred); (b) failed
for the raw reference, 10.0 → 11.1 cm (centred 11.4 → 11.2 cm); (c) 107 / 107 and (d) 109 / 142, 93.3 %
held. Afterwards, not part of the rule: the camera-vs-homography offset is small on games 8 and 10 (0.7
and 0.2 cm at the centre line) but 5.2 cm on game 9, and even there the lateral error did not fall
(3.90 → 4.12 cm). So the games 1–7 gain did not carry over to ball v3 with the SEKME court network, and
the proposed mechanism is not established; removed from the code (git history, commit d568168). For the
record, the same offset on the US Open clips is 0.1–3.3 cm at the centre line (sidelines p90 up to
6.8 cm), so the question matters on real footage; settling it needs labelled bounces from it.

## Open findings (games 1–5, WASB detections; descriptive, nothing changed)

Where each labelled first bounce of games 1–5 is lost (238; the flight that covers it, fitted bounce
within 3 frames counts as found):

| Outcome | First bounces |
|---|---|
| found | 159 |
| far→near flight ends at the bounce: the hit detector fired on the near-side bounce, so no kink is left inside the flight | 28 |
| a changepoint was found but the 3D fit was rejected (24 of 36: reprojection error) and the image fallback was not trusted; 23 of the 36 span a labelled stroke the hit detector missed (two shots fitted as one) | 36 |
| a bounce was fitted, more than 3 frames off (all 10 *earlier*, median 4.0 frames) | 10 |
| no flight covers it / too few calibrated detections | 5 |

An earlier version of this note counted 54 misses "inside a flight whose bounce was placed
elsewhere"; that group mixed the 36 rejected fits (no call made at all) with the 10 misplaced ones.
Timing is the smallest part: **hit/bounce segmentation** — false hits at near-side bounces and missed
strokes — accounts for most of the losses. It is the same step the rejected hit gate addressed. Games
8–10 have now been used for many accept/reject decisions; a further change there would need new
labelled data (e.g. bounces marked in the Review Room) to be tested honestly.

## Open findings 2 (games 1–7, `runs_dev_base`; descriptive, nothing changed; 2026-11-14)

**Hit timing predicts whether the next first bounce is found.** Each first bounce on the near half was paired
with the labelled stroke before it and the system's far-side hit nearest to that stroke:

| System's hit vs the labelled stroke | First bounces | Found |
|---|---|---|
| within ±3 frames | 89 | 77 (87 %) |
| more than 3 frames late | 50 | 18 (36 %) |
| more than 3 frames early | 26 | 11 (42 %) |
| no far-side hit within 20 frames (mostly near-side strokes) | 143 | 105 (73 %) |

About 47 of the ~103 missed first bounces follow a mistimed hit. Far-side hits are mistimed more often than
near-side ones (within ±3 frames of a labelled stroke: 56 % against 77 %; late by more than 3 frames: 29 %
against 6 %). This is an association, not yet a cause (hard rallies could produce both), but the mechanism is
plausible: a late hit hands the first detections of the next flight to the previous one, whose fit then fails.
A hit-timing refinement for rally hits (as the serve contact already has) is the candidate this points to; the
E2E-Spot labels time strokes directly, so its test can measure it. (Tried as a changepoint shift of each rally
hit: hit timing improved, the bounce check on games 6–7 failed; rejected, see the last section.)

**Near-side bounces taken for strokes are already separable.** Of the near-side hits, 118 match a labelled
stroke and 46 a labelled bounce; the rejected gate's single feature (change of horizontal image velocity over
the incoming speed) separates them with AUC 0.976 (medians 1.92 and 0.17), against 0.69–0.79 for vertical
reversal, distance to the player box and the next-to-player flag. A classifier would add little there; the
gate failed on call safety, not on classification. Script: `devir/scratch/seg_hit_features.py`.

## Ball tracker bugs found in review (rule written 2026-11-14, before the fixes were run)

An independent code review found two bugs in `perception/ball.py`, confirmed by hand:

1. `_remove_outliers` edits the track while it scans it, so later points are tested against neighbours it has
   already removed (NaN fits, and a NaN residual never counts as too far); on a clean straight track with one
   wild point it removes the two good points beside the outlier and keeps the outlier.
2. The constant-acceleration predictor in `track_ball` adds half an acceleration step too many
   (`pred = p1 + vel + acc / 2` with `vel` already holding the full step), which biases the choice between
   candidates near bounces and strokes.

A third, rarer one: a serve contact refined into a long detection gap takes its image point from a missing
frame (NaN), which silently drops the serve's contact-height measurement.

**Fixes:** outliers removed worst first against the remaining points (a removed point is never a neighbour
again); `pred = p1 + vel`; the serve contact point interpolated from the detections around it when its
frame has none.

**Rule.** These are bug fixes, so the bar is *no measurable harm*; only development data exists for it
(games 8–10 are used up). On TrackNet games 1–7 (`runs_dev_base` detections; baseline 220 / 329 first
bounces, precision 89.1 %, decisive calls 228 / 228) the fixes are adopted if first-bounce recall falls by no
more than 1 point, precision falls by no more than 1 point, and every decisive call agrees with the
reference. Otherwise the code is kept as it is and the result is documented here. Broadcast clips and the
serve labels are reported, not used.

**Result (2026-11-14): adopted.** TrackNet games 1–7 unchanged: 220 / 329 first bounces, precision 89.1 %,
228 / 228 decisive calls consistent (not a single ball position changed on those clips: their detections
never reached either bug). On the broadcast clips (reported) 94 of 33 209 ball frames changed on the 554 s
60 fps clip, 13 of 5 395 on the 90 s one and none on the 30 fps ones; no call verdict changed, bounces moved
at most 0.2 cm, one call was added; the 23 labelled serves of the 2025 final are matched exactly as before.
Two unreachable edge cases from the same review were hardened (`detect_hits` on a track shorter than its
direction window, an empty mask in `_fit_one`).

## Two scorer corrections (decided 2026-11-14, before measuring their effect)

A code review of `courtvision.eval.tracknet` found two places where the scorer does not do what this document
says it measures: (1) system bounces are paired with references greedily in the references' order, so an
earlier reference can take a system bounce that is closer to the next one; (2) a reference contact's two
track lines are fitted on the labelled points up to 4 frames before and after the bounce, even when that window
reaches back over the hit (status 1) or forward into the next bounce. Corrections: pairs closest-first, one to
one, as the other evaluations here do; the windows stop at the neighbouring hit or bounce. They are measurement
corrections, so they are **adopted whichever way the numbers move**; the development numbers (games 1–7, the
`runs_dev_base` detections) are re-stated below with both scorers, and games 8–10, which are used up, are not
re-scored.

**Result (games 1–7, `runs_dev_base`; adopted as decided).**

| | old scorer | corrected |
|---|---|---|
| first bounces found | 220 / 329 (66.9 %) | 220 / 323 (68.1 %) |
| games 6–7 | 62 / 88 (70.5 %) | 62 / 87 (71.3 %) |
| all bounces found, false alarms, precision | 236, 29, 89.1 % | 236, 29, 89.1 % |
| decisive calls consistent, median error | 228 / 228, 14.0 cm | 228 / 228, 14.0 cm |

The pairing correction changed no pair on these games. The window correction changed six references, all of
the same kind: one bounce labelled on two adjacent frames (game 1 clips 7, 10, 11, 12, 13; game 6 clip 2). The
old windows let the second label's track line run through the first, so both contacts landed on the first
label and both counted as first bounces, one found and one "missed"; now the duplicate is a second bounce. 15
of the 369 reference contacts now use the labelled point instead of crossing a neighbouring hit or bounce.
Numbers elsewhere in these documents were measured with the old scorer and stay as recorded. A bounce labelled on two
adjacent frames still counts as two references in the all-bounces recall (the second one usually "missed");
no rule uses that number, and merging such labels was not part of the correction decided above, so it is only
noted here.

## Development candidate: rally-hit timing refinement (rule for the games 6–7 check written 2026-11-14, before it)

**Change** (development only, not in the pipeline yet): each detected rally hit (not serves) is moved to the
two-piece changepoint of the ball's image track within ±12 frames when that split reduces the residual at least
3× (`events.image_track.changepoint`, as the serve contact is refined). The window and the gain were chosen on
games 1–5 by hit timing alone (far-side hits within ±3 frames of a labelled stroke 51 → 68 %, near 75 → 77 %);
on games 6–7, not used for the choice, far-side timing went 72 → 78 % (near 81 → 81 %).

**Check on games 6–7** (events re-run from the cached detections with the current code, baseline and candidate
alike): first bounces found must not fall, bounce precision must not fall by more than 1 point, and every decisive
call must agree with the label. Games 1–5 are reported. If it holds, the candidate is a development result that
may later take one of the two E2E-Spot test slots, with its own rule written there first; if not, it is recorded
here and dropped.

**Result (2026-11-14): rejected on all three counts; dropped.** Events re-run from the cached detections of
games 1–7 (`runs_dev_now` baseline, `runs_dev_refine` candidate, the same code apart from the change):

| | baseline | candidate |
|---|---|---|
| games 6–7: first bounces found | 62 / 87 (71.3 %) | 59 / 87 (67.8 %) |
| games 6–7: all bounces found, false alarms, precision | 73, 2, 97.3 % | 69, 4, 94.5 % |
| games 6–7: decisive calls consistent | 66 / 66 | 63 / 64 |
| games 1–5 (reported): first bounces found | 159 / 236 (67.4 %) | 163 / 236 (69.1 %) |
| games 1–5 (reported): false alarms, precision | 27, 85.9 % | 24, 87.4 % |
| games 1–5 (reported): decisive calls consistent | 163 / 163 | 166 / 166 |

On games 6–7 the candidate loses a first bounce in four clips (game 7 clips 1, 4, 8, 9) and gains one (game 7
clip 6); two false alarms are added (game 7 clips 4, 5), and one second bounce (game 6 clip 3, frame 97,
labelled IN) is called OUT, 32 cm from its label. On games 1–5, where window and gain were chosen, it moves
first bounces both ways in 9 clips for a net +4; over games 1–7 it gains 8 first bounces and loses 7. Better
hit timing did not carry over to the bounces as the association in "Open findings 2" suggested. A plausible
reason, not checked: moving a hit also moves where the next flight's fit starts, which can help or hurt. The change stays out of the pipeline and does not take an
E2E-Spot slot. (Baseline note: the current code finds 221 / 323 first bounces on games 1–7 against 220 for the
`runs_dev_base` events above; both are development numbers, and every comparison here is within one re-run.)

**Correction (found 2026-11-14 evening).** "Re-run from the cached detections" above is not what happened. The
re-run tool made the pipeline ignore the cache key by patching `perception_key`; after the cache-key change
earlier that day (`same_perception_key`, settings in the key) that patch no longer matched the old caches, so
both re-runs recomputed the detections on the CPU with the same networks (WASB, TCD). The recomputed detections
differ slightly from the cached ones, which is the likely source of the 221 vs 220 above, but the baseline and
the candidate got byte-identical detections on all 65 clips (checked), so the comparison and the rejection stand.
The tools now patch `same_perception_key` and reuse the caches.

## Image-kink bounce on sparse detections (rule written 2026-11-14, before the change is made)

**Problem.** When the 3D fit of a flight is rejected, its bounce is taken from the kink of the image track
around a detection, using two detections on each side. Nothing checks how far apart in time those four
detections are. On a heavily compressed copy of a 4 s broadcast excerpt (x264 CRF 40, 1/8 of the original's
size) the ball is found on 19 of 119 frames, three flights merge into one, its fit is rejected, and the image
kink, built from detections 14 frames apart (30 fps), gives an IN call at a point 4.7 m from the real bounce
with a stated 1σ of 8 cm. The Monte Carlo uncertainty only reflects pixel noise, not the gaps.

**Change.** The four detections around the kink must lie within 0.25 s; otherwise no image bounce (no call).
Chosen from TrackNet games 1–5, where every image-kink call spans at most 0.13 s (4 frames), with a margin.
Before writing this rule the spans were also looked at on games 6–7 (all 0.10 s) and on the six broadcast
runs (45 image-kink calls, at most 0.18 s), so the check below is a confirmation that nothing measured
changes, not a blind test.

**Check.** Adopted if (1) on TrackNet games 1–7, events re-run from the cached detections, every call and
every bounce is the same as without the change; (2) on the six broadcast `_v4` runs, re-analysed from
their cached detections, no call changes; (3) on the CRF 40 excerpt the spurious call is withdrawn.
Otherwise the code stays as it is and the result is recorded here.

**Result (2026-11-14): adopted.** (1) TrackNet games 1–7, events re-run twice from the cached detections (gate
off and on; the caches were really reused this time, checked byte for byte): no call and no bounce changed on any
of the 65 clips. (2) The six broadcast `_v4` runs, re-analysed twice from their cached detections: 590 flights,
none changed. (3) The CRF 40 excerpt: the spurious call is withdrawn; its flight now reads "görüntüden sekme
noktası güvenilmez: kırılmanın çevresindeki tespitler 2,34 sn'ye yayılıyor (top seyrek bulunmuş)". The span the
code measures is between the fit's own detections (calibrated, inside the flight), so it can be wider than the
spacing of all detections used to describe the problem above (0.47 s there). `IMAGE_BOUNCE_MAX_SPAN_S = 0.25`.
