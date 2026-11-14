# Models

SEKME uses four networks. The ball and court networks were designed and trained in this project and are
the defaults. Person and pose networks of our own were distilled from YOLO26 and tested on hand-labelled
frames; both fell short of their teacher, so YOLO26 stays the default for those two roles.
Everything after the networks (court refit and camera, ball tracking, events, the 3D physics fit, calls
and their uncertainty, self-checks) is this project's code; see [ARCHITECTURE.md](ARCHITECTURE.md).

| Role | Default network | Trained by | Training data | Alternative (`--model KEY=PATH`) |
|---|---|---|---|---|
| Ball | SEKME ball U-Net v4 (`models/ball_unet.py`) | this project | TrackNet tennis, games 1–7 (v1: 1–5), targets corrected for the labels' offset | WASB HRNet (its authors, MIT) |
| Court keypoints | SEKME court U-Net (`models/court_unet.py`) | this project | TennisCourtDetector labels: the 6 630 frames of `data_train.json` (of 8 841) | TennisCourtDetector network (its author, no published licence) |
| Players | YOLO26-s + ByteTrack (Ultralytics, AGPL-3.0) | Ultralytics | COCO | SEKME person U-Net, distilled from YOLO26-s |
| Pose | YOLO26-s-pose (Ultralytics, AGPL-3.0) | Ultralytics | COCO | SEKME pose U-Net, distilled from YOLO26-s-pose |

No weights are in the repository (`models/` is ignored); the training scripts reproduce ours.

**How the test sets were used.** The held-out sets (TrackNet games 8–10, the court validation split, the
human-labelled pose frames) were never used to fit or tune a parameter. They were used to accept or
reject a change against a rule written down and committed before the run; each such use is listed in
the git history and in [EVAL_TRACKNET.md](EVAL_TRACKNET.md). Repeated use makes them less independent,
which is why the human-labelled set below is kept for the final checks.

## SEKME ball network

This section describes v1 (games 1–5); v2, v3 and v4, the current default, are at the end of it.

**Task.** Find the ball in every frame of broadcast video. The ball is 2–6 px wide at the network's
input size, often blurred into a streak, and the court carries ball-sized white marks.

**Input and output.** Three consecutive frames, each resized to 512×288, RGB, ImageNet-normalised and
stacked on the channel axis (9×288×512). One logit heatmap per input frame (3×288×512). This is the
same contract as WASB, so the production detector (`perception/ball.py`: sliding three-frame window,
heatmaps averaged over the windows a frame appears in, connected-component centroids, then the
tracker) runs either network unchanged; the checkpoint says which one it holds.

**Architecture.** A compact U-Net, ~1.95 M parameters: four encoder levels (32, 64, 128, 256 channels;
two 3×3 conv + batch norm + ReLU each; 2×2 max pooling), a decoder with bilinear up-sampling and skip
connections, a 1×1 head. Seeing three frames at once lets the network use motion, which separates the
ball from static white marks.

**Data.** TrackNet tennis (Huang et al., 2019): 19 835 broadcast frames from 10 matches, 1280×720,
30 fps, ball position labelled by hand in every frame. The split is fixed by match, never by frame:

| Games | Frames | Use |
|---|---|---|
| 1–5 | 10 290 | training |
| 6–7 | 3 870 | choosing the epoch (F1 at 4 px) |
| 8–10 | 5 675 | final test only, once (also WASB's test split) |

**Targets and loss.** A disk of radius 2.5 px at each labelled ball position, empty maps where the
ball is not visible. Loss: the weighted BCE of TrackNetV2 (each term scaled by how wrong the network
still is), summed and divided by the number of ball pixels. The head bias starts at p = 0.01
(RetinaNet prior). Both matter: averaged over all ~147 000 pixels of a frame, the ~21 ball pixels
vanish, and the first run settled on a flat map (peak 0.12 everywhere, validation F1 0).

**Training.** AdamW (lr 1e-3, weight decay 1e-4, one-cycle schedule), batch 8, 10 epochs, horizontal
flips and brightness/contrast jitter; about 2.3 hours on an Apple M4 (PyTorch, MPS). The checkpoint
stores the split, the validation scores of every epoch and the arguments.

**Selection rule, written before the test.** Our network becomes the production default if, on games
8–10, its frame-level F1 at 4 px is at most 2 points below WASB's (or better) and its first-bounce
recall in the end-to-end benchmark is at most 2 points below. Games 6–7 cannot choose between the two:
WASB was trained on them.

### Results

Validation (games 6–7, epoch choice; middle frame of single windows, argmax): F1 at 4 px rose from
0.58 after epoch 1 to **0.770 at epoch 8** (precision 0.82, recall 0.73; at 10 px 0.98 / 0.87); epochs
9–10 did not improve it. Every epoch is stored in the checkpoint.

Test on games 8–10 (5 675 frames, unseen by both networks), through the production
detector (`scripts/eval_ball.py`); the highest-scoring candidate per frame, 1280×720 pixels:

| | SEKME U-Net (ours) | WASB HRNet |
|---|---|---|
| F1 at 4 px (precision / recall) | 0.912 (0.945 / 0.881) | 0.928 (0.967 / 0.893) |
| F1 at 10 px (precision / recall) | **0.958** (0.993 / 0.926) | 0.955 (0.995 / 0.918) |
| time per frame (M4, fp16) | **23 ms** | 44 ms |
| parameters | 1.95 M | 1.48 M |

End to end on the same games (`scripts/eval_tracknet.py --ball-model …`, everything after the ball
network unchanged):

| | SEKME U-Net (ours) | WASB HRNet |
|---|---|---|
| first bounce after a stroke found | 108 / 142 (76.1 %) | 110 / 142 (77.5 %) |
| precision | 91.7 % | 92.6 % |
| distance to the labelled contact, median / p90 | 15.0 / 37.7 cm | 11.4 / 33.4 cm |
| decisive calls consistent with the label | 105 / 105 | 108 / 108 |
| truth inside the reported 95 % ellipse (1 px label noise) | 87 % | 91 % |

**Decision.** Both pre-registered conditions hold (F1 at 4 px −1.7 points, first-bounce recall
−1.4 points), so the SEKME U-Net is the production default. The cost, which the rule did not cover:
bounces are placed ~3.6 cm further from the labelled contact at the median. The reported uncertainty
widens with it (median normalised error 1.23 vs 1.18 for a calibrated estimate), so calls do not
become over-confident. WASB stays available: `sekme analyze --ball-model models/wasb_tennis_best.pth.tar`.

**Real broadcast.** US Open 2023 final (720p, 30 fps, 90 s): 52 calls vs 53 with WASB, every shared
bounce with the same verdict; one serve is no longer recognised because the network misses the
motion-blurred first frames after its contact (7 frames; WASB sees them). US Open 2025 final (1080p,
60 fps): both networks were trained at 30 fps, so the detector feeds every second frame
(t, t+2, t+4); 24 calls vs 26 with WASB, no verdict differences on shared bounces. The same stride
removed two false OUT calls that both networks had made on dead balls thrown to ball kids.

**Radar** (`results/radar.txt`): with our ball network the fitted serve speeds agree less well with the
stadium radar than with WASB (mean |error| 7.7 % vs 4.9 % on 6 timed serves), mostly because of one fast
60 fps serve (−18.7 % vs −8.9 %): the same weakness on motion-blurred balls right after contact.

The validation and test numbers are not comparable: validation scores single windows by argmax,
the test uses the production detector (heatmaps averaged over three windows, blob centroids).

### Ball network v2

Aimed at the blurred-ball weakness above. Same architecture, loss and schedule as v1; two changes:

- **Data:** games 1–7 for training, like WASB, with every fifth clip of those games held out to choose
  the epoch (52 training clips, 13 selection clips). Games 8–10 stay the test set.
- **Motion blur:** on 30 % of the training windows, the same straight-line smear (3–9 px long, random
  angle) is applied to all three frames, imitating a fast ball or a camera pan.

12 epochs (~2.9 h on the M4); epoch 10 chosen (selection F1 at 4 px 0.847, not comparable with v1's
0.770: different clips). v1 is kept as `models/sekme_ball_v1.pt`.

**Rule, written before training** (commit b6f8f8c): v2 replaces v1 only if on games 8–10 its
frame-level F1 at 4 px and its end-to-end first-bounce recall are both at least v1's (0.912, 108 / 142).

| Games 8–10 | v1 | **v2** | WASB |
|---|---|---|---|
| F1 at 4 px (precision / recall) | 0.912 (0.945 / 0.881) | **0.922** (0.949 / 0.897) | 0.928 (0.967 / 0.893) |
| F1 at 10 px | 0.958 | **0.965** | 0.955 |
| first bounce after a stroke found | 108 / 142 | **110 / 142** | 110 / 142 |
| bounces found / false alarms / precision | 111 / 10 / 91.7 % | **113 / 7 / 94.2 %** | 112 / 9 / 92.6 % |
| distance to the labelled contact, median / p90 | 15.0 / 37.7 cm | 16.1 / 37.7 cm | 11.4 / 33.4 cm |
| decisive calls consistent with the label | 105 / 105 | 107 / 107 | 108 / 108 |
| truth inside the reported 95 % ellipse (1 px label noise) | 87 % | 89 % | 91 % |

**Decision.** Both conditions hold, so v2 is the default. Outside the rule: the median bounce distance
grew by 1.1 cm (p90 unchanged), and the reported uncertainty kept pace (89 % inside the 95 % ellipse).
Games 8–10 have now been used for several accept/reject decisions (see the git history); the remaining gap to WASB in bounce
position (16 vs 11 cm) is not something to tune on them further.

**Defect found after the decision** (independent review, 2026-11-13): 3 of the 7 blur lengths
(4, 6, 8 px) were even, and an even-length line kernel is not centred on `cv2.filter2D`'s anchor, so
in ~13 % of training windows the blurred frames sat ~0.6 px (network pixels; ~1.4 px at 1280×720),
mostly vertically, away from their labels. Fixed: odd lengths only, with a test that the blur never
moves a point's centre (`tests/test_ball_unet.py`). `scripts/eval_ball.py` now also reports the mean
detection − label offset (`bias_px`).

### TrackNet labels are offset from the ball (found 2026-11-13)

Measuring the mean detection − label offset (`eval_ball.py`, `bias_px`) on the selection clips gave
(dx, dy) = (−1.30, +0.62) px for v1 and (−1.37, +0.78) px for v2 (1280×720, SE 0.03): far beyond what
the blur defect could cause, and in v1 too, which has no blur. A check that uses no network
(`scripts/label_offset.py`: the moving ball isolated by frame differencing, its centroid within 8 px
of the label; frames where the ball moved under 16 px over either 3-frame leg are skipped, so the sample
leans to faster balls) shows where it comes from: **the TrackNet labels sit (+1.53, +1.39) px from the ball's
centre in the image** (medians +1.56, +1.53; n = 1 970; the same with unweighted masks, so lighting
does not drive it). An independent review then showed it has two parts: a constant, and a term
proportional to the ball's speed: on most games the labels lead the ball's image by 0.1–0.19 frame
(below). Its origin is not known; the labelling tool and convention are not documented. With the
training mapping `label × scale` (which also ignores `cv2.resize`'s half-pixel centres) this predicts
what the networks do:

- across the image (x), horizontal flips turn the label offset into ±, so the network learns the ball's
  image centre: predicted −1.5 px against the label, measured −1.30 / −1.37;
- down the image (y), nothing cancels it: predicted +0.75 px against the label (+2.1 px below the
  ball's image centre), measured +0.62 / +0.78.

So our ball networks place the ball ~2 px (1280×720) below its centre in the image. Down the image is
depth on court, the camera's weak axis: at the far baseline 2 px is of the order of 20 cm, pulling
bounces towards the camera. The TrackNet benchmark compares against the same labels, which carry
1.4 px of the offset themselves, so it under-reports this. `eval_tracknet.py --centred-reference`,
`calibration_check.py … --centred-reference` and `eval_ball.py --centred-truth` now score against
the label minus (1.5, 1.5) px (`LABEL_OFFSET_PX` in `courtvision/eval/tracknet.py`).

**Constant and speed term, per game** (`scripts/label_offset.py --per-game`, offset = constant + slope ×
velocity). The constant ranges from 0.6 to 1.7 px between games (games 8–10: (1.14, 0.84), (1.33, 1.18),
(0.62, 0.65) px); the slope is −0.10 to −0.19 frame on eight games and ~0 on games 4–5. Only the
constant moves a bounce: a point shifted along its own track stays on the same line, and the reference
contact is where two such lines meet. The centred references (`eval_tracknet.py --centred-reference`,
`calibration_check.py`) therefore subtract each game's constant (`GAME_LABEL_OFFSET_PX`). The ball
networks reproduce the labels' lead (`scripts/detection_offset.py`: slope against the labels ≈ 0).

### Ball network v3 (default until v4, 2026-11-13)

v2's recipe with the corrected blur and `--centre-labels`: targets at the label minus (1.5, 1.5) px,
through `cv2.resize`'s pixel-centre mapping; the epoch is chosen on the selection clips against the same
centred positions. v2 and v3 share their training and selection clips, so they are compared there
(`eval_ball.py --selection`, production detector), not on games 8–10.

**Rule, written before training:** v3 replaces v2 if both hold on the selection clips:
(a) its mean offset from the ball's image centre (`--centred-truth`, `bias_px`) is smaller than v2's
in both x and y (v2: +0.14, +2.28 px); (b) its F1 at 10 px against the raw labels, which a 1.5 px
shift barely touches, is at most 1 point below v2's. Games 8–10 are then run once to report v3, with
both references, not to decide.

12 epochs (~2.9 h); epoch 12, the last, was best on the selection clips (validation F1 at 4 px against
the centred positions 0.895, still rising slowly). v2 is kept as `models/sekme_ball_v2.pt`.

| Selection clips (games 1–7, every fifth clip; 3 012 frames) | v2 | **v3** |
|---|---|---|
| mean offset from label − (1.5, 1.5) px, dx / dy (SE 0.03; the rule's measure) | +0.13 / +2.27 px | **−0.08 / −0.04 px** |
| F1 at 10 px, raw labels | 0.959 | **0.964** |
| F1 at 4 px, centred positions | 0.820 | **0.937** |
| F1 at 4 px, raw labels | 0.893 | 0.844 |
| time per frame | 22.6 ms | 22.7 ms |

**Decision.** Both conditions hold — the vertical offset falls from 2.3 px to under 0.1 px, and F1 at
10 px rises by half a point — so v3 is the default. Condition (a) is close to circular, since the 1.5 px
came from these clips; games 8–10 below are the independent check. (The rule's +0.14 / +2.28 px for v2 was its raw-label
offset plus 1.5 px; measured directly against the centred positions it is +0.13 / +2.27 px.) F1 at 4 px against the *raw*
labels falls, as it must: v3 now places the ball 1.5 px from where those labels sit, which is where
the ball is.

**Games 8–10, run once after the decision** (for the report; the raw reference is the one the earlier
tables use, the centred one subtracts each game's constant label offset, see
[EVAL_TRACKNET.md](EVAL_TRACKNET.md#systematic-offsets-2026-11-13-evidence-in-resultsposition_biastxt)):

| Games 8–10 | v2 | **v3** | WASB |
|---|---|---|---|
| F1 at 4 px / 10 px, raw labels | 0.922 / 0.965 | **0.921 / 0.969** | 0.928 / 0.955 |
| constant offset from the ball's image centre, dx / dy (range over the 3 games) | +0.1 / +1.8…+2.1 px | **0.0 / −0.1…−0.3 px** | +0.5…+0.75 / +0.25…+0.64 px |
| first bounce after a stroke found | 110 / 142 | 109 / 142 | 110 / 142 |
| bounces found / false alarms / precision | 113 / 7 / 94.2 % | 112 / 8 / 93.3 % | 112 / 9 / 92.6 % |
| distance to the reference, median / p90, raw | 16.1 / 37.7 cm | **10.0 / 29.6 cm** | 11.4 / 33.4 cm |
| the same, centred reference | 21.1 / 41.9 cm | **11.4 / 31.3 cm** | 13.3 / 36.6 cm |
| median depth offset (towards the camera −), centred | −19.6 cm | **−5.9 cm** | −9.8 cm |
| decisive calls consistent with the label | 107 / 107 | 107 / 107 | 108 / 108 |
| truth inside the 95 % ellipse (1 px label noise), raw / centred | 89 / 87 % | 92 / 94 % | 91 / 89 % |
| time per frame | 22.5 ms | 22.6 ms | 40.1 ms |

The labels on games 8–10 sit closer to the ball's centre than on games 1–7, so v3 is further from
them than its training assumed, yet within 0.3 px of the ball itself. The bounce positions moved as
the offset analysis predicted: the median depth offset shrank by ~14 cm. With a clip bootstrap
(`scripts/bootstrap_bounce.py`, 30 clips) v3's median distance is 6.1 cm below v2's (95 % interval
2.9–10.6 cm) and **on par with WASB** (−1.4 cm, interval −2.9 to +1.0 cm; centred −1.9, −4.0 to +0.4):
not demonstrably better. One first bounce fewer was found than with v2 (109 vs 110).

**Per-frame precision.** On fast-moving frames of games 8–10, a three-cornered hat between the label,
the ball's image centroid and the detection (`scripts/label_noise.py`, `results/label_noise.txt`)
gives v3's per-frame noise as 0.34 / 0.39 px (x / y) against WASB's 0.48 / 0.51 px, and the labels'
own as ~1 px (0.91 / 1.09) with either network.

**Real broadcast and radar with v3** (`runs/<id>_v3`, `results/radar.txt`). US Open 2023 final, 90 s:
54 calls on 65 shots (v1: 52 on 66), the same 2 of 3 serves; v3 now sees 2 of the 7 blurred frames after
the missed serve's contact (WASB 4). Radar on 5 timed serves: mean |error| 3.9 % (v1 7.7 % on 6, WASB
4.9 %); the fast 60 fps serve v1 measured at −18.7 % is −0.4 %. Serves recognised on the five clips:
11 (v1 13); one of the lost ones had a radar pair (v1 −6.9 %).

Order of events: v3's games 8–10 run was first scored against the fixed (1.5, 1.5) px reference
(v3 13.7 cm, WASB 16.4, v1 23.5, v2 24.4 cm); the per-game constants came after it, following the
review. The ranking is the same under every reference.

### Ball network v4 (current default, 2026-11-13)

v3's recipe unchanged except the length: 20 epochs instead of 12 (one-cycle schedule over the whole
run). v3's selection score was still rising at its last epoch.

**Rule, written before training:** v4 replaces v3 only if all hold on the selection clips
(`eval_ball.py --selection`, production detector; the same clips also choose v4's epoch, as they did
v3's): (a) F1 at 4 px against the centred positions (`--centred-truth`) at least 0.5 point above v3's
0.937; (b) its mean offset from the centred positions within 0.3 px on both axes; (c) F1 at 10 px
against the raw labels at most 0.5 point below v3's 0.964. Games 8–10 are then run once to report,
not to decide.

**First run interrupted (2026-11-13).** The run was stopped from outside after epoch 14 of 20,
when the machine ran low on memory; it was not a training error. Under the one-cycle schedule an
epoch-14 checkpoint comes before the learning rate has annealed, so it is not the planned candidate.
It was not scored on the selection clips, and the rule above stands for a complete 20-epoch run.
For information only: the training log's validation F1 at 4 px was 0.905 at epoch 14 (v3: 0.895 at
its last epoch, 12). v3 stays the default.

**Complete run (2026-11-13, restarted from scratch with per-epoch saved state).** 20 epochs, ~15 min
each; epoch 19 was best on the selection clips (validation F1 at 4 px against the centred positions
0.907; epoch 20: 0.907). Scored once on the selection clips with v3 re-run alongside
(`data/tracknet/eval_selection_v4_{centred,raw}.txt`):

| Selection clips (3 012 frames) | v3 | **v4** | rule |
|---|---|---|---|
| (a) F1 at 4 px, centred positions | 0.937 | **0.943** | ≥ 0.942 |
| (b) mean offset from the centred positions, dx / dy (SE 0.03) | −0.08 / −0.04 px | **−0.11 / −0.04 px** | within ±0.3 px |
| (c) F1 at 10 px, raw labels | 0.964 | **0.966** | ≥ 0.959 |
| F1 at 4 px, raw labels | 0.844 | 0.852 | — |
| time per frame | 22.1 ms | 22.1 ms | — |

**Decision.** All three hold, so v4 is the default (`models/sekme_ball.pt`; v3 is kept as
`models/sekme_ball_v3.pt`). Condition (a) is met by a thin margin: 0.9426 against 0.942, a gain of
0.6 point over v3, measured on the clips that also chose v4's epoch, so it is slightly optimistic.
Games 8–10 below are the independent report.

**Games 8–10, run once after the decision** (report only, as the rule says; `data/tracknet/test_ball_v4.json`,
`centred_v4.json`, `runs_ball_v4`; v3 and WASB from their earlier runs):

| Games 8–10 | v3 | **v4** | WASB |
|---|---|---|---|
| F1 at 4 px / 10 px, raw labels | 0.921 / 0.969 | **0.910 / 0.965** | 0.928 / 0.955 |
| F1 at 4 px, label − (1.5, 1.5) px | 0.940 | **0.931** | 0.906 |
| first bounce after a stroke found | 109 / 142 | **110 / 142** | 110 / 142 |
| bounces found / false alarms / precision | 112 / 8 / 93.3 % | **113 / 9 / 92.6 %** | 112 / 9 / 92.6 % |
| distance to the reference, median / p90, raw | 10.0 / 29.6 cm | **10.3 / 32.4 cm** | 11.4 / 33.4 cm |
| the same, centred reference | 11.4 / 31.3 cm | **11.0 / 32.5 cm** | 13.3 / 36.6 cm |
| decisive calls consistent with the label | 107 / 107 | **108 / 108** | 108 / 108 |
| truth inside the 95 % ellipse (1 px label noise), raw / centred | 92 / 94 % | **91 / 93 %** | 91 / 89 % |
| time per frame | 22.6 ms | 22.6 ms | 40.1 ms |

Clip bootstrap (`scripts/bootstrap_bounce.py`): v4 − v3 median distance +0.2 cm raw (95 % interval
−0.6 to +1.6), −0.5 cm centred (−1.6 to +1.0); v4 − WASB −1.2 cm raw (−2.5 to +1.6), −2.3 cm centred
(−4.2 to +0.2).

**What this shows.** The selection-clip gain did **not** carry over to the unseen matches: frame-level
F1 is about one point lower than v3's at 4 px (half a point at 10 px), most likely because the longer
run fitted games 1–7 more closely. The bounce results, which the calls rest on, are on par with v3's
(one more first bounce, one more false alarm, distance within ±1 cm by bootstrap). Under the rule
games 8–10 do not decide, and reversing the decision because of them would make them a decision set
again, so v4 stays the default; v3 is one file copy away (`models/sekme_ball_v3.pt`). The next ball
network change needs data that has not been used yet.

**Real broadcast and radar with v4** (`runs/<id>_v4`, `results/radar.txt`). US Open 2023 final, 90 s:
53 calls on 66 shots (v3: 54 on 65), the same OUT ball and the same 2 of 3 serves. 2025 final: 22 calls
on 43 shots (v3: 26 on 42). Serves recognised on the five clips: 13 (v3 11, v1 13). Radar on 5 timed
serves: mean |error| 7.0 % (v3 3.9 %): four serves within 0.4 point of v3's, but the fast 60 fps serve
that v3 had brought to −0.4 % is back at −16.5 % (v1 −18.7 %). Together with games 8–10, the
independent checks do not show v4 better than v3; they were not part of its rule.

**v3 vs v4 on serves, decided on unused footage (2026-11-14, [EVAL_SERVE_CONTACT.md](EVAL_SERVE_CONTACT.md)).**
The radar gap above came from one serve: v4 missed the ball from the end of the toss until well after
the contact, so the contact was put on the last toss frame, about ten frames early. A change that puts
such a hidden contact inside the detection gap was tested, with v3 against v4, on 554 s of the 2025 final
nothing had been run on, against 23 serve contacts bracketed by eye by a separate labeller (rules
committed before any output existed). The change was **accepted** (paired mean contact error v3 1.49 →
0.53 frames, v4 2.08 → 0.18; good serves 4 → 6 for both; 3 serves moved, none got worse), and **v4 stays
the default**: with the change both nets have 6 good serves. Re-run with the change on the five radar
clips (`runs/<id>_v4`, `_v3`; no call changes): the QF serve moves from −16.5 % to +11.7 % (v4) and from
−0.4 % to +14.6 % (v3), so radar mean |error| is now 6.0 % for v4 and 6.7 % for v3 on the same 5 serves;
correct contact timing does not make that serve's speed right. The test also measured **serve recall:
6 of 23 for either net**. Most near-side serves are never split from their toss, because the ball net
loses the nearly still ball at the toss apex for 0.2 s and the struck ball (flying away from the camera)
is slow enough in the image to pass for the toss; that is a detector problem, not a ball-net one, and the
next thing to fix.

## SEKME court network

A 5-level U-Net (`src/courtvision/models/court_unet.py`; 32-64-128-256-256 channels, 22×40 bottleneck)
on one 640×360 frame, 14 keypoint heatmaps in the TennisCourtDetector order, disk targets (radius 4 px),
the same normalised weighted BCE and prior init as the ball network. Data: the TennisCourtDetector
labels (8 841 broadcast frames, 14 points each). Their split is by frame (434 of the 442 validation
videos also appear in training), so 10 % of the *training videos* were held out to choose the epoch;
both networks were trained on the same `data_train.json`. Augmentation: flips with left/right keypoint
swap, ±10 % zoom, ±5 % shift, brightness/contrast. Three epochs completed (the run was stopped for
memory); epoch 2 was best on the held-out videos (accuracy 0.979 at 7 px, median 1.51 px).

`data_val.json` (2 211 frames, 30 835 keypoints), through the production detector (blob centroids):

| | SEKME court U-Net | TennisCourtDetector |
|---|---|---|
| accuracy at 7 px (1280×720) | **0.983** | 0.958 |
| precision | **0.986** | 0.961 |
| median distance | **1.44 px** | 2.05 px |
| time per frame (M4, batch 16) | 120 ms | 166 ms |

**Selection rule, written before the end-to-end runs.** Our court network becomes the default if, on the
US Open clips, the share of calibrated frames is at most 1 point lower and the median court-line residual
is not worse, and the TrackNet first-bounce recall (games 8–10) is at most 2 points lower.

The validation split shares videos with training (an independent review flagged it), so the scores were
also computed on two cleaner subsets:

| `data_val.json` subset | SEKME court U-Net | TennisCourtDetector |
|---|---|---|
| videos we held out from our training (281 frames; unseen by ours, **seen** by TennisCourtDetector) | **0.986**, 1.52 px | 0.957, 2.15 px |
| videos absent from all training (8 frames, 112 keypoints) | 0.830, 1.61 px | 0.705, 3.00 px |

End to end (everything else unchanged):

| | TennisCourtDetector | SEKME court U-Net |
|---|---|---|
| US Open 2023 final: calibrated frames / line residual | 93.8 % / 0.34 px | 93.8 % / 0.34 px (calls identical) |
| US Open 2025 final: calibrated frames / line residual | 68.7 % / 0.38 px | **71.6 %** / 0.38 px |
| TrackNet games 8–10: first bounces found, median error, decisive calls | 108 / 142, 15.0 cm, 105 / 105 | 108 / 142, 15.1 cm, 105 / 105 |

**Decision.** All conditions hold; the SEKME court U-Net is the default. The line-ridge refit after the
keypoints (`calib/line_refine.py`) makes the final calibration nearly independent of which network
found the keypoints; the better network shows up as more frames calibrated on the harder 2025 clip.

## SEKME person network (distilled)

A centre-heatmap detector on the shared U-Net (`src/courtvision/models/person_unet.py`; 512×288 input,
one centre map plus log box width and height), our own IoU tracker (tracks reset at camera cuts). No
human box labels exist here for broadcast tennis, so it was **distilled**: the targets are the persons
YOLO26-s finds (imgsz 1280, conf 0.3) in 28 676 frames (TrackNet games 1–10 and the TennisCourtDetector
frames). Training: TrackNet games 1–5 (every second frame) plus the TennisCourtDetector frames; epoch
choice on games 6–7; six epochs, about 1.6 h. Agreement with the teacher on games 6–7 for persons taller
than 60 px (IoU ≥ 0.5): F1 0.74 (precision 0.68, recall 0.81) at the 0.3 threshold
(`data/players/train_players.log`). The low threshold is kept on purpose: the pipeline discards extra
persons on the court plane, a missed player cannot be recovered.

**Selection rule, written before the tests.** It becomes the default if, against the teacher (YOLO26-s):
on the human-labelled Tennis Player Actions frames (below) its player recall is at most 2 points lower;
on TrackNet games 8–10 the first-bounce recall is at most 2 points lower and every decisive call still
agrees with the label.

The human-labelled test set is the Tennis Player Actions dataset (Wang, Lai, Huang & Lin, 2024,
doi:10.17632/nv3rpsxhhk.1, CC BY 4.0): 2 000 frames from a camera high behind the baseline, one
hand-labelled player per frame (box + 17 COCO keypoints). It is used only for testing.

**Result: rejected as the default.** Share of the hand-labelled players found (IoU ≥ 0.5,
`results/players_human_labelled.txt`): YOLO26-s 85.5 %, SEKME person U-Net 63.4 % — 22 points lower,
far outside the rule. The distilled network learned broadcast frames only and does not carry over to a
club court and a different camera the way the COCO-trained teacher does. The end-to-end run was stopped
once the first condition failed. YOLO26-s stays the default; the network remains selectable with
`--model players=models/sekme_person.pt`. Training on human-labelled boxes from many courts would be
the way forward; none with a usable licence was found without an account.

## SEKME pose network (distilled)

A U-Net on the player crop (`src/courtvision/models/pose_unet.py`; 128×192 input cut exactly as at run
time, 17 COCO keypoint heatmaps, left/right swapped under flips). Distilled from YOLO26-s-pose run on the
same crops (imgsz 320, keypoints it is less than 0.5 sure of left out), for persons taller than 60 px in
TrackNet frames (every fourth) and the TennisCourtDetector frames. Split by match as for the other
networks: TrackNet games 1–5 + TennisCourtDetector frames train, 6–7 choose the epoch, 8–10 test.

**Selection rule, written before the tests.** It becomes the default if on the human-labelled Tennis
Player Actions frames its PCK@0.1 (share of hand-labelled keypoints within 10 % of the box height) is at
most 2 points below YOLO26-s-pose's, and on the US Open 2023 clip at least 90 % of the stroke types
(forehand / backhand / volley / overhead / serve) agree with those from YOLO26-s-pose.

**Result: rejected as the default** (`results/pose_human_labelled.txt`). Five epochs (about 80 min);
selection PCK@0.1 against the teacher 0.80. On the hand-labelled frames PCK@0.1 is 0.711 vs 0.806 for
YOLO26-s-pose (−9.4 points; serves 0.62 vs 0.84). On the US Open clip both classify the same 58 shots
with identical totals (27 forehand, 28 backhand) but agree on only 38 (65.5 %): forehand and backhand
swap in both directions, which points at left/right wrist confusion. YOLO26-s-pose stays the default;
the network remains selectable with `--model pose=models/sekme_pose.pt`.

## Third-party networks

- **WASB** (`src/courtvision/models/wasb_hrnet.py`, HRNet code by Bin Xiao, MIT): the reference ball
  network, selectable with `--ball-model`.
- **TennisCourtDetector**: our re-implementation of its network (`models/court_net.py`) loads its public
  checkpoint; the reference court network. Whichever network finds the keypoints, every frame is refined
  by fitting the painted line ridges (`calib/line_refine.py`).
- **YOLO26 / YOLO26-pose** (Ultralytics): person boxes and 17 keypoints through the `ultralytics`
  package; the teachers of our distilled person and pose networks.

Licences: [LICENSES.md](LICENSES.md).
