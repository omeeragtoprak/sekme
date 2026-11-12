# Models

SEKME uses four networks. The ball and court networks were designed and trained in this project and are
the defaults. Person and pose networks of our own were distilled from YOLO26 and tested on hand-labelled
frames; both fell short of their teacher, so YOLO26 stays the default for those two roles.
Everything after the networks (court refit and camera, ball tracking, events, the 3D physics fit, calls
and their uncertainty, self-checks) is this project's code; see [ARCHITECTURE.md](ARCHITECTURE.md).

| Role | Default network | Trained by | Training data | Alternative (`--model KEY=PATH`) |
|---|---|---|---|---|
| Ball | SEKME ball U-Net (`models/ball_unet.py`) | this project | TrackNet tennis, games 1–5 | WASB HRNet (its authors, MIT) |
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
