# Hit and bounce segmentation: leads for the next candidate (2026-11-14)

Why: on TrackNet games 1–5 most missed first bounces come from hit/bounce segmentation (`docs/EVAL_TRACKNET.md`,
"Open findings": 28 near-side bounces taken as strokes, 23 fits spanning a missed stroke), not from timing (10).
The one change aimed at it, the near-side reversal gate (`BOUNCE_VX_CHANGE`, commit e46775d), raised first
bounces on all three game sets (games 1–7: 220 → 242 / 329) but was rejected on games 8–10 for one confident wrong
call on an end-of-point bounce near a sideline ("next stroke not seen"). Any revival needs that call's uncertainty
fixed too, and checking a confident-call fix needs labelled bounce *positions* on new footage; the E2E-Spot test
labels carry timing only, so they can test the recall side, not the call safety.

## Literature leads (web scan; numbers from abstracts or third parties are marked unverified)

1. **E2E-Spot** (Hong et al., ECCV 2022, arXiv 2207.10213): end-to-end frame-precise event spotting on video;
   tennis is one of its datasets (exact tennis numbers unverified here). Needs video and training; its tennis
   labels are this project's independent test, so only its train/val matches could be used for training.
2. **A sequence classifier on the 2D ball track** (e.g. the GRU stage of SynthNet, ACM MM 2024): hits and bounces
   from the track alone. Trainable here on the TrackNet per-frame status labels (hit 1, bounce 2) of games 1–7;
   the most direct fit to the two loss groups above. Reported 3D numbers are disputed by a later paper
   (arXiv 2506.05763), unverified.
3. **Choosing split points by fit residual** (as in LATTE-MV, CVPR 2025, and TT3D, arXiv 2504.10035, for table
   tennis): candidate hits and bounces scored by the physics fit's residual, the best segmentation chosen
   jointly (dynamic programming) instead of greedily. Matches the existing 3D fit; no numbers checked.
4. **Audio** (racket and bounce sounds; TennisExpert, arXiv 2603.13397; table-tennis bounce sounds,
   arXiv 2409.11760; numbers unverified). See the check below.
5. **Pose or racket proximity** for hits: strong in padel (CVPRW 2024, F1 about 0.92 reported), weak evidence for
   tennis broadcast footage. Low priority.

## Checked here: audio onsets on the highlight clips (descriptive, nothing changed)

High-pass (> 2 kHz) energy flux peaks, at least 0.25 s apart, compared with the system's detected hits on the
four local clips with sound: about 180 onsets per minute (crowd, commentary), and the hits with an onset within
±3 frames were at or below the count for the same hits shifted by 1–3 s (15 vs 10.8, 21 vs 24.2, 26 vs 31.5,
35 vs 38.3). A simple onset detector adds nothing on highlight audio; an audio cue would need a learned
classifier and cleaner match sound. (The six-video test's downloads are video-only.)

## Suggested next candidate

Lead 2 combined with lead 3: a small classifier on track features (image velocity change, curvature, height
above the court in the image, distance to the nearest player box) proposing hit/bounce labels, used to choose the
segmentation by fit residual. Developed on TrackNet games 1–7 and the E2E-Spot dev matches (once downloaded);
recall tested on E2E-Spot test matches as one of its two slots, with its rule written first; call safety checked
on labelled bounce positions from new footage before it can become the default.
