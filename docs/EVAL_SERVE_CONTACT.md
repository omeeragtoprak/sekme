# Serve contact hidden in a detection gap, and ball net v3 vs v4, on unused broadcast footage

Plan written on 2026-11-14, **before any system output on this footage was produced or seen.**

## Why

- Ball net v4's worse radar figure (mean |error| 7.0 % vs v3's 3.9 %, MODELS.md) comes from one serve:
  US Open 2025 QF, board 175.4 km/h, v4 −16.5 %, v3 −0.4 %. Cause, found on that (development) clip:
  v4 misses the ball in frames 8881–8889, while the racket comes up and the ball passes in front of
  the backdrop lettering. The toss/struck split (`detect_serves`) can only land on a detected frame,
  so it landed on the last toss frame and the contact came out at 8878, about ten frames before the
  contact seen by eye (8887–8888). v3 detected frame 8884 and put the contact at 8885.
- **Candidate change** (`contact_in_gap`, same commit as this plan): when the split lands on the last
  toss detection, the next detection is at least 2 frames later, and the ball covers that gap at least
  twice as fast as the toss moved, the contact is put where the toss line meets the struck line, inside
  the gap (sub-frame). The image point there measures the contact height (as the toss detection did
  before), and the fit's contact-time prior is centred on that estimate with sd 1 frame.
- **Development evidence** (the five radar clips, v3 and v4; contacts bracketed by eye by the developer,
  not blind, development only): the change moves 9 of the 24 serves the pipeline finds; no serve is
  gained or lost and no 2023 radar pair changes. On ten serve contacts bracketed by eye (five serves of
  the 2024 final and the 2025 QF, each with v3 and v4) the pre-change contact is early by 5 frames on
  average (1.7–12.1), mean |error| 6.1 frames; with the change 2.7 (1.5 without one serve whose toss
  is split by a detection gap and which the change does not reach). A first version with a prior as
  wide as the gap (sd gap/√12) let the fit drift a further ~1 frame and was replaced before this plan.
- On the QF serve the speed goes 146.6 → 196 km/h with v4 (+11.7 % against the board) and 174.8 → 201
  with v3 (+14.6 %): contact timing moves serve speed by several per cent per frame, and one serve says
  nothing about speed. The change is judged on contact timing, on footage nothing has been run on.

## Data

- **Footage:** `data/dl/m2025.mp4`, US Open 2025 men's final highlights (official channel, 1080p,
  59.94 fps, 704 s). Its 60–150 s were used for the film and the radar table (`usopen2025_final_hd`);
  nothing else of it has been analysed. This plan uses **150–704 s**:
  `data/videos/usopen2025_final_150-704s.mp4` (stream copy from the keyframe at or before 150 s,
  554 s). OpenCV decodes 33 209 frames, the first at about 150.0 s; the container lists 33 405 because
  196 pre-roll frames are hidden by its edit list (*corrected before any scoring; the first version
  said 33 405*). The pipeline decodes with OpenCV too, so labels and system share frame numbers. Kept
  local, research only, like the other broadcast footage.
- **Labels, blind:** every serve shown in real time from the main broadcast camera (the wide view the
  pipeline calibrates) whose contact can be seen or bracketed is labelled by eye from the raw frames,
  by a separate labeller that has no access to any system output for this clip; other camera angles and
  slow-motion replays are not labelled. Frames are numbered by
  decoding the file in order with OpenCV, as the pipeline does. Per serve: a contact bracket
  [lo, hi] (frames that surely contain the racket–ball contact, as narrow as the frames allow), the
  server's side (near/far), and, when legible, the speed on the stadium radar board. Serves in the
  first 2 s of the clip are skipped (possible overlap with the used segment). The labels are committed
  before any scoring.

## Runs

Ball nets v3 (`models/sekme_ball_v3.pt`) and v4 (`models/sekme_ball.pt`): perception once each (one GPU
job at a time), then analysis from the cached perception with the code before the change (parent
commit) and after it. Four reports; nothing else differs.

## Measurement

- **System serve:** a flight the pipeline called a serve; its contact estimate is `contact_frame` (from
  the fit), else `start_frame`.
- **Matching:** a labelled serve and a system serve of the same server side whose contact estimate is
  within 30 frames (0.5 s) of the label bracket; one-to-one, closest first.
- **Contact error** of a matched serve: 0 inside [lo, hi], else the distance in frames to the nearer
  end. A **good serve** is matched with error ≤ 2 frames.
- **Reported** for all four reports: labelled serves, matched (recall), good serves, mean and median
  error, the per-serve list; radar pairs (system vs board read by eye) as description only.

## Rules

1. **The gap contact change** is accepted if, for both nets, the mean contact error over the labelled
   serves matched both with and without the change (paired) is no higher with it, strictly lower for at
   least one net, and the number of good serves does not fall for either net. Matches gained or lost
   are counted and reported. Otherwise it is rejected: the code is reverted and the rejection stays
   documented here. If fewer than 3 distinct labelled serves change (match or contact estimate, either
   net), this footage cannot decide and the change is not adopted.
2. **v3 vs v4,** with the code chosen by rule 1. Primary measure: good serves. v3 becomes the default
   again if its good-serve count exceeds v4's by at least 2; otherwise v4 stays (it has been the default
   since 2026-11-13), and a difference of 0 or 1 is recorded as no measurable difference on serves.
   Recall, mean error and the radar pairs are reported, not used.

## Amendments before scoring (2026-11-14, after an independent review; no system output on this footage seen)

- Rule 1 judged on the paired set instead of each run's own matched set (a serve gaining a match with a
  large error, or losing one, would otherwise move the mean for the wrong reason); the "fewer than 3"
  clause counts distinct labelled serves, not serves per net.
- The candidate uses the line-meeting image point for the contact height whenever the contact was put
  in a gap (before: only when the rounded contact frame had no detection), as the plan above says.
  Both "new" reports are produced with this code from the cached perception.
- Labels: 23 serves (18 near, 5 far), brackets 2–3 frames wide (21 of 2, 2 of 3; a 2-frame bracket is
  the last frame with a round ball on its toss and the first with a struck, streaked ball), radar
  board read for 22 (mph). Made by a separate labeller between 01:27 and 01:50; perception on this
  clip started at 01:49, so no report, cache or other output for it existed; three brackets spot-checked
  by eye (1933–1934, 16729–16730, 29304–29305). Committed with this amendment:
  `results/serve_contact_labels_2025final.json`.

## Limits

One match, one broadcast, a few dozen serves; contact bracketed by eye at 60 fps (about ±1 frame);
serve speed itself is not tested (the radar board is read where legible and only described).

## Result (scored once, 2026-11-14; `results/serve_contact_2025final.txt`)

Both "new" reports were rebuilt from the cached perception with the amended code (2154198) before
scoring; the "old" reports use the parent commit (07c35b1) on the same caches.

| Net, code | Matched (of 23) | Good (≤ 2 frames) | Mean error | Median error |
|---|---|---|---|---|
| v3, before the change | 6 | 4 | 1.49 | 1.04 |
| v3, with the change | 6 | **6** | 0.53 | 0.38 |
| v4, before the change | 6 | 4 | 2.08 | 0.38 |
| v4, with the change | 6 | **6** | 0.18 | 0.00 |

- **Rule 1: accepted.** Paired mean error (the same 6 serves per net) v3 1.49 → 0.53, v4 2.08 → 0.18;
  good serves 4 → 6 for both; no match gained or lost. Exactly 3 distinct labelled serves changed (the
  minimum the rule asks for), all far-side serves whose contact had been put 2.3–6.2 frames early
  (10343 with v4, 16729 with v3, 25425 with both); every one now lies within 1.1 frames of its bracket,
  and no serve got worse. The change stays in the code.
- **Rule 2: v4 stays.** With the accepted code v3 and v4 both have 6 good serves: no measurable
  difference on serves. They find different serves (v3: 5811 and 16729; v4: 7414 and 10343).
- **Recall, reported, not used:** each net finds 6 of the 23 labelled serves (26 %): of the 5
  far-side serves 3 (v4) or 2 (v3), of the 18 near-side ones 3 (v4) or 4 (v3). Of the 17 misses, 15 (v4) and 14
  (v3) are near-side serves with **no hit detected within 0.5 s of the contact**: one flight runs from
  well before (the toss or a ball bounce by the server; 42–66 frames before the contact for most, 142–356
  for four) through the serve, so toss and serve are never split. The others are far-side serves whose
  hit was found but not called a serve (v4 1, v3 3) or not found (v4 1). This was seen before on one
  serve of the 2025 final clip (`results/radar.txt`); here it is measured: **serve recall on this
  footage is about a quarter, and the cause is the missing hit at a near-side serve contact.**
- **Radar, description only** (board read by eye by the labeller): the three serves the change moved go
  −9.3 → +4.8 % (v4), −5.3 → −0.7 % (v3) and −3.2 / −6.3 → +11.9 / +12.6 % (v3 / v4): better contact
  timing does not by itself make the speed right. Two 91 mph second serves are overestimated by 15–20 %
  (3270 by both nets, 5811 by v3, the only net that finds it), with or without the change. Serve speed
  is not what this test measures.
- **OCR finding (side result):** the pipeline's own board pairing on this clip paired serve 7414 with
  122 mph, but the board showed 112 mph throughout (checked by eye at frames 7385–7925): Vision misread
  "112" as "122" at low confidence in two consecutive samples. Close-ups that show the board from
  another angle also produced a stale reading (130 mph at frames 6480–6510 while the board showed 91).
  Both make spurious board updates. Follow-up (development, not part of this test): a board value whose
  previous value is back within 4 s is dropped (two serves never fall that close), which removes both
  cases and the false pair; no update changes on the five radar clips (`scoreboard.BOARD_FLICKER_S`,
  `results/radar.txt`). Old boards shown in replays for longer than that remain possible.

**What this shows.** Putting a hidden serve contact inside the detection gap removes the early-contact
error it was designed for, on footage nothing was tuned on, for both nets. It is a small test (3 serves
moved), so it shows the direction, not the size. The larger problem it exposes is serve recall: most
near-side serves are never split from their toss.

**Afterwards (development, this footage no longer counts as unused for serve detection).** Followed
frame by frame, the near-side misses share two causes: the ball net loses the nearly still ball at the
toss apex for 11–15 frames, which splits the toss into two runs (the serve detector bridged at most 2
missed frames), and the struck ball, flying away from the camera, moves only 0.05–0.08 player heights
per frame, so it passes the toss gates and stretches the second run past the contact. A candidate that
bridges up to 0.3 s and looks for the contact over the whole run is off by default and waits for its
test on unused matches: candidate 1 in [EVAL_SPOT.md](EVAL_SPOT.md).
