# Event detection on independent broadcast matches (E2E-Spot tennis labels)

Plan written on 2026-11-13, **before any video was downloaded or any system output on these matches was
seen.** Nothing below has been run yet.

## Why

The TrackNet games 8–10 have been used for many accept/reject decisions (see EVAL_TRACKNET.md), and the
loss analysis there shows that most missed first bounces come from hit/bounce segmentation, not from
bounce timing or position. Testing a segmentation change honestly needs labelled matches the project has
never touched.

## Data

- **Labels:** E2E-Spot tennis (Hong et al., *Spotting Temporally Precise, Fine-Grained Events in Video*,
  ECCV 2022), `data/tennis/{train,val,test}.json` of github.com/jhong93/spot (BSD-3-Clause). Per point
  clip, the frame of every bounce, swing and serve, tagged near/far court. Timing only, no positions.
  Split by match: 9 test matches (1622 clips: 9312 bounces, 8282 swings, 1579 serves), 19 other matches
  for train/val.
- **Videos:** full-match broadcasts on the US Open and Wimbledon YouTube channels (ids in the repo's
  `videos.csv`). Kept local for research, not redistributed, like the other broadcast footage here.
- **Resolution: 720p** (H.264), the highest the matches' uploads offer here; 25 fps (Wimbledon) or
  29.97 fps (US Open), as in the labels. None of these matches overlaps TrackNet or the project's own
  footage. *Amended before any download:* the first version of this plan said 360p, because an older
  download tool on this machine offered nothing better; the project's own (newer) tool offers 720p.
- **Disk:** matches are processed one at a time; a match's video is deleted once all its clips are
  analysed, and each clip is deleted once analysed.

## Matches (chosen by download size, ranked on the first 360p probe; the only criterion, as local disk space is limited)

| Use | Match | Clips | ≈ MB (720p) |
|---|---|---|---|
| dev | usopen_2019_womens_round1_williams_sharapova | 97 | 249 |
| dev | usopen_2021_womens_semifinal_raducanu_sakkari | 140 | 302 |
| dev | wimbledon_2019_womens_round1_gauff_williams | 142 | 532 |
| test | wimbledon_2019_womens_final_halep_williams | 64 | 310 |
| test | usopen_2019_womens_osaka_gauff | 70 | 351 |
| test | usopen_2019_womens_final_andreescu_williams | 91 | 584 |
| test | wimbledon_2018_womens_quarterfinal_williams_giorgi | 78 | 689 |

Every clip of each listed match is used (dev: train and val clips together). All seven are women's
matches, because the men's best-of-five videos are the long ones; that limits how far the numbers
generalise.

## Measurement (`scripts/eval_spot.py`, `courtvision.eval.spot`)

- **Clips** are written by decoding each full match once, in order, with OpenCV, the way the labels
  number frames (the repo's `frames_as_jpg.py`); no seeking, so the clip's first frame is the labelled
  start frame by construction (`tests/test_spot_cut.py`: B-frames, a 1.5 s start offset, overlapping
  clips). The run stops if the video's frame rate differs from the labels' (a different upload) or the
  video ends before the last clip. Each clip is analysed (`sekme analyze --no-render --no-broadcast`,
  default models) and then deleted. Residual risk: if the labellers' own OpenCV seek drifted, labels
  and frames would disagree by a constant; the median stroke timing gap is reported to show it.
- **System events:** strokes are the hit detector's hits (`report.hits`). Serves are counted two ways:
  `serve`, the flights the pipeline called a serve (contact frame; this is what sets the service box,
  and it needs a fitted flight), and `serve_toss`, hits the toss detector flagged as serves. Bounces
  are flights with a measured bounce, from the 3D fit or the image fallback (the ones that get a call);
  a bounce's side is the half it landed on (court y), not the receiver.
- **Matching:** one-to-one, closest gaps first, within **0.1 s** for bounces and strokes (the TrackNet
  bounce tolerance) and **0.5 s** for serves (the Review Room's serve labels). A labelled serve is also
  a stroke. Recall at 0.05 and 0.2 s is shown for bounces and strokes, descriptively.
- **Reported:** per kind recall, precision, F1, side agreement of matched pairs and the median timing
  gap; for bounces also the recall of *first* bounces after a stroke; per match and pooled; failed
  analyses; the share of calibrated frames; labelled events per clip per match (so a skew from the
  size-based selection is visible); code commit, model hashes and the sample file's hash.
- **Uncertainty:** with three or four matches per set, the honest spread is the per-match values, not
  a bootstrap over so few clusters.
- **Failed analyses** count as clips with no system events, and are listed. Scoring refuses to run
  until every clip has either a report or a failure record.

## Rules

1. **Baseline:** the current default system is run once on the dev and the test matches and reported.
   The baseline decides nothing.
2. **Changes:** event-logic changes are developed on the dev matches only. Before its single run on
   the test matches, each candidate gets its own rule written here, of this form: pooled first-bounce
   recall rises by at least a stated margin **and** rises in every test match; pooled bounce precision
   falls by no more than a stated margin; pooled serve recall (flight serves) does not fall by more
   than a stated margin. (Labels carry no positions, so decisive-call checks stay with TrackNet and the
   Review Room labels.)
3. **At most two candidates** are ever run on these test matches; after that they count as used, like
   TrackNet games 8–10, and a further change needs other matches.
4. A test result that rejects a candidate is final for it; its code is removed and the result stays
   documented.

## Candidate 1: toss bridge in the serve detector (rule written 2026-11-14, before any E2E-Spot download)

**Change** (`sekme analyze --toss-bridge`, `serves.TOSS_BRIDGE_S = 0.3`; off by default): two toss runs
separated only by missed frames, at most 0.3 s of them, are joined when the ball rises into the gap and
falls out of it (the gap holds the apex); the serve contact is searched over the whole run, with the
toss checks on the detections before it (vertical travel counted as rise plus fall) and the direction
check on those after it. Two refinements were made on the same development footage before this rule
was written: the apex condition (a first version joined any gap and lost a far serve whose struck ball
was missed right after the contact) and the rise-plus-fall measure (a bridged toss drifting 36 px
sideways failed the old range-based check, US Open 2024 final serve 1409).

**Why** (development data only, `docs/EVAL_SERVE_CONTACT.md`): on the US Open 2025 final (150–704 s,
60 fps) the pipeline found 6 of 23 eye-labelled serves and missed 15 (v4) of the 18 near-side ones. On
those the ball net loses the nearly still ball at the toss apex for 11–15 frames (the toss splits in
two), and the struck ball, flying away from the camera, moves only 0.05–0.08 player heights per frame,
so it passes the toss gates and stretches the second run past the contact.

**Development check, full pipeline, ball v4** (`runs/<id>_v4_bridge` vs `runs/<id>_v4`):

| Clip (60 fps unless noted) | Serves | Calls (OUT) | By eye |
|---|---|---|---|
| 2025 final 150–704 s | 7 → 22; labelled 6 → 21 of 23, all within 2 frames | 123 → 136 (2 → 2) | the extra serve (13748) is real, filmed by a low camera the labels leave out; the mid-rally false serve (9572) is gone; 3 lost calls were the serve contact taken as a bounce, 13 gained are serve bounces in the service boxes, 5 gained rally calls unchecked |
| 2025 QF | 4 → 4 | 75 → 76 (0 → 0) | +8017 (real), 3689 → 3662 (same serve, contact 27 frames late before), −9388 (the far player still bouncing the ball) |
| 2024 final | 4 → 5 | 66 → 65 (0 → 0) | +5423 (real); two false serves split at a toss apex (1378, 3586) are no longer serve hits |
| 2025 final 60–150 s | 1 → 2 | 22 → 23 (0 → 0) | +4679 (real) |
| 2023 final, both clips (30 fps) | 2 → 2, 2 → 2 | no change | — |

That footage now counts as development data for this candidate; it says little about 25–30 fps video,
where the struck ball moves twice as far per frame.

**Rule for its single run on the four test matches.** Measures from `courtvision.eval.spot` /
`scripts/eval_spot.py score` (pooled over the test clips, and per match). The candidate is accepted
only if all hold:

1. **Serve recall** (flight serves matched within 0.5 s, in percentage points) rises by at least
   min(5, half of the baseline's missed share) *(amended, see below; first version: min(10, …))*; and in
   **every** test match it is strictly higher than the baseline's (at least one more labelled serve
   matched). Matched counts are reported with the percentages.
2. **Serve precision without likely faults** (`precision_excl_faults`: a system serve with no label but a
   labelled serve of the same side later in the same clip within 30 s is set apart, since the labels
   carry fewer serves than clips and probably leave faults out) falls by no more than 5 points. Plain
   precision is reported, not used.
3. **Pooled first-bounce recall** and **pooled bounce precision** each fall by no more than 1 point.

Otherwise it is rejected, its code is removed and the result stays here.

**How it is run.** `eval_spot.py run --candidate-flag toss-bridge`: on each clip the candidate is
analysed right after the baseline, from the baseline's cached detections of the same clip file, before
the clip is deleted, so the two runs differ only by the candidate's code path. The dev matches are run
first; any change to the candidate after seeing them is written here, with its reason, before the test
matches are run, and the test pass uses the commit recorded in its results' provenance. A dry run on
local footage with made-up labels (2026-11-14, nothing kept) went through cutting, the baseline, the
candidate from the cached detections (perception 0.02 s against 20.5 s for the baseline) and scoring.

**Notes.** This candidate targets serves, so its primary measure is serve recall, not first-bounce
recall as in rule 2's template; first bounces are a guard. The bridge is 0.3 s (8 frames at 25 fps, 9
at 30); the 2-frame run gap and the 3-frame contact window stay in frames as in the default detector.
With the bridge, the vertical check counts rise plus fall for every toss run, bridged or not. The
"slow struck ball" cause may matter less at 25–30 fps (twice the image motion per frame), so the gain
there may be smaller than on the development clip; the margin in rule 1 shrinks with the baseline's
headroom for that reason. **Serve speed is not part of the rule:** on the development clip the newly
found serves' fitted speeds are worse (mean |error| against the eye-read board 9.6 % on 5 serves → 14.9 %
on 20); if accepted, near-side serve speeds are reported with that caveat until a speed check passes.
This is the first of the two candidates the test matches allow (rule 3).

**Amendment before any download (2026-11-14, afternoon): margin 10 → 5 points, made knowing a development
estimate.** The rule above was written from 60 fps development data only. To see what to expect at the
test matches' 25–30 fps, every second frame of the same development clip (US Open 2025 final, 150–704 s)
was made into a 30 fps video and analysed: the default detector already finds 18 of the 23 labelled serves
there (6 at 60 fps), the candidate 20 (+8.7 points). A margin of 10 points would reject a change that adds
real serves without false ones; 5 points with a rise in every match is the bar for "worth adopting". This
amendment was made after seeing that estimate, on development data, and is stated as such.

Why 60 fps was so much worse (development finding, not part of this rule): the ball network does detect
the ball at the toss apex (scores 12–15 on the missed frames), but `drop_court_fixed`, which removes
candidates that stay on one court spot (painted marks), counts frames (≥ 10 of ±15): at 60 fps a ball
hanging at the apex for 0.17 s qualifies. The serve detector's run gap (2 frames) and toss speed limit (per
frame) are frame-based too. Scaling these by round(fps / 30), as the ball network's input stride already
is, finds 20 of 23 serves at 60 fps without the bridge; it also moves some 60 fps bounces and calls, so it
is a separate candidate for later, with its own test.
