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

## Candidate 1 (withdrawn before any download): toss bridge in the serve detector (rule written 2026-11-14)

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
   min(10, half of the baseline's missed share); and in **every** test match it is strictly higher than
   the baseline's (at least one more labelled serve matched). Matched counts are reported with the
   percentages.
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
It was to be the first of the two candidates the test matches allow (rule 3); it was withdrawn before any download (below), so both remain.

**Withdrawn before any download (2026-11-14).** To see what to expect at the test matches' 25–30 fps,
every second frame of the development clip (US Open 2025 final, 150–704 s) was made into a 30 fps video:
the default detector already finds 18 of the 23 labelled serves there (6 at 60 fps), the candidate 20
(+8.7 points, under the 10-point margin). A first reaction lowered the margin to 5 points (commit
53db63d); an independent review judged that fitting the bar to the estimate, and it is withdrawn. Instead
the candidate is withdrawn from these matches: its main effect is a 60 fps one, which a better-founded
candidate now addresses (below), and neither of the two test slots is used. The toss bridge stays in the
code, off by default, and is reported on the six US Open videos of `docs/EVAL_SERVE_SPEED.md`.

Why 60 fps was so much worse (development finding, not part of this rule): the ball network does detect
the ball at the toss apex (scores 12–15 on the missed frames), but `drop_court_fixed`, which removes
candidates that stay on one court spot (painted marks), counts frames (≥ 10 of ±15): at 60 fps a ball
hanging at the apex for 0.17 s qualifies. The serve detector's run gap (2 frames) and toss speed limit (per
frame) are frame-based too. Scaling these by round(fps / 30), as the ball network's input stride already
is, finds 20 of 23 serves at 60 fps without the bridge; it also moves some 60 fps bounces and calls, so it
is a separate candidate for later, with its own test.

## Runner and scorer corrected before any download (2026-11-14, after an independent review of the code)

A review of `scripts/eval_spot.py` and `courtvision.eval.spot` against this plan, before anything was downloaded,
found the matching and the recall / precision arithmetic as written, but nothing enforcing what surrounds them.
Corrected; no rule changed:

- **The samples are this plan's matches.** `sample --name dev3` and `--name test4` take exactly the matches of
  the table above, with their clip counts checked against it, and a sample file is never redrawn. Labels are
  validated (known event labels, frames inside their clip) and the label files' SHA-256 recorded.
- **At most two candidates, enforced.** A candidate runs on the test matches only as a slot (1 or 2) of the
  committed ledger `results/spot_test_candidates.json`, added with its rule before its run; a third slot is
  refused. Its runs go to fixed folders: the current default from scratch (`runs_spot_base_slot<N>`) and the
  candidate from that run's cached detections of the same clip files (`runs_spot_cand_slot<N>`), so the two
  differ only by the candidate's code path, as described above. The baseline report of rule 1 runs to
  `runs_spot` and decides nothing. A candidate run needs the test match videos again (they are deleted after
  the baseline report).
- **Every run names its code.** Runs start only from committed analysis code; each clip records the commit it
  was analysed with (taken before its analysis; a clip during which the code changes is discarded). Scoring
  checks every clip: made from the analysis code at HEAD, with exactly its flags, with as many frames as its
  labels, with the same models as all other clips; refuses uncommitted scorer code; writes a test result once.
- **A broken setup does not become missed events.** A clip whose analysis raises is recorded as failed and
  scored as a clip with no system events, as written above; any other exit (a usage error such as a mistyped
  flag, a killed process) stops the run and records nothing, and so do three failed clips in a row or more than
  10 % of a match's clips, before the match video is deleted.
- **The match videos are identified.** Each video's SHA-256, size and frame rate are recorded before its first
  clip is cut; a video that is not 720p, or not the file recorded before, is refused.
- **Tolerances at the nominal frame rate.** "Within 0.1 s, the TrackNet bounce tolerance" (±3 frames at 30 fps)
  would, counted at 29.97 fps, be 2.997 frames, so a stroke 3 frames off would match on a 30 fps file and not on
  the US Open's 29.97 fps ones. Seconds are now counted at the nominal rate (29.97 fps as 30): ±3 frames on the
  US Open matches, ±2.5 on Wimbledon's 25 fps, ±15 and ±12.5 for serves; the 30 s fault window likewise.
- A bounce without a court position no longer stops scoring (its side is "unknown"); matched counts are printed.

For a candidate, `score --name test4 --slot N` also prints the baseline-against-candidate numbers its rule uses
(pooled and per match, in points, with matched counts); the decision is the candidate's written rule applied to
them.

A second review of these corrections found no blocker and three gaps, closed the same day, still before any
download: **a spent test pass cannot be repeated** (test results are written to `results/spot/` and committed; a
result that exists or ever existed in the history marks the baseline report or that slot as spent, so deleting
files does not free it; a slot without flags is refused); **nothing is shown before everything is checked** (a
run that fails a check no longer prints its numbers first); **a candidate's second download is usable**
(YouTube may serve the same upload re-encoded: a later file is accepted when its size, frame rate and frame
count equal the first one's, every runs folder records which file it used, and a slot's baseline and candidate
share one). Also: a baseline that failed on a clip fails its candidate there too (a candidate on a failed
baseline's cache would not be comparable); the runs must use the committed default networks; a hand-edited
sample file is refused; the comparison is printed. A candidate that fails on clips is scored as written (those
clips have no events), without the stop that guards the baseline, since its failures are its own behaviour.

A third check found no blocker or major gap. Closed from it: `score` on the test matches now writes the result
and commits it (that one file) **before printing anything**, so a pass is spent from the moment its numbers can
be seen; each clip's run records the SHA-256 of the match file it was cut from (instead of the manifest listing
runs folders, which an aborted first attempt could leave ambiguous), and scoring requires every clip of a match
in a runs folder to come from one recorded file; a missing default network is refused. Known limit, not
changed: a re-download is recognised by its size, frame rate and container frame count, so an upload of the same
length with shifted content would pass; the reported median timing gap per match would show such an offset.

A second review (2026-11-16, still before any download) read `src/courtvision/eval/spot.py` against the
measurement and rules above and found it doing what they say; its edges (the 0.1 s and 0.5 s windows and the
sweep at 25 and 29.97 fps, the fault window, pooling, F1 and the undefined cases) are pinned in
`tests/test_scorers_edges.py`.

## Candidate 2 (test slot 1): far strokes from a 2D velocity break (rule written 2026-11-16, before any download)

**Change** (`sekme analyze --far-stroke-breaks`; off by default; `events/hits.py` `far_breaks`, `add_far_breaks`):
the hit detector looks for a reversal of the ball's image-y direction, but a far player's stroke towards the near
side keeps rising in the image after contact, so the reversal comes at the apex and many far strokes are missed or
found late. The candidate also marks a far hit where the detected ball's image velocity, fitted over 3 detected
frames on each side (0.1 s rounded up: 3 at 25 and 30 fps, 6 at 60), turns by more than 60° or changes speed by more than 2.5 times next to the far
player's box, merging breaks within 0.2 s; such a hit is added only where no hit is within 8 frames at 30 fps.

**Why** (development data only, TrackNet games 1–7, events re-run from their caches;
`devir/scratch/far_breaks_e2e_dev.py`): labelled strokes with a hit within 3 frames 269 → 285 of 374 (far strokes
161 → 177 of 253); first bounces 220 → 223 of 323; decisive calls 228 → 232, all consistent with the labels both
ways; false alarms 29 → 30; hits 434 → 463. Of the 97 strokes missed before, 87 were by the far player with the
ball well detected (`docs/ARCHITECTURE.md`, Known issues). Thresholds were chosen on those games. It was developed
on TrackNet, not on the dev matches of rule 2: these test matches are its first contact with E2E-Spot footage.

**Rule** (the form of rule 2 is for bounce changes; this candidate changes strokes, so its own rule is):
on the four test matches, default against `--far-stroke-breaks`, it becomes the default only if all hold:

1. pooled **stroke recall** (0.1 s) rises by **at least 2 points**, rises in **at least 3** of the 4 test matches
   and falls in **none**;
2. pooled **stroke precision** falls by **no more than 2 points**;
3. pooled **first-bounce recall** falls by **no more than 1 point**, and pooled **bounce precision** by no more
   than 1 point;
4. pooled **serve recall** (flight serves) falls by **no more than 1 point**.

Otherwise it is rejected: its code is removed and the result stays here. It uses test slot 1
(`results/spot_test_candidates.json`). Reported, not used: the side agreement and timing of the matched strokes,
far and near strokes separately where the labels name the side, and the dev matches' numbers. The scorer's
comparison now also lists stroke recall and precision (`scripts/eval_spot.py` `compare`), added for this rule.

**Amended before any download (2026-11-16, after an adversarial review).** "Points" are absolute percentage points,
decided on the unrounded `change_points` of the result file, not on the rounded printout. The window was 2 detected
frames at 25 fps (`round(2.5)` is 2), not the 3 the text gave; it is now 0.1 s rounded up, 3 at 25 and 30 fps, which
leaves the TrackNet numbers above (30 fps) unchanged. Criterion 1 asked only that no match fall, which a candidate
that only adds hits nearly always meets; it now also needs a rise in at least 3 of the 4 matches. The candidate was
developed on TrackNet rallies, not on these dev matches (rule 2), and E2E-Spot clips also show servers bouncing the
ball before a serve: before its test run it is run on the dev matches (no slot); their numbers are recorded here,
and any change they lead to is written down, with its rule, before the test. Strokes are matched by time only;
their side agreement is reported, not used.

**Development, reported only (2026-11-16): the six v4 broadcast runs** (US Open 2023–2025, from their caches;
`devir/scratch/far_breaks_broadcast_dev.py`): hits 555 → 592, flights 590 → 626, calls 367 → 366, of which 352
on the same bounce in both runs; no call turned from IN to OUT or back. Nothing there has stroke labels, so this
only shows the side effects on real broadcasts are small; it decides nothing.

**Dev matches, then one change, before the test (2026-11-16).** As the amendment asks, the candidate was first run on
the three dev matches (379 clips; baseline `runs_spot_dev`, candidate from the baseline's detection caches;
`data/spot/results_dev3_runs_spot_dev.json`), with the code of commit `33f67ee`. Against the baseline, in points:

| dev3 (379 clips) | stroke recall | stroke precision | first-bounce recall | bounce precision | serve recall |
|---|---|---|---|---|---|
| as first written | +3.5 (+3.9, +3.9, +2.8) | −1.5 | −0.8 | −1.0 | 0.0 |
| changed (below) | +3.4 (+3.7, +3.9, +2.7) | +0.4 | +0.7 | +0.1 | 0.0 |

(Baseline: stroke recall 0.743, precision 0.421; first-bounce recall 0.695; bounce precision 0.807; serve recall
0.772. Per-match stroke recall in the order Williams–Sharapova, Raducanu–Sakkari, Gauff–Williams.) As first written
it would have been at the edge of criterion 3 (bounce precision −1.0006 points unrounded). Looking at what it
added (`devir/scratch/far_breaks_spot_dev_diag.py`, `far_breaks_spot_features.py`, `far_breaks_spot_features_eval.py`):
of the breaks it adds, 122 lie at a labelled far stroke, 52 at a labelled far bounce and 129 elsewhere; a bounce in
front of the far player turns the ball's image velocity too, and it happens at the player's feet, a stroke at racket
height. With the ball's height above the far player's box bottom in box heights, labelled far strokes have a 10 %
quantile of 0.56 (median 1.32), labelled far bounces a 90 % quantile of 0.06.

**The change** (commit `a8d0d4d`, `events/hits.py` `BREAK_MIN_HEIGHT`): a merged break with the ball lower than
0.3 box heights above the far player's box bottom is dropped. On the dev matches it keeps 120 of the 122 stroke
breaks, none of the 52 bounce breaks, and 47 of the 129 others. The threshold was chosen on these dev matches, so
their numbers above are optimistic; the test matches have not been touched. On TrackNet games 1–7 (from their
caches, `devir/scratch/far_breaks_e2e_dev.py`) the changed candidate gives labelled strokes with a hit within 3
frames 269 → 284 of 374 (far 161 → 176; first written: 285, 177), first bounces 220 → 226 of 323 (first written:
223), decisive calls 228 → 235, all consistent with the labels, false alarms 29 → 30, hits 434 → 454.

**The rule is unchanged**: criteria 1–4 above, on the four test matches, decided once on the unrounded
`change_points`; slot 1 runs the candidate as changed here (the code at the commit that records this text).

**Known limitation, written before any test result (2026-11-16, after an independent review of the change).** The
height test sees only the strongest break of each 0.2 s merge, so a stroke break merged into a stronger one at the
feet is lost with it. Filtering before the merge would, on the dev matches, give 27 more breaks, 9 of them at a
labelled far stroke (some possibly the same stroke twice). It was left as measured: changing it now would be a
second, unmeasured change, and the loss works against the candidate, not for it. The slot-1 run uses the code as
recorded above (analysis code of commit `a8d0d4d`).
