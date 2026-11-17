# Decisions

Every change below was judged by a rule written down, and committed, before its result was seen. The rule
names the data, the measure and the threshold. Held-out data is used a limited number of times: TrackNet games
8–10 are used up, and the E2E-Spot test matches allow two candidates. A rejection is final and stays
documented: a rejected pipeline change is removed from the code, and a rejected network stays out of the
defaults. Numbers are as measured, failures included. Each
row links to the section that holds the rule, the result and what was learned.

34 decisions: 15 adopted, 11 rejected, 6 not adopted, 1 adopted as a correctness fix, 1 withdrawn.

| Date | Change | Outcome | Evidence | Where |
|---|---|---|---|---|
| before 2026-11-13 | Ball network v1 selection rule | adopted | Both pre-registered conditions hold (F1 at 4 px −1.7 points, first-bounce recall −1.4 points) | [MODELS.md](MODELS.md), SEKME ball network |
| before 2026-11-13 | Ball network v2 motion-blur candidate | adopted | F1 at 4 px 0.912 → 0.922; first bounce after a stroke found 108 / 142 → 110 / 142 | [MODELS.md](MODELS.md), Ball network v2 |
| before 2026-11-13 | SEKME court keypoint network | adopted | US Open 2025 final calibrated frames 68.7 % → 71.6 % (line residual 0.38 px); games 8–10 first bounces 108 / 142 for both | [MODELS.md](MODELS.md), SEKME court network |
| before 2026-11-13 | SEKME person network, distilled | rejected | Share of hand-labelled players found: YOLO26-s 85.5 %, SEKME person U-Net 63.4 % | [MODELS.md](MODELS.md), SEKME person network (distilled) |
| before 2026-11-13 | SEKME pose network, distilled | rejected | On the hand-labelled frames PCK@0.1 is 0.711 vs 0.806 for YOLO26-s-pose (−9.4 points; serves 0.62 vs 0.84) | [MODELS.md](MODELS.md), SEKME pose network (distilled) |
| 2026-11-13 | Ball network v3 label-centred offset | adopted | mean offset v2 +0.13 / +2.27 px, v3 −0.08 / −0.04 px; F1 at 10 px raw 0.959 → 0.964 | [MODELS.md](MODELS.md), Ball network v3 (default until v4, 2026-11-13) |
| 2026-11-13 | Ball network v4 longer training | adopted | (a) F1 at 4 px centred 0.937 → 0.943 (rule ≥ 0.942); (c) F1 at 10 px raw 0.964 → 0.966 (rule ≥ 0.959) | [MODELS.md](MODELS.md), Ball network v4 (current default, 2026-11-13) |
| 2026-11-13 | Serve dead-ball check | rejected | games 1–7 first bounces 220 / 329 before and after; (c) failed: neither of the two uninspected serves came back | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Serve detection: the dead-ball check (rule written before the change, 2026-11-13) |
| 2026-11-13 | Bounce-versus-stroke hit gate | rejected | Games 8–10: 113 / 142 first bounces (baseline 109); 110 / 111 decisive calls consistent | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Hit detection: a bounce taken for a stroke (rule written before the games 8–10 run, 2026-11-13) |
| 2026-11-13 | Bounces in the frame of the lines | rejected | (a) failed: median lateral error 3.17 → 3.21 cm (centred); (b) failed for the raw reference, 10.0 → 11.1 cm (centred 11.4 → 11.2 cm) | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Bounces in the frame of the lines (rule written before the games 8–10 run, 2026-11-13) |
| 2026-11-14 | OCR score column tolerance fix | adopted as a correctness fix | 2402 frames, one every 0.5 s; no reading changed (the second, "0 0" correction was not written) | [ARCHITECTURE.md](ARCHITECTURE.md), Known issues (code review, 2026-11-14) |
| 2026-11-14 | Radar board pairing wide update window | not adopted | default run 1 agreeing before and after; scaled run 8 agreeing and 1 disagreeing before and after (+0, the rule asked for +3) | [ARCHITECTURE.md](ARCHITECTURE.md), Radar board pairing: wide update windows without a cut (rule written 2026-11-14, before measuring) |
| 2026-11-14 | Ball network v5 sensor-noise augmentation | not adopted | (a) fails: clean 0.928 against at least 0.938 (rule and result dated 2026-11-14 and 2026-11-15 respectively) | [MODELS.md](MODELS.md), Ball network v5 candidate: sensor-noise augmentation (rule written 2026-11-14, before training) |
| 2026-11-14 | Ball tracker bug fixes | adopted | 220 / 329 first bounces, precision 89.1 %, 228 / 228 decisive calls consistent | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Ball tracker bugs found in review (rule written 2026-11-14, before the fixes were run) |
| 2026-11-14 | Two scorer corrections | adopted | games 1–7 first bounces found 220 / 329 (66.9 %) → 220 / 323 (68.1 %); decisive calls 228 / 228 both | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Two scorer corrections (decided 2026-11-14, before measuring their effect) |
| 2026-11-14 | Rally-hit timing refinement | rejected | games 6–7 first bounces found 62 / 87 (71.3 %) baseline, 59 / 87 (67.8 %) candidate | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Development candidate: rally-hit timing refinement (rule for the games 6–7 check written 2026-11-14, before it) |
| 2026-11-14 | Image-kink bounce span limit | adopted | no call and no bounce changed on any of the 65 clips; 590 flights, none changed; spurious CRF 40 call withdrawn | [EVAL_TRACKNET.md](EVAL_TRACKNET.md), Image-kink bounce on sparse detections (rule written 2026-11-14, before the change is made) |
| 2026-11-14 | Toss bridge in serve detector (candidate 1) | withdrawn | default detector already finds 18 of the 23 labelled serves there (6 at 60 fps), the candidate 20 (+8.7 points, under the 10-point margin) | [EVAL_SPOT.md](EVAL_SPOT.md), Candidate 1 (withdrawn before any download): toss bridge in the serve detector (rule written 2026-11-14) |
| 2026-11-14 | Pin fit to serve contact height | adopted | Six US Open videos, 89 paired serves: mean \|error\| against the board 10.58 % → 5.74 % (rule −1.5 points), 65 closer, 13 further; no verdict changed (near side 11.07 → 2.97 %, far 9.98 → 9.13 %) | [EVAL_SERVE_SPEED.md](EVAL_SERVE_SPEED.md), Result (scored once, 2026-11-17) |
| 2026-11-14 | Camera-based player size gate, 0.5 band | rejected | Six US Open videos: near coverage 58.2 % → 90.4 %, far unchanged, but wrong selections 2.45 % → 4.20 % of player-frames (rule ≤ +1 point) | [EVAL_SERVE_SPEED.md](EVAL_SERVE_SPEED.md), Result (scored once, 2026-11-17) |
| 2026-11-14 | Frame counts scaled at 60 fps | adopted | Six US Open videos (60 fps), 173 labelled serves: recall 32.9 % → 63.0 % (rule +10 points), within 2 frames 50.9 % → 75.2 %, no decisive flip, false serves 3 → 4 | [EVAL_SERVE_SPEED.md](EVAL_SERVE_SPEED.md), Result (scored once, 2026-11-17) |
| 2026-11-16 | Player names from consistent score readings | rejected | Six US Open videos, 227 labelled flights: wrong near-player names 28 → 23 (rule: at most half, 14), right 138 → 123 | [EVAL_SERVE_SPEED.md](EVAL_SERVE_SPEED.md), Result (scored once, 2026-11-17) |
| 2026-11-17 | Far strokes from a 2D velocity break | adopted | E2E-Spot test slot 1 (4 matches, 303 clips): stroke recall +3.32 points (rises in all 4 matches), stroke precision +0.28, first-bounce recall +0.86, bounce precision +0.31, serve recall 0.00 | [EVAL_SPOT.md](EVAL_SPOT.md), Candidate 2 (test slot 1): far strokes from a 2D velocity break |
| 2026-11-14 | Gap contact change for hidden serves | adopted | paired mean error v3 1.49 → 0.53, v4 2.08 → 0.18; good serves 4 → 6 for both | [EVAL_SERVE_CONTACT.md](EVAL_SERVE_CONTACT.md), Rules |
| 2026-11-14 | Serve net choice, v3 vs v4 | not adopted | With the accepted code v3 and v4 both have 6 good serves: no measurable difference on serves | [EVAL_SERVE_CONTACT.md](EVAL_SERVE_CONTACT.md), Rules |
| 2026-11-14 | Player size gate from camera, 0.6 band | not adopted | worst fall (2023 final 90–179 s, far player) 99.0 % → 97.9 % (−1.15 points), fails (limit 1 point) | [EVAL_PLAYERS.md](EVAL_PLAYERS.md), Rule |
| 2026-11-15 | Player identity from clothing v1 | not adopted | re-scored on 279 flights: graphic 205 right, 72 wrong (25.8 %); clothing 259 right, 20 wrong (7.2 %) | [ARCHITECTURE.md](ARCHITECTURE.md), Player identity from clothing (rule written 2026-11-15, before the test match was looked at) |
| 2026-11-15 | Player identity from clothing v2 | rejected | 54 labelled flights: graphic (default) 50 right, 4 wrong; version 2 14 right, 40 wrong | [ARCHITECTURE.md](ARCHITECTURE.md), Player identity from clothing, version 2 (rule written 2026-11-15, before its test data was looked at) |
| 2026-11-15 | End changes from plausible score readings | rejected | on the 279 reference flights, right 205 → 207, wrong 72 → 62, missing 2 → 10; (a) fails: 62 against at most 36 | [ARCHITECTURE.md](ARCHITECTURE.md), End changes only from plausible score readings (rule written 2026-11-15, before the test below was run) |
| 2026-11-15 | Edits from the same camera position | adopted | histogram test finds 5 (1.7 %) of 294 synthetic edits, candidate 289 (98.3 %); fires on none of 5 806 in-clip pairs | [ARCHITECTURE.md](ARCHITECTURE.md), Edits from the same camera position (rule written 2026-11-15, before the test below was run) |
| 2026-11-15 | Constant-rate copy for variable frame rate | adopted | burst20 speed error 64.8 → 9.4 km/h, time error 0.263 → 0.015 s; drop5 speed error 7.7 → 7.1 km/h | [ARCHITECTURE.md](ARCHITECTURE.md), Variable frame rate |
| 2026-11-15 | Ignoring ball on repeated frames | adopted | (iii) speed error 408.3 → 208.7 km/h (0.51 times, allowed 0.8); all 30 reference calls matched with their verdict in both arms | [ARCHITECTURE.md](ARCHITECTURE.md), Variable frame rate |
| 2026-11-15 | Ball network v6 fine-tuned noise | not adopted | (a) fails: clean 0.933 against at least 0.938 | [MODELS.md](MODELS.md), Ball network v6 candidate: v4 fine-tuned with sensor noise (rule written 2026-11-15, before training) |
| 2026-11-15 | Interlaced video analysed field by field | rejected | summed speed error 74.8 → 86.9 km/h (needed at most half); time 77 → 10 ms, bounce distance 49.9 → 18.2 cm; two window-start calls drive the sum | [ARCHITECTURE.md](ARCHITECTURE.md), Interlaced video |

Outcomes: *adopted* (now the default); *not adopted* (the rule was not met, or, in the first clothing
identity test, a flaw found in the reference showed it was not met); *rejected* (failed and dropped);
*withdrawn* (taken out before its test, the default already did as well); *open* (written and waiting for
data not used yet); *adopted as a correctness fix* (no measured output changed).
