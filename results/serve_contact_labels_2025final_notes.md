# Serve contact labelling, usopen2025_final_150-704s.mp4

Blind: only the video and own scripts were used (no runs/, results/, docs/, no detector, no pipeline).

## Method
1. Pass 1: one sequential cv2 decode, every 15th frame at 320x180 with frame number, 8x6 contact sheets
   (46 sheets), viewed by eye to find main-camera shots and serve moments.
2. One sequential decode dumped every frame as 960x540 jpg (name = sequential index); used only for browsing
   (crop around the server at step 3-4 frames) to find the coarse contact time of each candidate.
3. Pass 2: one sequential full-resolution decode, crop of +-8 frames around each serve, every frame, 300x400 (near) or
   240x210 upscaled 1.6x (far); brackets read there. Far serves and ambiguous cases re-checked at 3.4-3.6x zoom
   (zoom.py). Bracket rule: last frame with a round, undisturbed ball on its toss path = lo; first frame where the ball is a
   motion streak leaving the racket = hi (usually lo+1: the first streaked frame). If the ball is hidden in the
   racket blur the bracket is widened to 3 frames (21426-21428, 25425-25427).
4. Radar: stadium speed board (IBM "SERVE SPEED", wall behind far baseline) cropped at full resolution every 60 frames
   from c-30 to c+510. The board updates 90-210 frames (1.5-3.5 s) after contact; a reading is recorded only if the
   number changes between samples after the contact (frame = first sample showing the new number). No change => null.
5. Helper scripts in this folder: pass1.py, pass1b.py, feat.py (motion features used only to suggest candidates), win.py,
   fine.py, zoom.py, board.py; cand_notes.txt = working log.

## Frame count
cv2 returns 33209 frames; the container says 33405 (ffprobe nb_frames). All numbers are from the cv2 loop
(frames beyond 33208 are an end card, nothing labelled there). JSON keeps n_frames=33405 as requested plus
n_frames_decoded_by_cv2=33209.

## Result
23 serves: 18 near, 5 far. Radar read for 22 of 23.

## Skipped / not labelled
- Serves shown only by other cameras: low tracking camera at ~13725-14040 (Alcaraz serve ~13760-13790), closeup-only
  toss views (3180-3240, 14070-14145, 26280-26385 are closeups of serves that are also labelled from the main camera).
- 7930-7993: server (Alcaraz, far) only bouncing the ball, cut to closeup at 7995 before the serve.
- Slow-motion replays: none identified.
- Serves < frame 120: the serve at 187 is the first one; ball bouncing before it is earlier.

## Difficulties / uncertainty
- Far serves: ball ~6 px, against the backdrop; 16729 (ball vanishes at 16730) and 25426/26451 are least certain.
- 29304: very late contact (ball hangs, server stalls), easy to mis-date as ~29293.
- 873 and 187 follow an edit (dissolve at ~686 / clip start): time jumps, board values before the jump belong to other points.
- Completeness: far serves are the hardest to spot; scanning was by 15-frame sheets plus a motion-onset helper; a missed
  far serve in a long main-camera shot is possible (checked 2000-2200, 12200-12320, 17240-17440, 20260-20380: rallies).
- Board 7415: reads 112 before and after the serve (same number), so no radar attributed.
