# Blind labels: m2025_sf_alcaraz_djokovic (Alcaraz v Djokovic, 2025 US Open SF)

Start 2026-11-16 19:34, end 2026-11-16 20:21 (local).

## Method
Only the video, the template/helper scripts and the presence frame list were opened. One sequential cv2 decode
(37389 frames, same as the frame list's n_frames) gave contact sheets (every 15th frame, 320x180) to find main-camera serves.
Then sequential decodes cropped around each server: coarse crops every 4 frames, then every frame +-6 at 1.6x (near) /
2x (far). Bracket: lo = last frame with a round ball on its toss path, hi = first frame with a streak (usually lo+1).
Radar: IBM board cropped at contact -30, +45, +90, +135, +180, +240, +300, +420, +540; value = first new number.
Presence: the 80 listed frames were saved at full resolution during the first decode, then labelled on grid-overlaid
crops (far player 5x, near player 3x or overview); feet = point between the feet on the ground (ground foot when
the other is airborne). Near feet are accurate to roughly +-8 px, far feet to +-4 px, fast/blurred frames worse.

## Result
22 serves: 12 near, 10 far; radar read for 21, null for 1 (5994: cut to closeup 135 frames after contact).
Presence: 47 main, 33 not main (replays, closeups, low tracking camera, end card).

## Servers (from clothing/board)
Far: Djokovic (174-176, 12330, 16183, 17136, 29466, 31906), Alcaraz (5994, 13276, 17952, 23614).
Near: Djokovic (1276, 7144, 8598, 14405, 20044, 20890, 33995), Alcaraz (2057, 5118, 9964, 25662, 28310).

## Uncertain
- 174-176 (far Djokovic): ball hidden against the white US Open logo; bracket widened.
- 5994-5996 (far Alcaraz), 23614-23616 (far Alcaraz), 29466-29468 (far Djokovic): ball hidden in racket blur, widened to 3 frames.
- 20890-20892 (near Djokovic): ball at 20891 looks half round half streak; widened.
- Radar of 5994: not readable (cut at +180). Radar of 29466: board showed the previous serve's 110 until about +90, new value 75.

## Skipped
- Serves whose contact is only in closeup or a dissolve: Alcaraz near ~33205-33215 (dissolve from closeup into the wide
  view, follow-through already visible at 33212), Alcaraz far ~31110-31125 (closeup, cut to wide after the toss).
- Low tracking-camera views (e.g. 4087-4384, 26340-26385, 30180-30225) and replays: none labelled.
- Rallies/returns checked and rejected as serves: ~225, ~1390, ~10050, ~17985, ~18150, ~20580, ~25780, ~33480.
- Completeness: first 24000 frames were scanned by 15-frame sheets; a missed far serve in a long wide shot is possible.
