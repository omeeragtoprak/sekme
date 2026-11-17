# w2024_final_sabalenka_pegula: blind labels (serves, presence)

Labeller work window: 2026-11-16 20:19 to 2026-11-16 21:30 (local time). Only the video, the brief/template, the helper
scripts (copied and adapted in a scratch folder outside the repository) and `w2024_final_sabalenka_pegula_presence_frames.json` were used.

## Method
- One sequential OpenCV decode (`cv2.VideoCapture` + `read()`, no seeking): 44101 frames decoded.
- Main-camera segments found by comparing 64x36 grey thumbnails with the median of a known main-camera shot;
  every segment (about 31) then viewed as contact sheets (every 12th frame, server crops), serves located coarsely, then
  every frame +-8..11 around the contact inspected at full resolution (near server crop 1-1.6x, far server crop 2-3x).
- `lo` = last frame with a round ball on its toss path (or last frame before the racket hides it), `hi` = first frame with
  the ball a streak leaving the racket. Brackets are 1 frame apart (`hi = lo + 1`) except 4 serves where the ball was
  hidden in the racket blur or leaving the crop (`hi = lo + 2`): lo/hi 4068/4070, 15615/15617, 17477/17479, 29594/29596.
- Radar: board crop sampled at contact-5 and about every 60 frames up to +540; first changed value after contact.
  All 32 serves have a reading (mph); where the board showed a stale number from an earlier unlabelled serve at contact,
  the first change after contact was taken (e.g. serves at 31777 and 33611).
- Server names in `note` (Sabalenka, Pegula) from the pink dress / blue dress.
- Presence: the 80 listed frames were dumped (in order) to a scratch folder; main-camera or not judged by eye
  (22 non-main: close-ups, crowd, low and net-level angles, dissolves, celebration/ceremony);
  feet read from 3-4x crops (far) and 1.5x crops (near) with a crosshair check; points are x,y in 1920x1080 pixels.

## Skipped / uncertain
- No slow-motion replays of serves were labelled. The 16254-18871 stretch (serves 14-16) is a closer-zoom variant of the
  behind-baseline view (both players large, net visible); I treated it as the main camera; presence frames 17304, 17582,
  17615, 18442 follow the same decision (main = true).
- Serves at about 28025 and 28665 (both far, Sabalenka): the board went blank between them; readings used are 101 and 105
  (first new value after each contact).
- Presence frames 4538 and 4668: the near player is at the bottom edge, feet beyond the frame; y set to 1079 (clipped).
- Presence frame 27729: near player's feet hidden behind the score graphic; the point is an estimate under the graphic.
- Presence frame 26304: one of the near player's feet hidden by the score graphic; point is mid-way, estimated.
- Presence frames 12605, 12729, 14431, 24928, 36747: near player is at or close to the net, feet labelled where they are.
- Presence frame 40566: motion blur; feet positions approximate (about +-10 px).
- No commit made.

## Second round

Start about 09:23, end about 09:36 (2026-11-17, local time). One sequential OpenCV decode per script (CPU only); the video decodes 44101 frames per the container, decoding stopped at the last needed frame (40279).

Telling the players apart: Sabalenka = hot pink/magenta dress, dark hair in a bun/ponytail, no visor. Pegula = periwinkle blue-violet dress with a black side panel, white visor, long dark ponytail. They swap ends during the match, so near/far was read per frame. The score graphic agrees with this in every tie: the serving marker sits next to SABALENKA when the pink player serves and next to PEGULA when the blue-violet player serves.

Task A (60 frames): 50 main camera, 10 not main (close or tight cameras: 8720, 8730, 8846, 8855, 8877, 33158, 34540, 34806, 37557, 37573). Feet were placed on crops at 2.2-5x with a pixel grid; precision is about 5-8 px for the far player and 5-10 px for the near player. Estimated rather than seen (near player feet hidden by the score graphic or below the frame edge): frame 5407 (feet just below the bottom edge, y set to 1079), 24690, 24698, 24700, 24707, 26105, 26311, 26325, 27692, 27930, 40274, 40275, 40278 (estimated from the hips/dress hem, +-15 px). Where one foot was lifted or blurred the midpoint of the two feet was used (17786, 18549, 18552, 18648, 17761).

Task B ties (6): 5430 Sabalenka (far server, contact about 5430), 16360 Sabalenka (near, in toss), 18440 Sabalenka (near, contact about 18445), 23080 Pegula (near, in toss, contact about 23085), 24460 Pegula (near, toss starts about 24451), 25800 Sabalenka (far, in serve motion). Frames are inside the serve motion, not exact contact frames.
Flights (40): Sabalenka near 26, Pegula near 12, null 2 (flights 296 and 297: close camera, near player not tellable as main view). Each flight was judged from 4 frames (start, 2 middle, end) at reduced size; colour of the near player's dress was unambiguous in all main-camera flights.
