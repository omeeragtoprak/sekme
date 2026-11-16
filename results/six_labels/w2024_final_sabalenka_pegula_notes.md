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
