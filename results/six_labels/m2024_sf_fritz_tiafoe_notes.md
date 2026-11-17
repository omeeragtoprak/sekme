# m2024_sf_fritz_tiafoe: blind labels (serves and player presence)

Work started 2026-11-16 22:34, finished 2026-11-16 23:08. Blind labeller; only the video, the template, the helper
scripts and the presence frame list were opened. Frames were decoded in order with OpenCV (43780 frames, 59.94 fps).

## Method (serves)
1. One sequential decode, every 15th frame at 320x180 on contact sheets, to see the camera layout. In this video the
   "main camera" is a high wide view from behind the near baseline (whole court, stadium around it).
2. Segmented the wide view from the rest (close-ups, tight/zoomed shots, replays) with a colour and layout test, then
   browsed each wide segment with strips of the far and near baseline areas (every 12-30 frames), to find each serve.
3. Fine pass: every frame +-12..14 around each serve (far server 1.6x, near server 1.0x crops), contact bracket read there.
   Near servers: ball is visible, bracket is last round ball (lo) and first streak (hi), 2-3 frames wide.
   Far servers: ball is often hidden in the racket blur or against the wall logo; bracket is 3 frames wide around the
   frame with arm and racket at full extension (ball seen leaving as a streak where visible).
4. Radar: the board (top left of the back wall) changes about 170-240 frames (3-4 s) after contact; the first new value
   after contact was read on board crops at contact+90 ... +330 (step 30). Board unit is mph. Two cases null (below).
   First-serve faults get their own board value; both serves of a point are labelled separately.

## Result
32 serves: 15 near, 17 far; 30 with a radar reading, 2 null. Servers (note field): Fritz and Tiafoe alternate by game.

## Skipped serves
Serves shown only in tight or zoomed shots (camera framed on the server, far baseline not in the picture, e.g. Fritz
around frame 6267-6303) were not labelled: not the wide view. A few such shots exist in frames about 5160-7555,
20005-21435 and elsewhere; they were not inspected one by one.

## Uncertain
- 19659-19661 (near, Fritz second serve after the fault at 19519-19521): back view, ball above the crop; bracket from the
  racket apex. Radar null: board still showed the previous value when the clip was cut (about frame 19900).
- 40886-40888 (far, Fritz): radar null, clip cut at 41005 before the board changed.
- Far brackets 221-223, 1974-1976, 3235-3237, 7950-7952, 9687-9689, 10881-10883, 21786-21788, 25456-25458, 26766-26768,
  35410-35412: ball not resolvable, apex based, +-1 frame.
- 7950-7952: long toss (ball up at about 7897), single serve. 31278-31280 and 19519-19521 were faults (no return), the
  second serves are 31763-31765 and 19659-19661.
- Radar digits with the hundreds digit cut off by the camera pan (frames 11982, 12694, 13388, 15196, 36210): value read as
  108, 116, 97, 78, 118 (previous board value gave the missing digit).

## Presence (task 2)
80 frames listed; 52 labelled main (wide high view), 28 not main (close-ups, low angle, net-level cameras, replays, titles)
and 6 tight/zoomed shots of the same camera (5417, 5641, 6278, 6830, 6849, 7615: far baseline not in the picture), which
I labelled main=false; the coordinator may want to check these six. Feet from the full-resolution frame, refined on 3-4x
crops (+-3 px for most).
Hand-placed or doubtful feet: 10872 and 32274 (near player's feet hidden by the score graphic, estimated), 32550 (near
legs cut by the bottom edge, y=1079), 3699 and 3763 (Fritz sitting on court; point under the body/feet), 14565, 22247,
24310, 29082, 34049 (far player with one foot in the air: ground foot or mid-point), 38737 (Fritz sliding, feet apart),
27229 (far player Tiafoe far left, near the chair), 24915 (near foot partly behind the graphic).

## Second round
Start 2026-11-17 07:05 (+03), end 07:20. Opened only the allowed files; no src/, runs/ or system output seen.
- Decode: OpenCV in order, CPU only, one process at a time (the video decodes to 39354 frames). Frames saved as crops/sheets in a scratch folder.
- Task A (60 frames): 57 main, 3 not main (7619, 7620: camera zoomed tight on the near player, far baseline not in the picture; 7752: dissolve between two shots). The high wide view from behind the near baseline (frames 3788, 7880, 7901, 9605-9679, 14036, 21644, 22567/22568, 25231, 28500) counted as main, as in round 1. Feet: first from a gridded half-size view, then refined on crops (6x far, 3x near) with a coordinate grid; point between the feet on the ground (planted foot when one is clearly in the air). Uncertain: near feet hidden behind the score graphic in 6584, 6661, 7901, 9605-9679, 21644 (estimated from the visible shoe or shoe soles below the graphic, +-15 px), blurred running feet in 7386, 7552, 20454, 20509.
- Task B, telling them apart: Fritz = black shirt, white/grey shorts, white headband, pink racket; Tiafoe = lavender tank top with magenta shorts, white headband. Ends swap at changeovers, so the name is read from clothing in every flight.
- Ties (serving marker next to the name, server seen serving on the main camera): 6298 Fritz (near), 20270 Tiafoe (near), 21786 Fritz (far), 28626 Tiafoe (near). Serves under the "forehand winners" / "net points won" panels were skipped (marker hidden).
- Flights: 26 Fritz, 14 Tiafoe, 0 null. Flight 169 (17728-17749): near player is a black-shirt player standing wide of the court, taken as Fritz.
