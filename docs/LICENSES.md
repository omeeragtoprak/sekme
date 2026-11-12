# Third-party models, code and data

| Component | Where | Source | Licence | Use in CourtVision |
|---|---|---|---|---|
| WASB ball detector (HRNet) — code | `src/courtvision/models/wasb_hrnet.py` (vendored, header kept) | github.com/nttcom/WASB-SBDT | MIT | production-eligible |
| WASB tennis weights | `models/wasb_tennis_best.pth.tar` | WASB-SBDT model zoo (Google Drive) | MIT (repository licence) | production-eligible |
| Court keypoint weights | `models/court_kps_yastrebksv.pt` | github.com/yastrebksv/TennisCourtDetector | **none published** | **research only** — retrain on own data before any commercial use |
| Court keypoint network code | `src/courtvision/models/court_net.py` | independent implementation, layer names match the checkpoint | ours | — |
| Player detector + pose | `models/yolo26s.pt`, `models/yolo26s-pose.pt` via `ultralytics` | Ultralytics YOLO26 / YOLO26-pose | **AGPL-3.0** | fine for research/open source; a closed networked product needs an Ultralytics enterprise licence or an Apache-2.0 detector (e.g. RF-DETR) behind `perception/players.py` |
| Ultralytics ByteTrack | `ultralytics` | Ultralytics | AGPL-3.0 | same as above |
| Sample clip (Vienna, Djokovic–Sonego) | `data/videos/` (git-ignored) | ATP Tennis TV broadcast, via the public abdullahtarek/tennis_analysis repo | copyright of the broadcaster | local research and evaluation only, never redistributed |
| US Open 2023 final highlights (first 90 s) | `data/videos/` (git-ignored) | official US Open YouTube channel | copyright of USTA / broadcaster | local research and evaluation only, never redistributed |
| Davis Cup 2018 clips (optional) | Wikimedia Commons | Uruguay vs Mexico | CC BY-SA 4.0 | free to use with attribution |

Product naming: avoid "Hawk-Eye" (trademark) and FIFA "VAR" in product names or marketing;
"CourtVision" may collide with an existing sports product name — check before any public release.
