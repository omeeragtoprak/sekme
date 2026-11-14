"""Score bounce detection and bounce position against the TrackNet tennis labels.

The TrackNet tennis set (Huang et al., 2019) labels the ball pixel in every frame of broadcast clips
(1280x720, 30 fps) and marks the frame of each bounce (status 2). That is an independent, human-made
reference for *when* and *where in the image* the ball touched the court.

The labelled bounce frame is only the frame nearest to the contact; at 30 fps the ball moves several
decimetres per frame. The reference contact is therefore taken where the labelled incoming and
outgoing image tracks meet (lines through the two labels on each side), between those frames.

Both the reference and the system's bounce are put on the ground with the *same* homography, so this
measures detection, tracking, the 3D fit and the bounce timing, not the court calibration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from courtvision.eval.labels import world_from_px
from courtvision.geometry.court import BALL_RADIUS


@dataclass
class Labels:
    visible: np.ndarray  # (N,) bool
    xy: np.ndarray  # (N, 2) pixels, NaN when not visible
    status: np.ndarray  # (N,) 0 flying, 1 hit, 2 bounce


@dataclass
class Contact:
    frame: float  # sub-frame time of the contact, in frames
    px: np.ndarray  # (2,) image point of the contact
    from_tracks: bool  # False: fell back to the labelled bounce frame itself


# TrackNet ball labels sit ~1.5 px right of and below the ball's centre in the image (1280x720 pixels),
# measured without any network by scripts/label_offset.py on the selection clips of games 1-7. The ball
# network v3 was trained against labels minus this offset.
LABEL_OFFSET_PX = (1.5, 1.5)

# The offset is not one convention. Per game it is a constant of 0.6 to 1.7 px plus a term proportional to the
# ball's speed (the labels lead the image by up to ~0.16 frame); scripts/label_offset.py --per-game, every fifth
# clip of games 1-7, all clips of games 8-10. These are the constants: only they move a bounce reference (a speed
# term shifts each point along its own track). Centred references use their own game's constant.
GAME_LABEL_OFFSET_PX = {
    "game1": (1.54, 1.39), "game2": (1.41, 0.82), "game3": (1.71, 1.20), "game4": (1.40, 1.56),
    "game5": (1.60, 1.60), "game6": (1.09, 1.54), "game7": (1.21, 1.44),
    "game8": (1.14, 0.84), "game9": (1.33, 1.18), "game10": (0.62, 0.65),
}


def load_labels(path: Path, offset: tuple[float, float] = (0.0, 0.0)) -> Labels:
    """Labels of one clip; ``offset`` (px) is subtracted from every position (GAME_LABEL_OFFSET_PX: image centre)."""
    rows = list(csv.DictReader(path.open(newline="")))
    n = len(rows)
    vis = np.zeros(n, bool)
    xy = np.full((n, 2), np.nan)
    st = np.zeros(n, int)
    for i, r in enumerate(rows):
        v = (r.get("visibility") or "0").strip()
        vis[i] = v not in ("", "0")
        if vis[i]:
            xy[i] = [float(r["x-coordinate"]) - offset[0], float(r["y-coordinate"]) - offset[1]]
        s = (r.get("status") or "").strip()
        st[i] = int(float(s)) if s else 0
    return Labels(vis, xy, st)


def contacts(lab: Labels, reach: int = 4) -> list[Contact]:
    """Reference contacts: meeting point of the labelled tracks just before and just after each bounce."""
    out = []
    n = len(lab.status)
    events = np.flatnonzero(lab.status > 0)  # hits (1) and bounces (2): a track line never crosses one
    for i in np.flatnonzero(lab.status == 2):
        lo = max([max(0, i - reach), *(e + 1 for e in events if e < i)])
        hi = min([min(n, i + 1 + reach), *(e for e in events if e > i)])
        pre = [j for j in range(lo, i) if lab.visible[j]][-2:]
        post = [j for j in range(i + 1, hi) if lab.visible[j]][:2]
        if len(pre) < 2 or len(post) < 2:
            if lab.visible[i]:
                out.append(Contact(float(i), lab.xy[i].copy(), False))
            continue
        a1, b1 = lab.xy[pre[1]], (lab.xy[pre[1]] - lab.xy[pre[0]]) / (pre[1] - pre[0])
        a2, b2 = lab.xy[post[0]], (lab.xy[post[1]] - lab.xy[post[0]]) / (post[1] - post[0])
        da = (a1 - b1 * pre[1]) - (a2 - b2 * post[0])
        db = b1 - b2
        f = -float(da @ db) / max(float(db @ db), 1e-9)
        f = min(max(f, float(pre[1])), float(post[0]))
        p = 0.5 * ((a1 + b1 * (f - pre[1])) + (a2 + b2 * (f - post[0])))
        out.append(Contact(f, p, True))
    return out


@dataclass
class Match:
    ref_frame: float
    sys_frame: float | None
    ref_xy: np.ndarray | None  # ground (m), None when the frame has no calibration
    sys_xy: np.ndarray | None
    sys_verdict: str | None = None
    ref_verdict: str | None = None

    @property
    def error_m(self) -> float | None:
        if self.ref_xy is None or self.sys_xy is None:
            return None
        return float(np.linalg.norm(self.ref_xy - self.sys_xy))


def ball_centre_ground(g: np.ndarray, camera: np.ndarray | None) -> np.ndarray:
    """Ground point under a ball whose *centre* projects to ground point ``g``.

    Labels mark the ball centre, 3.35 cm above the court at contact; put on the ground plane it lands
    behind the contact as seen from the camera (about 10 cm for a broadcast camera 18 deg above the court).
    """
    if camera is None:
        return g
    C = np.asarray(camera, float)
    return g + (C[:2] - g) * BALL_RADIUS / C[2]


def match(refs: list[Contact], bounces: list[tuple[float, np.ndarray, str | None]], H: list,
          tol: float = 3.0, camera: np.ndarray | None = None) -> tuple[list[Match], int]:
    """Pair reference contacts with system bounces within ``tol`` frames, one to one, closest pairs first.

    ``bounces`` holds (frame, ground xy, verdict) of the system's measured bounces; ``H`` the per-frame
    world->image homographies (None where uncalibrated). Returns the matches (unmatched references keep
    ``sys_frame`` None) and the number of system bounces with no reference (false alarms).
    """
    refs = sorted(refs, key=lambda c: c.frame)
    pairs = sorted((abs(f - c.frame), i, j) for i, c in enumerate(refs) for j, (f, _, _) in enumerate(bounces)
                   if abs(f - c.frame) <= tol)
    taken: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in taken and j not in used:
            taken[i] = j
            used.add(j)
    out = []
    for i, c in enumerate(refs):
        k = int(round(c.frame))
        h = H[k] if 0 <= k < len(H) else None
        ref_xy = ball_centre_ground(np.asarray(world_from_px(h, tuple(c.px))), camera) if h is not None else None
        if i not in taken:
            out.append(Match(c.frame, None, ref_xy, None))
            continue
        f, xy, verdict = bounces[taken[i]]
        out.append(Match(c.frame, f, ref_xy, np.asarray(xy, float), verdict))
    return out, len(bounces) - len(used)
