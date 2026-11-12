"""ITF court geometry in a metric, net-centred world frame.

World frame (metres, right-handed): origin on the ground at the centre of
the net, ``x`` across the court (positive to the camera's right), ``y`` along
the court (positive towards the FAR baseline, away from the camera), ``z`` up.

ITF dimensions are measured to the *outer* edges of the boundary lines, so
the lines themselves lie inside the court. Keypoint detectors see line
*centres*, therefore keypoint world coordinates are placed on line centres
while line calls use the outer edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

HALF_LENGTH = 11.885
DOUBLES_HALF_WIDTH = 5.485
SINGLES_HALF_WIDTH = 4.115
SERVICE_LINE_Y = 6.40
NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07
NET_POST_X = DOUBLES_HALF_WIDTH + 0.914
BALL_RADIUS = 0.0335


@dataclass(frozen=True)
class LineWidths:
    """Painted line widths in metres (ITF: baseline <= 10 cm, others 2.5-5 cm)."""

    baseline: float = 0.10
    sideline: float = 0.05
    service: float = 0.05
    center_service: float = 0.05


@dataclass(frozen=True)
class CourtModel:
    widths: LineWidths = field(default_factory=LineWidths)

    @property
    def baseline_c(self) -> float:
        return HALF_LENGTH - self.widths.baseline / 2

    @property
    def doubles_c(self) -> float:
        return DOUBLES_HALF_WIDTH - self.widths.sideline / 2

    @property
    def singles_c(self) -> float:
        return SINGLES_HALF_WIDTH - self.widths.sideline / 2

    @property
    def service_c(self) -> float:
        return SERVICE_LINE_Y - self.widths.service / 2

    def keypoints(self) -> np.ndarray:
        """14 keypoints in the TennisCourtDetector order, shape (14, 2).

        0/1 far baseline doubles corners (L/R), 2/3 near baseline doubles corners,
        4/5 left singles sideline (far/near), 6/7 right singles sideline (far/near),
        8/9 far service line x singles sidelines, 10/11 near service line x singles,
        12/13 centre service line T (far/near).
        """
        b, d, s, v = self.baseline_c, self.doubles_c, self.singles_c, self.service_c
        return np.array(
            [
                [-d, b], [d, b], [-d, -b], [d, -b],
                [-s, b], [-s, -b], [s, b], [s, -b],
                [-s, v], [s, v], [-s, -v], [s, -v],
                [0.0, v], [0.0, -v],
            ],
            dtype=np.float64,
        )

    def line_segments(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Line-centre segments for drawing: (name, start_xy, end_xy)."""
        b, d, s, v = self.baseline_c, self.doubles_c, self.singles_c, self.service_c
        p = np.array
        return [
            ("far_baseline", p([-d, b]), p([d, b])),
            ("near_baseline", p([-d, -b]), p([d, -b])),
            ("left_doubles", p([-d, -b]), p([-d, b])),
            ("right_doubles", p([d, -b]), p([d, b])),
            ("left_singles", p([-s, -b]), p([-s, b])),
            ("right_singles", p([s, -b]), p([s, b])),
            ("far_service", p([-s, v]), p([s, v])),
            ("near_service", p([-s, -v]), p([s, -v])),
            ("center_service", p([0.0, -v]), p([0.0, v])),
        ]


class Half(StrEnum):
    FAR = "far"
    NEAR = "near"


def half_of(y: float) -> Half:
    return Half.FAR if y > 0 else Half.NEAR


class ZoneKind(StrEnum):
    SINGLES = "singles"
    DOUBLES = "doubles"


@dataclass(frozen=True)
class Zone:
    """Axis-aligned target area bounded by painted lines (outer edges).

    ``open_sides`` lists sides that are *not* lines (the net side); they are
    ignored when measuring the margin to the nearest line.
    """

    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    line_names: dict[str, str]
    open_sides: frozenset[str] = frozenset()

    def _bounds(self, grow: float) -> tuple[float, float, float, float]:
        return (
            -np.inf if "x0" in self.open_sides else self.x0 - grow,
            np.inf if "x1" in self.open_sides else self.x1 + grow,
            -np.inf if "y0" in self.open_sides else self.y0 - grow,
            np.inf if "y1" in self.open_sides else self.y1 + grow,
        )

    def contains(self, xy: np.ndarray, grow: float = 0.0) -> np.ndarray:
        """Vectorised containment for points (..., 2); ``grow`` expands the line sides."""
        x0, x1, y0, y1 = self._bounds(grow)
        x, y = xy[..., 0], xy[..., 1]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    def signed_margin(self, xy: np.ndarray) -> tuple[float, str]:
        """Signed distance (m) from a point to the governing line edge; >0 inside.

        Returns the margin and the name of the governing line.
        """
        x, y = float(xy[0]), float(xy[1])
        x0, x1, y0, y1 = self._bounds(0.0)
        gaps = {"x0": x - x0, "x1": x1 - x, "y0": y - y0, "y1": y1 - y}
        closed = {k: v for k, v in gaps.items() if k not in self.open_sides}
        if all(v >= 0 for v in closed.values()):
            side = min(closed, key=closed.__getitem__)
            return closed[side], self.line_names[side]
        dx = max(-gaps["x0"], 0.0, -gaps["x1"])
        dy = max(-gaps["y0"], 0.0, -gaps["y1"])
        side = min(closed, key=closed.__getitem__)
        return -float(np.hypot(dx, dy)), self.line_names[side]


def rally_zone(target: Half, kind: ZoneKind = ZoneKind.SINGLES) -> Zone:
    """Legal landing area for a rally shot travelling into the ``target`` half."""
    hw = SINGLES_HALF_WIDTH if kind == ZoneKind.SINGLES else DOUBLES_HALF_WIDTH
    side = f"{kind} sideline"
    if target == Half.FAR:
        return Zone(f"far-{kind}", -hw, hw, 0.0, HALF_LENGTH,
                    {"x0": f"left {side}", "x1": f"right {side}", "y0": "net", "y1": "far baseline"},
                    frozenset({"y0"}))
    return Zone(f"near-{kind}", -hw, hw, -HALF_LENGTH, 0.0,
                {"x0": f"left {side}", "x1": f"right {side}", "y0": "near baseline", "y1": "net"},
                frozenset({"y1"}))


def service_box(target: Half, server_x: float, widths: LineWidths | None = None) -> Zone:
    """Diagonal service box for a server standing at ``server_x`` on the other half.

    The centre service line belongs to both boxes, so each box extends to the
    far edge of that line.
    """
    w = widths or LineWidths()
    half_c = w.center_service / 2
    # Serves go cross-court: a server right of centre (x>0) serves into the left box.
    to_left = server_x > 0
    if target == Half.FAR:
        y0, y1 = 0.0, SERVICE_LINE_Y
        names = {"y0": "net", "y1": "far service line"}
        open_sides = frozenset({"y0"})
    else:
        y0, y1 = -SERVICE_LINE_Y, 0.0
        names = {"y0": "near service line", "y1": "net"}
        open_sides = frozenset({"y1"})
    if to_left:
        x0, x1 = -SINGLES_HALF_WIDTH, half_c
        names |= {"x0": "left singles sideline", "x1": "centre service line"}
    else:
        x0, x1 = -half_c, SINGLES_HALF_WIDTH
        names |= {"x0": "centre service line", "x1": "right singles sideline"}
    return Zone(f"service-{target}-{'left' if to_left else 'right'}", x0, x1, y0, y1, names, open_sides)
