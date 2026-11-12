"""Turkish wording for everything a person reads (report notes, burned-in overlays).

Identifiers in the data (verdict codes, ``line`` names such as ``"far baseline"``) stay English so
that reports, the review log and tests remain stable; only their display text is translated here.
Python's ``str.upper`` is not Turkish-aware (``"i".upper() == "I"``), so upper-case forms are spelled out.
"""

from __future__ import annotations

VERDICT = {"IN": "içeride", "OUT": "dışarıda", "REVIEW": "belirsiz"}
VERDICT_UPPER = {"IN": "İÇERİDE", "OUT": "DIŞARIDA", "REVIEW": "İNCELEME"}

_LINES = {
    "far baseline": "uzak dip çizgi",
    "near baseline": "yakın dip çizgi",
    "far service line": "uzak servis çizgisi",
    "near service line": "yakın servis çizgisi",
    "centre service line": "orta servis çizgisi",
    "net": "file",
}
_SIDE = {"left": "sol", "right": "sağ"}
_KIND = {"singles": "tekler", "doubles": "çiftler"}


def num(v: float, digits: int = 1) -> str:
    """Number with a Turkish decimal comma, e.g. ``num(12.34) == "12,3"``."""
    return f"{v:.{digits}f}".replace(".", ",")


def prob(p: float, digits: int = 1) -> str:
    """Probability as a Turkish percentage that never claims certainty (no "%100" or "%0")."""
    lo = 0.5 * 10 ** -(digits + 2)
    if p >= 1 - lo:
        return f">%{num(100 - 10 ** -digits, digits)}" if digits else ">%99"
    if p <= lo:
        return f"<%{num(10 ** -digits, digits)}" if digits else "<%1"
    return f"%{num(100 * p, digits)}"


def line_name(line: str) -> str:
    """Turkish name of a court line identifier; unknown names are returned unchanged."""
    if line in _LINES:
        return _LINES[line]
    parts = line.split()
    if len(parts) == 3 and parts[0] in _SIDE and parts[1] in _KIND and parts[2] == "sideline":
        return f"{_SIDE[parts[0]]} {_KIND[parts[1]]} yan çizgisi"
    return line
