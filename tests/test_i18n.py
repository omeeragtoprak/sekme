"""Turkish display text: every line a call can name is translated, and probabilities never claim certainty."""

from courtvision.geometry.court import Half, ZoneKind, rally_zone, service_box
from courtvision.i18n import line_name, prob


def test_every_line_a_call_can_name_has_a_turkish_name() -> None:
    zones = [rally_zone(h, k) for h in Half for k in ZoneKind]
    zones += [service_box(h, x) for h in Half for x in (-2.0, 2.0)]
    names = {n for z in zones for n in z.line_names.values()}
    assert names >= {"far baseline", "near baseline", "centre service line", "left singles sideline"}
    for n in names:
        assert line_name(n) != n, n
    assert line_name("right doubles sideline") == "sağ çiftler yan çizgisi"


def test_probability_text_never_says_certain() -> None:
    assert prob(1.0) == ">%99,9" and prob(0.0) == "<%0,1"
    assert prob(0.99999, 0) == ">%99" and prob(1e-5, 0) == "<%1"
    assert prob(0.131) == "%13,1"
