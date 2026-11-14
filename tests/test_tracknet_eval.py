import numpy as np

from courtvision.eval.tracknet import Contact, Labels, contacts, match


def _labels(xy, status):
    xy = np.asarray(xy, float)
    return Labels(np.isfinite(xy[:, 0]), xy, np.asarray(status))


def test_contact_is_where_incoming_and_outgoing_tracks_meet():
    # Ball falls 10 px/frame, bounces at frame 3.5 at (35, 135), then rises 6 px/frame.
    xs = 10.0 * np.arange(8)
    ys = np.array([100 + 10 * f if f < 3.5 else 135 - 6 * (f - 3.5) for f in range(8)])
    c = contacts(_labels(np.c_[xs, ys], [0, 0, 0, 0, 2, 0, 0, 0]))
    assert len(c) == 1 and c[0].from_tracks
    assert abs(c[0].frame - 3.5) < 1e-6
    assert np.allclose(c[0].px, [35.0, 135.0], atol=1e-6)


def test_contact_falls_back_to_the_labelled_frame_without_tracks():
    xy = np.full((5, 2), np.nan)
    xy[2] = [50, 60]
    c = contacts(_labels(xy, [0, 0, 2, 0, 0]))
    assert len(c) == 1 and not c[0].from_tracks and c[0].frame == 2.0


def test_match_pairs_nearest_bounce_and_counts_false_alarms():
    H = [np.eye(3)] * 50
    refs = [Contact(10.2, np.array([1.0, 2.0]), True), Contact(30.0, np.array([5.0, 5.0]), True)]
    bounces = [(10.0, np.array([1.1, 2.0]), "IN"), (20.0, np.array([0.0, 0.0]), "IN")]
    m, false_alarms = match(refs, bounces, H)
    assert m[0].sys_frame == 10.0 and abs(m[0].error_m - 0.1) < 1e-9
    assert m[1].sys_frame is None
    assert false_alarms == 1


def test_ball_centre_label_is_moved_towards_the_camera():
    from courtvision.eval.tracknet import ball_centre_ground

    # Camera 10 m up at y = -30: a centre seen at y = 10 lies 0.0335/10 of the way back towards it.
    g = ball_centre_ground(np.array([0.0, 10.0]), np.array([0.0, -30.0, 10.0]))
    assert np.allclose(g, [0.0, 10.0 - 40.0 * 0.0335 / 10.0])


def _scripts_on_path() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_label_offset_fit_separates_the_constant_from_the_speed_term():
    # TrackNet labels: a constant offset plus a lead proportional to the ball's speed. Only the constant moves
    # a bounce reference, so the fit must recover it whatever the mix of up- and down-moving frames.
    _scripts_on_path()
    from label_offset import constant_and_slope

    rng = np.random.default_rng(0)
    v = np.concatenate([rng.uniform(5, 25, 300), -rng.uniform(5, 25, 100)])  # unbalanced directions
    offset = -1.2 - 0.13 * v + rng.normal(0, 0.5, len(v))
    c, se, k = constant_and_slope(offset, v)
    assert abs(c + 1.2) < 3 * se and se < 0.1
    assert abs(k + 0.13) < 0.01
    assert abs(offset.mean() + 1.2) > 0.5  # the plain mean would be badly biased here


def test_game_offsets_are_subtracted_from_the_labels():
    import tempfile
    from pathlib import Path

    from courtvision.eval.tracknet import GAME_LABEL_OFFSET_PX, load_labels

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Label.csv"
        p.write_text("file name,visibility,x-coordinate,y-coordinate,status\n0000.jpg,1,100,200,0\n0001.jpg,0,,,0\n")
        lab = load_labels(p, (1.5, 0.5))
    assert np.allclose(lab.xy[0], [98.5, 199.5]) and not lab.visible[1]
    assert all(0.5 <= dx <= 2.0 and 0.5 <= dy <= 2.0 for dx, dy in GAME_LABEL_OFFSET_PX.values())


def test_matching_takes_the_closest_pairs_first_not_the_references_in_order() -> None:
    import numpy as np

    from courtvision.eval.tracknet import Contact, match

    refs = [Contact(10.0, np.array([0.0, 0.0]), True), Contact(12.0, np.array([0.0, 0.0]), True)]
    bounces = [(11.5, np.array([0.0, 0.0]), "IN"), (8.0, np.array([0.0, 0.0]), "IN")]
    m, false_alarms = match(refs, bounces, [None] * 20)
    # In reference order, 10 would take 11.5 and leave 12 unmatched (8 is 4 frames from it).
    assert [x.sys_frame for x in m] == [8.0, 11.5] and false_alarms == 0


def test_a_contact_s_track_lines_stop_at_the_neighbouring_label() -> None:
    import numpy as np

    from courtvision.eval.tracknet import Labels, contacts

    n = 40
    xy = np.array([[10.0 * i, 100.0] for i in range(n)])
    st = np.zeros(n, int)
    st[20], st[21] = 2, 2  # one bounce labelled on two adjacent frames
    c = contacts(Labels(np.ones(n, bool), xy, st))
    assert [round(x.frame) for x in c] == [20, 21]  # the second does not borrow the first's track line
