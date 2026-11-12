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
