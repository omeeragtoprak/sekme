import numpy as np
import torch

from courtvision.models.person_unet import IN_H, IN_W, PersonUNet, decode, person_loss, person_targets
from courtvision.perception.players import IouTracker


def test_targets_decode_back_to_the_boxes():
    boxes = np.array([[100.0, 50.0, 120.0, 110.0], [300.0, 200.0, 330.0, 280.0]])
    centre, size, mask = person_targets(boxes)
    logits = np.where(centre > 0, 6.0, -6.0)
    out = np.stack([logits, size[0], size[1]])
    dets = sorted(decode(out), key=lambda d: d[0][0])
    assert len(dets) == 2
    for (box, _), want in zip(dets, boxes, strict=True):
        np.testing.assert_allclose(box, want, atol=0.6)


def test_network_shape_and_loss():
    net = PersonUNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, IN_H, IN_W))
    assert out.shape == (1, 3, IN_H, IN_W)
    centre, size, mask = (torch.from_numpy(a)[None] for a in person_targets(np.array([[10.0, 10.0, 30.0, 60.0]])))
    assert torch.isfinite(person_loss(out, centre, size, mask))


def test_iou_tracker_keeps_ids_across_a_short_gap_and_starts_new_ones():
    t = IouTracker(min_iou=0.3, max_age=2)
    a, b = np.array([0.0, 0.0, 10.0, 20.0]), np.array([100.0, 0.0, 110.0, 20.0])
    first = t.update([a, b])
    assert t.update([a + 1, b + 1]) == first
    assert t.update([]) == []
    assert t.update([a + 2]) == [first[0]]  # one frame unseen: same player
    assert t.update([np.array([500.0, 0, 510, 20])]) != [first[0]]
