"""§4 joint pseudo-label fusion."""
import numpy as np

from endosemi.losses.joint_pseudo import joint_pseudo_labels


def test_matched_pair_picks_lower_uncertainty():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[1, 1, 11, 11]])          # high IoU with b1
    u1 = np.array([0.9])                     # f1 uncertain
    u2 = np.array([0.1])                     # f2 confident -> should win
    boxes, u, prov = joint_pseudo_labels(b1, u1, b2, u2, iou_thresh=0.5)
    assert prov == ["f2"]
    np.testing.assert_allclose(boxes[0], b2[0])
    assert np.isclose(u[0], 0.1)


def test_unmatched_admitted_only_under_strict_threshold():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[100, 100, 110, 110]])    # no overlap -> both unmatched
    u1 = np.array([0.2])                     # below strict -> admitted
    u2 = np.array([0.9])                     # above strict -> dropped
    boxes, u, prov = joint_pseudo_labels(
        b1, u1, b2, u2, iou_thresh=0.5, strict_threshold=0.5)
    assert prov == ["f1_solo"]
    np.testing.assert_allclose(boxes[0], b1[0])


def test_unmatched_dropped_when_no_strict_threshold():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[100, 100, 110, 110]])
    boxes, u, prov = joint_pseudo_labels(
        b1, np.array([0.1]), b2, np.array([0.1]),
        iou_thresh=0.5, strict_threshold=None)
    assert len(boxes) == 0 and prov == []


def test_empty_inputs():
    boxes, u, prov = joint_pseudo_labels(
        np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros(0))
    assert len(boxes) == 0
