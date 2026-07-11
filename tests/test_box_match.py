"""§9.4: unit-test the IoU matcher in isolation before wiring into the loss."""
import numpy as np

from endosemi.matching import iou_match
from endosemi.utils.boxes import box_iou_matrix


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    iou = box_iou_matrix(a, b)
    assert iou.shape == (1, 2)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 0.0)


def test_half_overlap():
    a = np.array([[0, 0, 10, 10]])          # area 100
    b = np.array([[5, 0, 15, 10]])          # area 100, inter 50, union 150
    assert np.isclose(box_iou_matrix(a, b)[0, 0], 50 / 150)


def test_perfect_match():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    m = iou_match(boxes, boxes.copy(), iou_thresh=0.5)
    assert sorted(m.matches) == [(0, 0), (1, 1)]
    assert m.unmatched1 == [] and m.unmatched2 == []


def test_unmatched_reported():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
    m = iou_match(b1, b2, iou_thresh=0.5)
    assert m.matches == [(0, 0)]
    assert m.unmatched1 == []
    assert m.unmatched2 == [1]


def test_below_threshold_not_matched():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[7, 0, 17, 10]])   # IoU = 3/17 ~ 0.18 < 0.5
    m = iou_match(b1, b2, iou_thresh=0.5)
    assert m.matches == []
    assert m.unmatched1 == [0] and m.unmatched2 == [0]


def test_empty_inputs():
    empty = np.zeros((0, 4))
    m = iou_match(empty, np.array([[0, 0, 10, 10]]), iou_thresh=0.5)
    assert m.matches == [] and m.unmatched1 == [] and m.unmatched2 == [0]


def test_greedy_and_hungarian_agree_on_clean_case():
    b1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b2 = np.array([[21, 21, 31, 31], [0, 0, 10, 10]])
    g = iou_match(b1, b2, 0.5, algorithm="greedy")
    h = iou_match(b1, b2, 0.5, algorithm="hungarian")
    assert sorted(g.matches) == sorted(h.matches) == [(0, 1), (1, 0)]
