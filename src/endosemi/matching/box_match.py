"""IoU-based box matching between the two networks' detections (§4).

The spec calls this "the most bug-prone part" and says to unit-test it in
isolation before wiring it into the loss (§9.4) — see tests/test_box_match.py.

Given two box sets, return:
  - matches   : list of (i, j) index pairs with IoU >= iou_thresh
  - unmatched1: indices in set 1 with no partner
  - unmatched2: indices in set 2 with no partner

Two algorithms:
  - 'hungarian': global optimum via scipy linear_sum_assignment on -IoU, then
    drop pairs below threshold. Best when boxes are dense/ambiguous.
  - 'greedy'   : repeatedly take the highest-IoU available pair. Cheaper; fine
    for the typical sparse polyp case (1-3 boxes/frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.boxes import box_iou_matrix


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched1: list[int]
    unmatched2: list[int]
    iou: np.ndarray            # the full IoU matrix, for downstream inspection


def _greedy(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_i, used_j = set(), set()
    # iterate candidate pairs by descending IoU
    order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
    for i, j in order:
        i, j = int(i), int(j)
        if iou[i, j] < thr:
            break
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)
    return pairs


def _hungarian(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment

    # maximize total IoU == minimize -IoU
    row, col = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(row, col) if iou[i, j] >= thr]


def iou_match(
    boxes1_xyxy: np.ndarray,
    boxes2_xyxy: np.ndarray,
    iou_thresh: float = 0.5,
    algorithm: str = "hungarian",
) -> MatchResult:
    b1 = np.asarray(boxes1_xyxy, dtype=float).reshape(-1, 4)
    b2 = np.asarray(boxes2_xyxy, dtype=float).reshape(-1, 4)
    iou = box_iou_matrix(b1, b2)

    if len(b1) == 0 or len(b2) == 0:
        return MatchResult([], list(range(len(b1))), list(range(len(b2))), iou)

    if algorithm == "greedy":
        matches = _greedy(iou, iou_thresh)
    elif algorithm == "hungarian":
        matches = _hungarian(iou, iou_thresh)
    else:
        raise ValueError(f"unknown matching algorithm: {algorithm}")

    matched_i = {i for i, _ in matches}
    matched_j = {j for _, j in matches}
    unmatched1 = [i for i in range(len(b1)) if i not in matched_i]
    unmatched2 = [j for j in range(len(b2)) if j not in matched_j]
    return MatchResult(matches, unmatched1, unmatched2, iou)
