"""Box-coordinate uncertainty from the DFL distribution (§3.2).

YOLO's DFL box head predicts each box edge (left/top/right/bottom) as a discrete
probability distribution over bins rather than a single number. The entropy of
that distribution is a free per-edge uncertainty — no MC-Dropout passes needed
for box coordinates.

    box_uncertainty = mean over the 4 edges of entropy(edge_bin_distribution)
"""
from __future__ import annotations

import numpy as np


def _entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy along `axis`. Same formula used in the segmentation version."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def dfl_edge_entropy(dfl_probs: np.ndarray) -> np.ndarray:
    """Per-edge entropy.

    dfl_probs: (..., n_bins) softmax distribution for one edge (or batched).
    returns:   (...) entropy scalar per edge.
    """
    return _entropy(dfl_probs, axis=-1)


def box_dfl_uncertainty(box_dfl_probs: np.ndarray) -> np.ndarray:
    """Mean edge entropy per box.

    box_dfl_probs: (n_boxes, 4, n_bins) — 4 edges, each a bin distribution.
    returns:       (n_boxes,) mean over the 4 edges.
    """
    box_dfl_probs = np.asarray(box_dfl_probs, dtype=float)
    if box_dfl_probs.ndim != 3 or box_dfl_probs.shape[1] != 4:
        raise ValueError(f"expected (n,4,n_bins), got {box_dfl_probs.shape}")
    per_edge = dfl_edge_entropy(box_dfl_probs)      # (n, 4)
    return per_edge.mean(axis=1)                    # (n,)
