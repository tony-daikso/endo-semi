"""Classification/objectness epistemic uncertainty via MC-Dropout (§3.2).

Given K stochastic forward passes' per-box class-probability samples
(shape (n_boxes, K, nc), produced by models.mc_dropout.mc_dropout_forward):

    P_i_cls = mean_k(cls_probs)                 # mean predictive distribution
    U_i_cls = mean_k(entropy(cls_probs_k))      # expected entropy across passes
"""
from __future__ import annotations

import numpy as np

from .dfl_entropy import _entropy


def cls_mc_uncertainty(cls_prob_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cls_prob_samples: (n_boxes, K, nc) softmax over classes for each MC pass.
    returns: (P_i_cls (n_boxes, nc), U_i_cls (n_boxes,))
    """
    x = np.asarray(cls_prob_samples, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"expected (n_boxes, K, nc), got {x.shape}")
    P_i_cls = x.mean(axis=1)                       # (n, nc)
    U_i_cls = _entropy(x, axis=-1).mean(axis=1)    # entropy per pass, mean over K -> (n,)
    return P_i_cls, U_i_cls
