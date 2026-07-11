"""Combined per-box uncertainty (§3.2).

    U_box = alpha * U_cls + (1 - alpha) * U_dfl

alpha=0.5 is the spec's starting point; tune on validation. U_cls (MC-Dropout
classification entropy) and U_dfl (DFL edge entropy) live on different scales,
so an optional running normalization is provided to keep the mix meaningful.
"""
from __future__ import annotations

import numpy as np


def combine_box_uncertainty(
    u_cls: np.ndarray,
    u_dfl: np.ndarray,
    alpha: float = 0.5,
    normalize: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    u_cls = np.asarray(u_cls, dtype=float)
    u_dfl = np.asarray(u_dfl, dtype=float)
    if u_cls.shape != u_dfl.shape:
        raise ValueError(f"shape mismatch: U_cls {u_cls.shape} vs U_dfl {u_dfl.shape}")
    if normalize:
        u_cls = u_cls / (u_cls.max() + eps)
        u_dfl = u_dfl / (u_dfl.max() + eps)
    return alpha * u_cls + (1.0 - alpha) * u_dfl
