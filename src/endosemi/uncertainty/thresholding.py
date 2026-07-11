"""Dynamic uncertainty threshold for pseudo-label admission (§3.3).

Base rule (per the segmentation paper):

    T = min(mu + sigma, P95)

applied to a population of box uncertainties, keeping boxes with U < T.

Detection caveat (§3.3): a sparse image (1-3 polyps) gives a near-meaningless
mu/sigma/P95 over 1-3 values. So we compute the threshold over a **mini-batch**
or a **running window** of recent box uncertainties, then apply that shared T to
each image's boxes. This is a deliberate adaptation, not in the original paper —
flag it as an experimental design choice when publishing.

Modes:
  'batch'  : T from all boxes in the current mini-batch.
  'window' : T from a running window of the last `window_size` box uncertainties.
  'image'  : original per-image rule. NOT recommended for sparse detection.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def base_threshold(u: np.ndarray) -> float:
    """T = min(mu + sigma, P95). Empty -> +inf (admit nothing meaningfully)."""
    u = np.asarray(u, dtype=float).ravel()
    if u.size == 0:
        return float("inf")
    mu, sigma = u.mean(), u.std()
    p95 = np.percentile(u, 95)
    return float(min(mu + sigma, p95))


class DynamicThreshold:
    """Stateful threshold provider covering all three §3.3 modes."""

    def __init__(self, mode: str = "batch", window_size: int = 512):
        if mode not in {"batch", "window", "image"}:
            raise ValueError(f"unknown threshold mode: {mode}")
        self.mode = mode
        self._window: deque[float] = deque(maxlen=window_size)

    def update(self, batch_uncertainties: np.ndarray) -> None:
        """Feed the current batch's box uncertainties into the running window."""
        if self.mode == "window":
            self._window.extend(np.asarray(batch_uncertainties, dtype=float).ravel().tolist())

    def compute(self, image_uncertainties: np.ndarray,
                batch_uncertainties: np.ndarray | None = None) -> float:
        """Return T to apply to THIS image's boxes.

        - image  : from this image's own boxes (sparse-unstable; see caveat).
        - batch  : from the whole mini-batch (pass `batch_uncertainties`).
        - window : from the accumulated running window (call `update` first).
        """
        if self.mode == "image":
            return base_threshold(image_uncertainties)
        if self.mode == "batch":
            if batch_uncertainties is None:
                raise ValueError("mode='batch' needs batch_uncertainties")
            return base_threshold(batch_uncertainties)
        # window
        if not self._window:
            return float("inf")
        return base_threshold(np.fromiter(self._window, dtype=float))

    @staticmethod
    def keep_mask(image_uncertainties: np.ndarray, T: float) -> np.ndarray:
        """Boolean keep mask: U < T (§3.3)."""
        return np.asarray(image_uncertainties, dtype=float) < T
