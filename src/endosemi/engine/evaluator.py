"""Evaluation, stratified by lesion size (§8).

Endo-SemiS Table 4 found SSL gains were largest for large lesions and that
small lesions stayed hardest. So we report metrics both overall AND split into
size bins (by GT box area in px²), matching the project's existing
size-stratified analysis.

At inference, evaluate whichever single net is chosen for deployment (or an
ensemble of f1+f2). For video, optionally apply §7 temporal correction first.
"""
from __future__ import annotations

import numpy as np

from ..utils.boxes import box_area, xywh_to_xyxy
from ..utils.logging import get_logger

log = get_logger(__name__)

SIZE_NAMES = ["small", "medium", "large"]


def size_bin(area_px2: float, bins: list[float]) -> str:
    """bins e.g. [1024, 4096, 16384] -> small/medium/large. `bins` are the upper
    cutoffs for small and medium; anything larger is 'large'."""
    if area_px2 < bins[0]:
        return "small"
    if area_px2 < bins[1]:
        return "medium"
    return "large"


class SizeStratifiedEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bins = cfg["eval"]["size_bins_px2"]
        self.metrics = cfg["eval"]["metrics"]

    def evaluate(self, model, val_list: str, imgsz: int) -> dict:
        """Return {'overall': {...}, 'small': {...}, 'medium': {...}, 'large': {...}}.

        Implementation plan:
          - run model over val_list; collect (pred boxes, GT boxes) per image.
          - assign each GT to a size bin by its px² area (bin cutoffs from cfg).
          - compute mAP@50, mAP@50-95, precision, recall overall and per bin by
            restricting the GT set (and matched preds) to each bin.
        """
        # TODO(hook): use ultralytics DetMetrics for the overall numbers; for the
        # per-bin split, filter GT by size_bin() and recompute matches per bin.
        raise NotImplementedError
