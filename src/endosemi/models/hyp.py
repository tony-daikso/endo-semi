"""Hyperparameter namespace for the detection loss.

``v8DetectionLoss`` / ``E2ELoss`` read gains + schedule off ``model.args``:
  - box / cls / dfl : loss-term gains
  - epochs          : E2ELoss one2many->one2one decay schedule

We start from ultralytics' DEFAULT_CFG so every key the criterion might touch
exists, then override what we care about.
"""
from __future__ import annotations

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


def build_hyp(epochs: int = 100, overrides: dict | None = None):
    args = get_cfg(DEFAULT_CFG)
    args.epochs = epochs
    if overrides:
        for k, v in overrides.items():
            setattr(args, k, v)
    return args
