#!/usr/bin/env python3
"""Train the semi-supervised dual-YOLO26 detector.

    python scripts/train.py --config configs/endo_semi_yolo26.yaml

Bring components up in the §9 order by toggling `components.*` in the config:
  1. cross_supervision only (no uncertainty) vs supervised baseline
  2. + uncertainty_filter
  3. + joint_pseudo
  4. + mutual_learning
  5. tune loss weights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from endosemi.config import Cfg
from endosemi.engine import SSLTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--device", default=None, help="cpu | mps | 0")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per split (smoke test)")
    # Quick overrides — leave unset to use the config (source of truth).
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--burn-in", type=int, default=None,
                    help="supervised-only warmup epochs before cross-supervision")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence floor for pseudo-label boxes")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.burn_in is not None:
        cfg["train"]["burn_in_epochs"] = args.burn_in
    if args.conf is not None:
        cfg["uncertainty"]["conf_thresh"] = args.conf

    trainer = SSLTrainer(cfg, device=args.device, limit=args.limit)
    trainer.fit()


if __name__ == "__main__":
    main()
