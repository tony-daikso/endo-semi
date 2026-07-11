#!/usr/bin/env python3
"""Train the supervised baseline on the labeled subset (§9.1 comparison point).

    python scripts/train_baseline.py --config configs/endo_semi_yolo26.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from endosemi.engine import train_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--arch", default=None, help="e.g. yolo26n.yaml / yolo26s.yaml")
    ap.add_argument("--device", default=None, help="cpu | mps | 0")
    args = ap.parse_args()
    train_baseline(args.config, arch=args.arch, device=args.device)


if __name__ == "__main__":
    main()
