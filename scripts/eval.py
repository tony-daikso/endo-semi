#!/usr/bin/env python3
"""Evaluate a trained detector on the val split.

    python scripts/eval.py --config configs/endo_semi_yolo26.yaml --weights runs/endo_semi_10pct/best.pt

Reports overall precision / recall / mAP@50 / mAP@50-95. (Size-stratified
breakdown, §8, arrives in step 8 via SizeStratifiedEvaluator.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from endosemi.config import Cfg
from endosemi.data import build_loader
from endosemi.engine import DetectionEvaluator
from endosemi.models import YOLO26Wrapper


def _val_list(cfg: Cfg) -> str:
    doc = yaml.safe_load(Path(cfg["data"]["yaml"]).read_text())
    val = Path(doc["val"])
    return str(val if val.is_absolute() else Path(doc["path"]) / val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default=None, help="cpu | mps | 0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    device = args.device or cfg.get_path("train.device", "cpu")
    imgsz = cfg["model"]["imgsz"]

    arch = cfg["model"]["arch"]
    arch = arch if arch.endswith(".yaml") else f"{arch}n.yaml"
    net = YOLO26Wrapper(arch, weights=args.weights, nc=cfg["data"].get("nc", 1),
                        imgsz=imgsz, device=device)

    loader = build_loader(_val_list(cfg), imgsz, cfg.get_path("eval.eval_batch", 16),
                          with_labels=True, shuffle=False,
                          num_workers=cfg.get_path("train.num_workers", 4),
                          limit=args.limit, drop_last=False)
    evaluator = DetectionEvaluator(imgsz, device, conf=cfg.get_path("eval.conf", 0.001))
    results = evaluator.evaluate(net, loader)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
