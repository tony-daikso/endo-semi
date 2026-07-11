#!/usr/bin/env python3
"""Evaluate a trained detector, stratified by lesion size (§8).

    python scripts/eval.py --config configs/endo_semi_yolo26.yaml --weights runs/.../best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from endosemi.config import Cfg, Paths
from endosemi.engine import SizeStratifiedEvaluator
from endosemi.models import YOLO26Wrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    paths = Paths.from_config_file(args.config)
    data_yaml = yaml.safe_load(paths.rel(cfg["data"]["yaml"]).read_text())
    val_list = str(Path(data_yaml["path"]) / data_yaml["val"])

    model = YOLO26Wrapper(cfg["model"]["arch"], weights=args.weights,
                          imgsz=cfg["model"]["imgsz"])
    evaluator = SizeStratifiedEvaluator(cfg)
    results = evaluator.evaluate(model, val_list, cfg["model"]["imgsz"])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
