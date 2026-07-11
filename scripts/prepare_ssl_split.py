#!/usr/bin/env python3
"""Carve the labeled subset out of the training pool at a given label ratio (§9.8).

    python scripts/prepare_ssl_split.py --config configs/endo_semi_yolo26.yaml --label-ratio 0.10

Writes labeled_<pct>.txt and unlabeled_<pct>.txt next to the dataset, and prints
the paths to drop into the config's data.labeled_list / data.unlabeled_list.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from endosemi.config import Cfg, Paths
from endosemi.data import make_ssl_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--label-ratio", type=float, default=None,
                    help="override config data.label_ratio")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    paths = Paths.from_config_file(args.config)
    ratio = args.label_ratio if args.label_ratio is not None else cfg["data"]["label_ratio"]

    data_yaml = yaml.safe_load(paths.rel(cfg["data"]["yaml"]).read_text())
    dataset_root = Path(data_yaml["path"])
    train_txt = dataset_root / data_yaml["train"]

    # Write to the exact paths the config/trainer expect (repo-relative). The
    # list *contents* are absolute image paths, so the list file location is
    # arbitrary — we just keep it where everything else looks for it.
    out_labeled = paths.rel(cfg["data"]["labeled_list"])
    out_unlabeled = paths.rel(cfg["data"]["unlabeled_list"])
    out_labeled.parent.mkdir(parents=True, exist_ok=True)

    make_ssl_split(
        train_txt=train_txt,
        label_ratio=ratio,
        seed=cfg["data"]["ssl_split_seed"],
        out_labeled=out_labeled,
        out_unlabeled=out_unlabeled,
    )
    print(f"labeled_list:   {out_labeled}")
    print(f"unlabeled_list: {out_unlabeled}")


if __name__ == "__main__":
    main()
