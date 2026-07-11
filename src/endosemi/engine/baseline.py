"""Supervised baseline (§9.1 comparison point).

Trains a single YOLO26 on ONLY the labeled subset (labeled_<pct>.txt) with the
stock ultralytics pipeline. This is the number the SSL model must beat at the
same label ratio. Uses the same val split as SSL for a fair comparison.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from ..config import Cfg, Paths
from ..utils.logging import get_logger

log = get_logger(__name__)


def _write_baseline_yaml(cfg: Cfg, paths: Paths) -> str:
    """Build a temp data.yaml whose train = labeled subset, val = shared val."""
    base = yaml.safe_load(paths.rel(cfg["data"]["yaml"]).read_text())
    dataset_root = Path(base["path"])
    labeled = Path(cfg["data"]["labeled_list"])
    labeled = labeled if labeled.is_absolute() else (dataset_root / labeled.name)

    doc = {
        "path": str(dataset_root),
        "train": str(labeled),
        "val": base["val"],
        "nc": base.get("nc", 1),
        "names": base.get("names", ["polyp"]),
    }
    tmp = Path(tempfile.mkstemp(suffix="_baseline.yaml")[1])
    tmp.write_text(yaml.safe_dump(doc))
    log.info("baseline data.yaml -> %s (train=%s)", tmp, labeled.name)
    return str(tmp)


def train_baseline(config_path: str, arch: str | None = None, device: str | None = None):
    from ultralytics import YOLO

    cfg = Cfg.load(config_path)
    paths = Paths.from_config_file(config_path)
    data_yaml = _write_baseline_yaml(cfg, paths)

    arch = arch or (cfg["model"]["arch"] + "n.yaml"
                    if not cfg["model"]["arch"].endswith(".yaml") else cfg["model"]["arch"])
    model = YOLO(arch)
    return model.train(
        data=data_yaml,
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["model"]["imgsz"],
        batch=cfg["train"]["batch_labeled"],
        device=device or cfg.get_path("train.device", "cpu"),
        project=cfg["train"]["project"],
        name=cfg["train"]["name"] + "_baseline",
    )
