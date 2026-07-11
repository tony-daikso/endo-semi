"""Config loading. Thin wrapper over the YAML in configs/ with dotted access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict that also allows attribute access and nested dotted lookup.

    >>> c = Cfg.load("configs/endo_semi_yolo26.yaml")
    >>> c.model["imgsz"]
    640
    >>> c.get_path("uncertainty.threshold.mode")
    'batch'
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        return Cfg(val) if isinstance(val, dict) else val

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @classmethod
    def load(cls, path: str | Path) -> "Cfg":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw)


@dataclass
class Paths:
    """Resolved absolute paths derived from the config, rooted at repo root."""

    root: Path

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "Paths":
        # repo root = parent of the configs/ directory holding the config file
        return cls(root=Path(config_path).resolve().parent.parent)

    def rel(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)
