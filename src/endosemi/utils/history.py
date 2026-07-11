"""Per-epoch CSV logger (ultralytics-style results.csv).

One row per epoch: train losses + val metrics for both nets. The column set is
fixed from the first row; later rows are reordered to match and missing keys are
left blank (e.g. epochs where validation didn't run).
"""
from __future__ import annotations

import csv
from pathlib import Path

from .logging import get_logger

log = get_logger(__name__)


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header: list[str] | None = None
        # resume-safe: if a results.csv already exists, adopt its header
        if self.path.exists():
            try:
                first = self.path.read_text().splitlines()[0]
                self.header = first.split(",")
            except Exception:  # pragma: no cover - defensive
                self.header = None

    def log(self, row: dict) -> None:
        row = {k: _fmt(v) for k, v in row.items()}
        if self.header is None:
            self.header = list(row.keys())
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.header)
        ordered = [row.get(col, "") for col in self.header]
        with open(self.path, "a", newline="") as fh:
            csv.writer(fh).writerow(ordered)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return v
