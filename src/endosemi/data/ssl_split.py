"""Carve the training pool into a labeled subset + unlabeled remainder.

Split is **patient-level** using ``split.csv`` (columns: patient_id, split).
Frame filenames encode the patient id as the leading token, e.g.
``R0003_20220704_100122_000.jpg`` -> patient ``R0003``. Splitting by patient
(not by frame) prevents the same lesion appearing in both the labeled and
unlabeled pools, which would leak supervision.

Only the ``train`` split is divided; ``val`` is never touched.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)


def patient_id_of(image_path: str) -> str:
    """`.../R0003_20220704_100122_000.jpg` -> `R0003`."""
    return Path(image_path).stem.split("_", 1)[0]


def make_ssl_split(
    train_txt: str | Path,
    label_ratio: float,
    seed: int = 42,
    out_labeled: str | Path | None = None,
    out_unlabeled: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return (labeled_paths, unlabeled_paths), splitting whole patients.

    ``label_ratio`` is the fraction of *patients* (≈ frames) placed in the
    labeled pool. §9.8 headline point is 0.10.
    """
    lines = [ln.strip() for ln in Path(train_txt).read_text().splitlines() if ln.strip()]

    by_patient: dict[str, list[str]] = {}
    for ln in lines:
        by_patient.setdefault(patient_id_of(ln), []).append(ln)

    patients = sorted(by_patient)
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_labeled = max(1, round(len(patients) * label_ratio))
    labeled_patients = set(patients[:n_labeled])

    labeled, unlabeled = [], []
    for pid, frames in by_patient.items():
        (labeled if pid in labeled_patients else unlabeled).extend(frames)

    log.info(
        "SSL split @ %.0f%%: %d/%d patients labeled -> %d labeled frames, %d unlabeled frames",
        label_ratio * 100, n_labeled, len(patients), len(labeled), len(unlabeled),
    )

    if out_labeled:
        Path(out_labeled).write_text("\n".join(labeled) + "\n")
    if out_unlabeled:
        Path(out_unlabeled).write_text("\n".join(unlabeled) + "\n")

    return labeled, unlabeled
