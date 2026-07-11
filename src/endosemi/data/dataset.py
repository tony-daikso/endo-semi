"""Datasets for the SSL loop.

- ``LabeledDataset``   : image + YOLO GT boxes. Used for L_s, cross-sup on
  labeled data, and mutual learning (§5, needs GT for anchor assignment).
- ``UnlabeledDataset`` : yields a (weak_view, strong_view) pair per frame
  (§3.1). No labels; pseudo-labels are produced online during training.
- ``SSLBatchSampler``  : pairs a labeled mini-batch with an unlabeled one each
  step, per §6's separate L(x_l) / L(x_u) objectives.

These wrap ultralytics' YOLO dataset/loader rather than reimplementing decode +
letterbox. The TODO(hook) markers are where we attach to it.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger
from .augment import WeakAugment, StrongAugment

log = get_logger(__name__)


def _read_list(list_txt: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(list_txt).read_text().splitlines() if ln.strip()]


def _label_path_for(image_path: str) -> str:
    """YOLO convention: images/<name>.jpg -> labels/<name>.txt."""
    p = Path(image_path)
    return str(p.parent.parent / "labels" / (p.stem + ".txt"))


class LabeledDataset:
    """Labeled frames with GT boxes (YOLO txt: cls cx cy w h, normalized)."""

    def __init__(self, list_txt: str | Path, imgsz: int = 640, augment: bool = True):
        self.paths = _read_list(list_txt)
        self.imgsz = imgsz
        self.augment = WeakAugment(imgsz=imgsz) if augment else None
        # TODO(hook): construct ultralytics.data.YOLODataset over self.paths so
        # letterbox, cache and collate match the supervised baseline exactly.

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img_path = self.paths[i]
        lbl_path = _label_path_for(img_path)
        # TODO(hook): load image + parse lbl_path into (n,5) targets, letterbox,
        # apply self.augment (weak geometry). Return tensors for the trainer.
        return {"image_path": img_path, "label_path": lbl_path}


class UnlabeledDataset:
    """Unlabeled frames -> (weak, strong) view pair for §3.1 aleatoric branch.

    The strong view is the weak view + intensity-only perturbations, so
    pseudo-boxes computed on the weak view align to the strong view without
    coordinate transforms (spec §3.1).
    """

    def __init__(self, list_txt: str | Path, imgsz: int = 640):
        self.paths = _read_list(list_txt)
        self.imgsz = imgsz
        self.weak = WeakAugment(imgsz=imgsz)
        self.strong = StrongAugment()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img_path = self.paths[i]
        # TODO(hook): load image; weak_view = self.weak(img); strong_view =
        # self.strong(weak_view). Track the weak geometric transform so pseudo-
        # boxes from the weak view can be reused on the strong view directly.
        return {"image_path": img_path}


class SSLBatchSampler:
    """Emit (labeled_batch, unlabeled_batch) tuples for one optimizer step.

    Cycles the (usually smaller) labeled loader against the unlabeled loader.
    During the §6 burn-in, the trainer ignores the unlabeled half.
    """

    def __init__(self, n_labeled: int, n_unlabeled: int, batch_l: int, batch_u: int):
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.batch_l = batch_l
        self.batch_u = batch_u

    def __len__(self) -> int:
        # One epoch = one pass over the (larger) unlabeled pool.
        return max(1, self.n_unlabeled // self.batch_u)
