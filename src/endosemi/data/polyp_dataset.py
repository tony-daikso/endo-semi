"""Minimal YOLO-format dataset + collate producing ultralytics loss batches.

A batch dict matches what ``v8DetectionLoss`` expects:
    img       : (B, 3, H, W) float in [0,1]
    batch_idx : (N,)  image index per target box
    cls       : (N, 1) class id
    bboxes    : (N, 4) normalized xywh
    im_file   : list[str]  (for debugging / video grouping)

Preprocessing uses ultralytics' ``LetterBox`` so geometry matches the supervised
baseline. Weak/strong augmentation (§3.1) plugs in here from step 2 — for step 1
(plain cross-supervision) we use letterbox-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics.data.augment import LetterBox


def _label_path_for(image_path: str) -> Path:
    p = Path(image_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def _read_list(list_txt: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(list_txt).read_text().splitlines() if ln.strip()]


class PolypDataset(Dataset):
    """YOLO detection frames from a .txt list of image paths.

    with_labels=False (unlabeled pool) still parses labels if present but the
    trainer ignores them — pseudo-labels are produced online.
    """

    def __init__(self, list_txt: str | Path, imgsz: int = 640,
                 with_labels: bool = True, limit: int | None = None):
        self.paths = _read_list(list_txt)
        if limit:
            self.paths = self.paths[:limit]
        self.imgsz = imgsz
        self.with_labels = with_labels
        self.letterbox = LetterBox((imgsz, imgsz), auto=False, scaleup=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        path = self.paths[i]
        img = cv2.imread(path)                       # BGR HxWx3
        if img is None:
            raise FileNotFoundError(path)
        img = self.letterbox(image=img)              # -> (imgsz, imgsz, 3)

        boxes = np.zeros((0, 4), dtype=np.float32)   # normalized xywh
        cls = np.zeros((0, 1), dtype=np.float32)
        lbl = _label_path_for(path)
        if self.with_labels and lbl.exists():
            rows = [r.split() for r in lbl.read_text().strip().splitlines() if r.strip()]
            if rows:
                arr = np.array(rows, dtype=np.float32)
                cls = arr[:, :1]
                boxes = arr[:, 1:5]                   # already normalized xywh
                # NOTE: letterbox is centered + square with scaleup; for the
                # square source frames here normalized xywh is preserved. For
                # non-square sources, transform boxes with the letterbox params
                # (added alongside weak-aug in step 2).

        img_t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return {
            "img": img_t,
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(boxes),
            "im_file": path,
        }


def collate(samples: list[dict]) -> dict:
    """Stack a list of PolypDataset items into an ultralytics loss batch."""
    imgs = torch.stack([s["img"] for s in samples], 0)
    cls, bboxes, batch_idx, files = [], [], [], []
    for i, s in enumerate(samples):
        n = s["bboxes"].shape[0]
        if n:
            cls.append(s["cls"])
            bboxes.append(s["bboxes"])
            batch_idx.append(torch.full((n,), float(i)))
        files.append(s["im_file"])
    return {
        "img": imgs,
        "cls": torch.cat(cls, 0) if cls else torch.zeros(0, 1),
        "bboxes": torch.cat(bboxes, 0) if bboxes else torch.zeros(0, 4),
        "batch_idx": torch.cat(batch_idx, 0) if batch_idx else torch.zeros(0),
        "im_file": files,
    }


def build_loader(list_txt: str | Path, imgsz: int, batch_size: int,
                 with_labels: bool = True, shuffle: bool = True,
                 num_workers: int = 4, limit: int | None = None,
                 drop_last: bool = True) -> DataLoader:
    ds = PolypDataset(list_txt, imgsz=imgsz, with_labels=with_labels, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate, drop_last=drop_last)


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out
