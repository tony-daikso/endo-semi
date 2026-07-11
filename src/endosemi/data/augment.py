"""Weak / strong augmentation for the aleatoric branch (§3.1).

    weak   = geometric only (flip, scale, crop [, mosaic, copy-paste])
             -> box coordinates transform WITH the image.
    strong = weak + intensity/color (brightness, blur, noise)
             -> pixel-only, boxes DO NOT move.

Consequence used by the trainer: pseudo-boxes generated on the *weak* view can
supervise the network's prediction on the *strong* view directly, with no box
transform in between.

Ordering rule (§8): Copy-Paste changes object placement/count, so it belongs to
the WEAK (geometric) branch, applied BEFORE the strong intensity perturbations —
otherwise pasted boxes wouldn't correspond to the weak-view geometry.

CutMix caveat (§3.1): mixing regions creates partial/occluded boxes. Prefer
Mosaic (detection-native, already standard in YOLO). If CutMix is used, drop
boxes whose visible area after mixing falls below an IoU-with-original threshold.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeakAugment:
    """Geometric-only. Returns image + the transform so boxes can be mapped."""

    imgsz: int = 640
    flip_prob: float = 0.5
    scale: tuple[float, float] = (0.8, 1.2)
    crop: bool = True
    mosaic: bool = False
    copy_paste_prob: float = 0.0

    def __call__(self, sample: dict) -> dict:
        # TODO(hook): implement via ultralytics augment ops (RandomFlip,
        # RandomPerspective/scale, Mosaic, CopyPaste). Must return the applied
        # geometric transform matrix so boxes on the weak view are recoverable.
        raise NotImplementedError("wire to ultralytics geometric augment ops")


@dataclass
class StrongAugment:
    """Intensity/color-only on top of a weak view. Leaves box coordinates fixed."""

    brightness: float = 0.4
    blur_prob: float = 0.3
    noise_std: float = 0.02

    def __call__(self, weak_sample: dict) -> dict:
        # TODO(hook): ColorJitter / GaussianBlur / additive noise. No geometry.
        raise NotImplementedError("wire to intensity-only transforms")


def weak_strong_pair(sample: dict, weak: WeakAugment, strong: StrongAugment) -> tuple[dict, dict]:
    """Produce the (weak_view, strong_view) pair, guaranteeing the strong view is
    the weak view plus intensity only (shared geometry)."""
    weak_view = weak(sample)
    strong_view = strong(weak_view)
    return weak_view, strong_view
