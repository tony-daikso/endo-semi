"""Dual YOLO26 (§1) — real implementation over ultralytics DetectionModel.

Two independent YOLO26 models f1, f2: same architecture, different init, no
weight sharing. Each exposes exactly the two hooks the SSL loop needs:

  - ``loss(imgs, targets)``  -> scalar detection loss  (training)
  - ``predict(imgs, conf)``  -> per-image boxes         (pseudo-labels)

Important YOLO26 facts (verified against ultralytics 8.4.x, NOT assumed by the
spec):
  * reg_max == 1  -> **YOLO26 has NO DFL box head.** The spec's "free" DFL box
    uncertainty (§3.2) is therefore unavailable; box uncertainty must come from
    MC-Dropout or be dropped. This only matters from step 2 onward.
  * The head is **end-to-end / NMS-free** (E2ELoss, one2one branch). Inference
    returns the final top-`max_det` boxes directly as (x1,y1,x2,y2,conf,cls) in
    letterboxed-pixel space — so §2's "run NMS" step is a no-op here; we just
    confidence-threshold the model's own output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ultralytics.nn.tasks import DetectionModel

from .hyp import build_hyp
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DetOutput:
    """Per-image detections in letterboxed-pixel space."""

    boxes_xyxy: np.ndarray            # (n, 4)
    scores: np.ndarray                # (n,)
    classes: np.ndarray               # (n,)
    boxes_xyxy_t: torch.Tensor | None = None   # same boxes, on-device tensor


class YOLO26Wrapper:
    """Adapter over one ultralytics YOLO26 DetectionModel."""

    def __init__(self, arch: str = "yolo26n.yaml", weights: str | None = None,
                 nc: int = 1, imgsz: int = 640, init_seed: int = 0,
                 hyp=None, device: str = "cpu"):
        torch.manual_seed(init_seed)
        self.device = device
        self.imgsz = imgsz
        self.model = DetectionModel(arch, nc=nc, verbose=False).to(device)
        if weights:
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            state = ckpt["model"].float().state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state, strict=False)
        self.model.args = hyp if hyp is not None else build_hyp()
        self.model.criterion = self.model.init_criterion()
        log.info("YOLO26 (%s) nc=%d seed=%d on %s | DFL=%s end2end=%s",
                 arch, nc, init_seed, device,
                 self.model.model[-1].reg_max > 1, getattr(self.model, "end2end", False))

    # -- torch passthrough ---------------------------------------------------
    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train(); return self

    def eval(self):
        self.model.eval(); return self

    # -- training hook -------------------------------------------------------
    def loss(self, imgs: torch.Tensor, targets: dict) -> torch.Tensor:
        """Scalar detection loss (box+cls+dfl summed) for `imgs` against `targets`.

        targets: {'batch_idx','cls','bboxes'} — bboxes normalized xywh. GT for
        supervised loss, or detached pseudo-boxes for cross-supervision (§2).
        """
        batch = {"img": imgs, **targets}
        loss_vec, _ = self.model.loss(batch)
        return loss_vec.sum()

    # -- inference hook ------------------------------------------------------
    @torch.no_grad()
    def predict(self, imgs: torch.Tensor, conf: float = 0.25) -> list[DetOutput]:
        """End2end inference -> per-image boxes above `conf`. NMS-free (§2)."""
        was_training = self.model.training
        self.model.eval()
        out = self.model(imgs)
        y = out[0] if isinstance(out, (list, tuple)) else out   # (B, N, 6)
        results: list[DetOutput] = []
        for det in y:                                            # (N, 6)
            keep = det[:, 4] >= conf
            d = det[keep]
            results.append(DetOutput(
                boxes_xyxy=d[:, :4].cpu().numpy(),
                scores=d[:, 4].cpu().numpy(),
                classes=d[:, 5].cpu().numpy(),
                boxes_xyxy_t=d[:, :4],
            ))
        if was_training:
            self.model.train()
        return results

    def features(self, imgs):
        """Backbone/neck feature maps for mutual learning (§5). TODO(step 5)."""
        raise NotImplementedError("mutual learning is §9 step 5")


class DualYOLO26:
    """Holds f1 and f2 with independent inits (§1)."""

    def __init__(self, cfg: dict, device: str = "cpu", hyp=None):
        m = cfg["model"]
        seeds = m.get("init_seeds", [1, 2])
        data = cfg["data"]
        nc = data.get("nc", 1)
        common = dict(arch=_arch_yaml(m["arch"]), weights=m.get("weights"),
                      nc=nc, imgsz=m["imgsz"], hyp=hyp, device=device)
        self.f1 = YOLO26Wrapper(init_seed=seeds[0], **common)
        self.f2 = YOLO26Wrapper(init_seed=seeds[1], **common)
        log.info("Dual YOLO26 initialized with seeds %s (no weight sharing)", seeds)

    def nets(self):
        return self.f1, self.f2


def _arch_yaml(arch: str) -> str:
    """'yolo26' -> 'yolo26n.yaml' (nano default). Pass a full '*.yaml' to override size."""
    return arch if arch.endswith(".yaml") else f"{arch}n.yaml"
