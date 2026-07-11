"""Validation metrics.

``DetectionEvaluator`` computes the standard detection metrics on the held-out
val split, reusing ultralytics' ``DetMetrics`` + IoU matching so the numbers are
directly comparable to a stock ``YOLO.val()`` / the supervised baseline:

    precision, recall, mAP@50, mAP@50-95

Recall is the clinically important one for polyps (missed lesion = missed
cancer). This is what tells you whether SSL actually beats the baseline — loss
alone cannot (cross-supervision loss just measures net-to-net agreement).

``SizeStratifiedEvaluator`` (§8) will layer per-size-bin breakdowns on top; it
is still a step-8 stub.
"""
from __future__ import annotations

import numpy as np
import torch

from ultralytics.utils.metrics import DetMetrics, box_iou

from ..utils.logging import get_logger
from ..data import move_batch

log = get_logger(__name__)

# 10 IoU thresholds 0.50:0.05:0.95 — the standard COCO/YOLO mAP ladder.
IOUV = torch.linspace(0.5, 0.95, 10)


def match_predictions(pred_classes: torch.Tensor, true_classes: torch.Tensor,
                      iou: torch.Tensor, iouv: torch.Tensor) -> np.ndarray:
    """Correct-prediction matrix (N_pred, 10), one column per IoU threshold.

    Faithful port of ultralytics BaseValidator.match_predictions (greedy,
    highest-IoU-first, one GT per prediction). `iou` is (N_gt, N_pred).
    """
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes          # (N_gt, N_pred)
    iou = (iou * correct_class).cpu().numpy()
    for i, thr in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= thr)).T               # (k, 2): [gt, pred]
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
        # (scipy Hungarian variant omitted; greedy matches ultralytics' default)
    return correct


class DetectionEvaluator:
    """Overall val metrics for one detector."""

    def __init__(self, imgsz: int, device: str, conf: float = 0.001,
                 names: dict | None = None):
        self.imgsz = imgsz
        self.device = device
        self.conf = conf                      # low floor -> full PR curve (YOLO uses 0.001)
        self.names = names or {0: "polyp"}

    @torch.no_grad()
    def evaluate(self, net, val_loader) -> dict:
        metrics = DetMetrics()
        metrics.names = self.names
        metrics.clear_stats()
        iouv = IOUV.to(self.device)

        for batch in val_loader:
            batch = move_batch(batch, self.device)
            dets = net.predict(batch["img"], conf=self.conf)
            bi = batch["batch_idx"]
            for si, d in enumerate(dets):
                # ground truth for image si (normalized xywh -> xyxy pixel)
                m = bi == si
                gt_xywh = batch["bboxes"][m]
                gt_cls = batch["cls"][m].view(-1)
                gt_xyxy = self._xywhn_to_xyxy(gt_xywh)

                pred_xyxy = torch.as_tensor(d.boxes_xyxy, dtype=torch.float32, device=self.device)
                pred_cls = torch.as_tensor(d.classes, dtype=torch.float32, device=self.device)
                pred_conf = torch.as_tensor(d.scores, dtype=torch.float32, device=self.device)

                no_pred = pred_cls.shape[0] == 0
                if gt_cls.shape[0] == 0 or no_pred:
                    tp = np.zeros((pred_cls.shape[0], len(iouv)), dtype=bool)
                else:
                    iou = box_iou(gt_xyxy, pred_xyxy)             # (N_gt, N_pred)
                    tp = match_predictions(pred_cls, gt_cls, iou, iouv)

                tcls = gt_cls.cpu().numpy()
                metrics.update_stats({
                    "tp": tp,
                    "target_cls": tcls,
                    "target_img": np.unique(tcls),
                    "conf": np.zeros(0) if no_pred else pred_conf.cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else pred_cls.cpu().numpy(),
                    "im_name": d.__class__.__name__,
                })

        metrics.process()
        r = metrics.results_dict
        return {
            "precision": float(r.get("metrics/precision(B)", 0.0)),
            "recall": float(r.get("metrics/recall(B)", 0.0)),
            "map50": float(r.get("metrics/mAP50(B)", 0.0)),
            "map5095": float(r.get("metrics/mAP50-95(B)", 0.0)),
            "fitness": float(r.get("fitness", 0.0)),
        }

    def _xywhn_to_xyxy(self, xywh: torch.Tensor) -> torch.Tensor:
        if xywh.numel() == 0:
            return xywh.new_zeros((0, 4))
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        s = self.imgsz
        return torch.stack([(cx - w / 2) * s, (cy - h / 2) * s,
                            (cx + w / 2) * s, (cy + h / 2) * s], dim=1)


# ---------------------------------------------------------------------------
SIZE_NAMES = ["small", "medium", "large"]


def size_bin(area_px2: float, bins: list[float]) -> str:
    if area_px2 < bins[0]:
        return "small"
    if area_px2 < bins[1]:
        return "medium"
    return "large"


class SizeStratifiedEvaluator:
    """§8 — per-size-bin metrics. Wraps DetectionEvaluator; size split is step 8."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.bins = cfg["eval"]["size_bins_px2"]
        self.metrics = cfg["eval"]["metrics"]

    def evaluate(self, model, val_list: str, imgsz: int) -> dict:
        # TODO(step 8): run DetectionEvaluator once overall, then repeat with GT
        # filtered per size bin (size_bin() on GT px² area) for small/medium/large.
        raise NotImplementedError("size-stratified split is §9 step 8")
