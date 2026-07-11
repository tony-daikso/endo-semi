"""Component 1 — cross-supervision (§2), real implementation.

    L_det_cross(x) = L_det(f1(x), B2~(x)) + L_det(f2(x), B1~(x))

Each network is supervised on the OTHER network's confidence-thresholded boxes,
treated as (detached) pseudo-ground-truth. Because YOLO26 is end-to-end, B~(x)
is just the model's own top-`max_det` output filtered by confidence — no NMS.

When uncertainty filtering is on (step 3+), pass already-gated box sets; this
function is agnostic to how the boxes were produced.
"""
from __future__ import annotations

import torch

from ..models.dual_yolo import DetOutput


def boxes_to_targets(dets: list[DetOutput], imgsz: int, device: str) -> dict:
    """Turn per-image predicted boxes (pixel xyxy) into a detached loss batch
    {'batch_idx','cls','bboxes'} with bboxes as normalized xywh.

    Detached: pseudo-targets must carry no gradient back into the teacher net.
    """
    batch_idx, cls_list, box_list = [], [], []
    for i, d in enumerate(dets):
        n = len(d.boxes_xyxy)
        if n == 0:
            continue
        b = torch.as_tensor(d.boxes_xyxy, dtype=torch.float32, device=device)
        # xyxy(px) -> xywh(norm)
        cx = (b[:, 0] + b[:, 2]) / 2 / imgsz
        cy = (b[:, 1] + b[:, 3]) / 2 / imgsz
        w = (b[:, 2] - b[:, 0]) / imgsz
        h = (b[:, 3] - b[:, 1]) / imgsz
        xywh = torch.stack([cx, cy, w, h], 1).clamp_(0.0, 1.0)
        box_list.append(xywh)
        cls_list.append(torch.as_tensor(d.classes, dtype=torch.float32, device=device).view(-1, 1))
        batch_idx.append(torch.full((n,), float(i), device=device))

    if not box_list:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, 1, device=device),
            "bboxes": torch.zeros(0, 4, device=device),
        }
    return {
        "batch_idx": torch.cat(batch_idx, 0).detach(),
        "cls": torch.cat(cls_list, 0).detach(),
        "bboxes": torch.cat(box_list, 0).detach(),
    }


def cross_supervision_loss(f1, f2, imgs, conf: float, imgsz: int, device: str):
    """L_det_cross(imgs). Each net predicts (no grad); the other net is trained
    on those boxes. Returns (loss, n_pseudo_f1, n_pseudo_f2)."""
    b1 = f1.predict(imgs, conf=conf)      # f1's boxes -> supervise f2
    b2 = f2.predict(imgs, conf=conf)      # f2's boxes -> supervise f1

    t_from_f1 = boxes_to_targets(b1, imgsz, device)
    t_from_f2 = boxes_to_targets(b2, imgsz, device)

    n1 = int(t_from_f1["bboxes"].shape[0])
    n2 = int(t_from_f2["bboxes"].shape[0])

    loss = imgs.new_zeros(())
    if n2:
        loss = loss + f1.loss(imgs, t_from_f2)
    if n1:
        loss = loss + f2.loss(imgs, t_from_f1)
    return loss, n1, n2
