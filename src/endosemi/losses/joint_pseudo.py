"""Component 3 — joint pseudo-label supervision (§4).

Fuse the two networks' (uncertainty-annotated) boxes into a single joint set:

  - matched pair (IoU >= thresh): keep the box from the LOWER-uncertainty net.
  - unmatched box (only one net saw it): no cross-validation, so admit it only
    under a STRICTER threshold (strict_factor * per-batch T, default 0.5x).

Then the joint set is filtered again by the per-batch dynamic threshold (§3.3)
to yield Bj_uc, used on the STRONG view (§4 full cross loss).

This module implements the fusion (`joint_pseudo_labels`) and the strong-view
loss term (`joint_cross_loss`).
"""
from __future__ import annotations

import numpy as np

from ..matching.box_match import iou_match


def joint_pseudo_labels(
    boxes1_xyxy: np.ndarray, u1: np.ndarray,
    boxes2_xyxy: np.ndarray, u2: np.ndarray,
    iou_thresh: float = 0.5,
    algorithm: str = "hungarian",
    strict_threshold: float | None = None,
):
    """Return (joint_boxes_xyxy, joint_uncertainty, provenance).

    provenance[i] in {'f1','f2','f1_solo','f2_solo'} — useful for debugging /
    ablations on how many joint boxes came from single-network detections.
    """
    b1 = np.asarray(boxes1_xyxy, dtype=float).reshape(-1, 4)
    b2 = np.asarray(boxes2_xyxy, dtype=float).reshape(-1, 4)
    u1 = np.asarray(u1, dtype=float).ravel()
    u2 = np.asarray(u2, dtype=float).ravel()

    m = iou_match(b1, b2, iou_thresh=iou_thresh, algorithm=algorithm)

    joint_boxes: list[np.ndarray] = []
    joint_u: list[float] = []
    prov: list[str] = []

    # matched: pick lower-uncertainty box (§4)
    for i, j in m.matches:
        if u1[i] <= u2[j]:
            joint_boxes.append(b1[i]); joint_u.append(u1[i]); prov.append("f1")
        else:
            joint_boxes.append(b2[j]); joint_u.append(u2[j]); prov.append("f2")

    # unmatched: single-network detections require a stricter admission bar (§4)
    if strict_threshold is not None:
        for i in m.unmatched1:
            if u1[i] < strict_threshold:
                joint_boxes.append(b1[i]); joint_u.append(u1[i]); prov.append("f1_solo")
        for j in m.unmatched2:
            if u2[j] < strict_threshold:
                joint_boxes.append(b2[j]); joint_u.append(u2[j]); prov.append("f2_solo")

    if joint_boxes:
        return np.stack(joint_boxes), np.asarray(joint_u), prov
    return np.zeros((0, 4)), np.zeros((0,)), []


def joint_cross_loss(f1, f2, x_strong, joint_targets):
    """Strong-view supervision from the joint pseudo-labels (§4):

        L_det(f1(x_u_s), Bj_uc) + L_det(f2(x_u_s), Bj_uc)

    `joint_targets` is a {'batch_idx','cls','bboxes'} batch built from the fused
    joint boxes (see losses.cross_supervision.boxes_to_targets). Wired in §9 step 4.
    """
    return f1.loss(x_strong, joint_targets) + f2.loss(x_strong, joint_targets)
