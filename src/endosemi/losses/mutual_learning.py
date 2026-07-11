"""Component 4 — multi-level mutual learning (§5). Labeled data ONLY.

    L_m(x_l) = L_ssim(f1_backbone, f2_backbone)
             + 0.5 * [ KL(p1_cls || p2_cls) + KL(p2_cls || p1_cls) ]
             + 2   * L_mse(box1_matched, box2_matched)

Key difference from segmentation (§5): pixel-level alignment is automatic
(both nets output the same H×W grid). For detection, cls/box alignment applies
ONLY where both networks produce a prediction for the SAME ground-truth object.
Because we have y_l here, use **GT-anchor assignment** to identify corresponding
anchors across the two networks — NOT prediction-to-prediction matching.
"""
from __future__ import annotations


def mutual_learning_loss(f1, f2, x_l, y_l, weights: dict):
    """
    f1, f2  : YOLO26Wrapper
    x_l, y_l: labeled images and GT boxes
    weights : {'ssim','kl','box_mse'} from config loss.mutual
    """
    feat1 = f1.features(x_l)
    feat2 = f2.features(x_l)
    l_ssim = _feature_alignment(feat1, feat2)                       # SSIM or MSE

    # GT-anchor assignment: for each GT box, find the anchor(s) each net assigns
    # it to; align cls logits + box params only at those shared anchor positions.
    anchors = _gt_anchor_assignment(f1, f2, x_l, y_l)               # TODO(hook)
    l_kl = _symmetric_kl_cls(f1, f2, anchors)                       # TODO(hook)
    l_box = _matched_box_mse(f1, f2, anchors)                       # TODO(hook)

    return (weights["ssim"] * l_ssim
            + weights["kl"] * l_kl
            + weights["box_mse"] * l_box)


# -- pieces ------------------------------------------------------------------
def _feature_alignment(feat1, feat2):
    """SSIM or feature-map MSE between backbone/neck (FPN) outputs (§5 row 1)."""
    # TODO(hook): SSIM over matched-resolution FPN levels, or MSE fallback.
    raise NotImplementedError


def _gt_anchor_assignment(f1, f2, x_l, y_l):
    """Return, per GT object, the anchor indices each net assigns to it, so the
    two nets' predictions are compared at corresponding positions (§5)."""
    # TODO(hook): reuse YOLO26's TaskAlignedAssigner on y_l for BOTH nets;
    # intersect assigned anchors so alignment happens only where both agree.
    raise NotImplementedError


def _symmetric_kl_cls(f1, f2, anchors):
    """0.5*(KL(p1||p2)+KL(p2||p1)) over matched anchor class logits (§5 row 2)."""
    raise NotImplementedError


def _matched_box_mse(f1, f2, anchors):
    """MSE on DFL distributions / regressed box params at matched anchors (§5 row 3)."""
    raise NotImplementedError
