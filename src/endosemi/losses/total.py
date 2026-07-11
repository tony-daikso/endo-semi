"""Total objective (§6).

    L(x_l) = L_s(x_l) + 0.5 * L_det_cross(x_l) + 0.5 * L_m(x_l)
    L(x_u) = 0.5 * L_det_cross(x_u, x_u_s)

Weights come from config `loss` (w_cross, w_mutual, w_unlabeled_cross). The 0.5
split was tuned for the segmentation task — treat as a starting point and tune
on the validation set (§6).

Component switches (config `components`) let terms be zeroed to reproduce the
§9 build order incrementally.
"""
from __future__ import annotations


def total_labeled_loss(supervised_loss, cross_loss, mutual_loss, cfg) -> object:
    """L(x_l). Any term may be None/0 when its component is disabled."""
    w = cfg["loss"]
    comp = cfg["components"]
    total = supervised_loss
    if comp.get("cross_supervision") and cross_loss is not None:
        total = total + w["w_cross"] * cross_loss
    if comp.get("mutual_learning") and mutual_loss is not None:
        total = total + w["w_mutual"] * mutual_loss
    return total


def total_unlabeled_loss(cross_uc_loss, joint_loss, cfg) -> object:
    """L(x_u) = w_unlabeled_cross * (uncertainty-guided cross-sup + joint pseudo).

    Per §4's "full cross detection loss", the unlabeled objective is the sum of:
      - uncertainty-guided cross-supervision on the weak view (B2_uc / B1_uc)
      - joint pseudo-label supervision on the strong view (Bj_uc)
    both scaled by the same 0.5 unlabeled weight.
    """
    w = cfg["loss"]
    comp = cfg["components"]
    parts = []
    if comp.get("cross_supervision") and cross_uc_loss is not None:
        parts.append(cross_uc_loss)
    if comp.get("joint_pseudo") and joint_loss is not None:
        parts.append(joint_loss)
    if not parts:
        return None
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return w["w_unlabeled_cross"] * total
