"""MC-Dropout for epistemic uncertainty on the classification/objectness side
(§1, §3.2).

Box-coordinate uncertainty comes for free from the DFL distribution
(see uncertainty/dfl_entropy.py) — MC-Dropout is only needed for cls/obj.

Usage:
    enable_mc_dropout(model, p=0.1)        # insert/activate dropout in the head
    probs = mc_dropout_forward(model, x, K=5)   # K stochastic passes
"""
from __future__ import annotations


def enable_mc_dropout(model, p: float = 0.1) -> None:
    """Insert dropout into the detection head's classification branch and keep
    those dropout layers in *train* mode at inference time.

    Only the head is stochastic — backbone/neck stay deterministic so the K
    passes share features and stay cheap.
    """
    # TODO(hook): locate the Detect head cls branch (model.model[-1]); wrap
    # conv/linear layers with nn.Dropout(p) if absent; register them so that
    # mc_dropout_forward can flip only THESE modules to .train() while the rest
    # of the network is in .eval().
    raise NotImplementedError


def mc_dropout_forward(model, images, K: int = 5):
    """Run K stochastic forward passes; return per-matched-box cls prob samples.

    Returns an object from which uncertainty/mc_dropout.py computes:
        P_i_cls = mean_k(cls_probs)
        U_i_cls = mean_k(entropy(cls_probs))
    Matching boxes across passes uses IoU association (positions jitter slightly
    between passes because only the head is stochastic, so IoU>0.9 is safe).
    """
    # TODO(hook): set head dropout to train(), rest eval(); loop K forwards;
    # associate boxes across passes; stack cls prob vectors -> (n_boxes, K, nc).
    raise NotImplementedError
