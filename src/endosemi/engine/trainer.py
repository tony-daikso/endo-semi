"""SSL trainer.

STEP 1 (§9.1) is implemented and runnable: two independent YOLO26 + plain
cross-supervision (box pseudo-labels, NO uncertainty), validated against the
supervised baseline.

Per optimizer step:
  labeled batch x_l, y_l:
    L_s = f1.loss(GT) + f2.loss(GT)                         # supervised
    (after burn-in) + w_cross * cross_supervision(x_l)      # §2 on labeled
  unlabeled batch x_u:
    (after burn-in) w_unlabeled_cross * cross_supervision(x_u)   # §2 on unlabeled
  backprop the sum over BOTH nets' parameters.

Steps 2-7 (uncertainty gating, joint pseudo-labels, mutual learning, temporal)
attach at the marked hooks; their modules already exist and are unit-tested.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import torch

from ..config import Cfg
from ..models import DualYOLO26, build_hyp
from ..data import build_loader, move_batch
from ..losses.cross_supervision import cross_supervision_loss
from ..utils.logging import get_logger

log = get_logger(__name__)


class SSLTrainer:
    def __init__(self, cfg: Cfg, device: str | None = None, limit: int | None = None):
        self.cfg = cfg
        self.device = device or cfg.get_path("train.device", "cpu")
        if self.device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS unavailable, falling back to CPU")
            self.device = "cpu"

        self.imgsz = cfg["model"]["imgsz"]
        self.conf = cfg["uncertainty"]["conf_thresh"]
        self.burn_in = cfg["train"]["burn_in_epochs"]

        hyp = build_hyp(epochs=cfg["train"]["epochs"])
        self.model = DualYOLO26(cfg, device=self.device, hyp=hyp)
        self.f1, self.f2 = self.model.nets()

        # dataloaders
        d = cfg["data"]
        self.labeled = build_loader(
            d["labeled_list"], self.imgsz, cfg["train"]["batch_labeled"],
            with_labels=True, shuffle=True,
            num_workers=cfg.get_path("train.num_workers", 4), limit=limit)
        self.unlabeled = build_loader(
            d["unlabeled_list"], self.imgsz, cfg["train"]["batch_unlabeled"],
            with_labels=False, shuffle=True,
            num_workers=cfg.get_path("train.num_workers", 4), limit=limit)

        params = list(self.f1.parameters()) + list(self.f2.parameters())
        self.optimizer = torch.optim.SGD(
            params, lr=cfg["train"]["lr0"], momentum=0.9, weight_decay=5e-4)
        log.info("Trainer ready | device=%s labeled_batches=%d unlabeled_batches=%d",
                 self.device, len(self.labeled), len(self.unlabeled))

    # -- one optimizer step --------------------------------------------------
    def train_step(self, labeled_batch, unlabeled_batch, epoch: int) -> dict:
        comp = self.cfg["components"]
        w = self.cfg["loss"]
        burn = epoch < self.burn_in

        lb = move_batch(labeled_batch, self.device)
        x_l = lb["img"]
        tgt_l = {"batch_idx": lb["batch_idx"], "cls": lb["cls"], "bboxes": lb["bboxes"]}

        self.f1.train(); self.f2.train()
        self.optimizer.zero_grad()

        # supervised term (both nets on GT)
        l_sup = self.f1.loss(x_l, tgt_l) + self.f2.loss(x_l, tgt_l)
        logs = {"L_sup": float(l_sup.detach())}
        total = l_sup

        # cross-supervision (§2), enabled after burn-in
        if not burn and comp["cross_supervision"]:
            l_cross_l, _, _ = cross_supervision_loss(
                self.f1, self.f2, x_l, self.conf, self.imgsz, self.device)
            total = total + w["w_cross"] * l_cross_l
            logs["L_cross_l"] = float(l_cross_l.detach())

            ub = move_batch(unlabeled_batch, self.device)
            l_cross_u, n1, n2 = cross_supervision_loss(
                self.f1, self.f2, ub["img"], self.conf, self.imgsz, self.device)
            total = total + w["w_unlabeled_cross"] * l_cross_u
            logs["L_cross_u"] = float(l_cross_u.detach())
            logs["n_pseudo"] = n1 + n2
            # STEP 2+ HOOK: replace plain cross-sup on x_u with the
            # uncertainty-gated + joint-pseudo path (see losses.total,
            # uncertainty/, losses.joint_pseudo).

        total.backward()
        self.optimizer.step()
        logs["L_total"] = float(total.detach())
        return logs

    # -- full training -------------------------------------------------------
    def fit(self) -> None:
        epochs = self.cfg["train"]["epochs"]
        for epoch in range(epochs):
            u_iter = itertools.cycle(self.unlabeled)   # unlabeled pool paired to labeled
            for it, lb in enumerate(self.labeled):
                logs = self.train_step(lb, next(u_iter), epoch)
                if it % 10 == 0:
                    msg = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in logs.items())
                    log.info("epoch %d/%d it %d | %s", epoch, epochs, it, msg)
            self.save(epoch)

    def save(self, epoch: int) -> None:
        out = Path(self.cfg["train"]["project"]) / self.cfg["train"]["name"]
        out.mkdir(parents=True, exist_ok=True)
        for name, net in (("f1", self.f1), ("f2", self.f2)):
            torch.save({"model": net.model, "epoch": epoch}, out / f"{name}_last.pt")
        log.info("saved checkpoints to %s", out)
