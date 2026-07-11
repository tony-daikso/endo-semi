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
import yaml

from ..config import Cfg
from ..models import DualYOLO26, build_hyp
from ..data import build_loader, move_batch
from ..losses.cross_supervision import cross_supervision_loss
from .evaluator import DetectionEvaluator
from ..utils.logging import get_logger

log = get_logger(__name__)


def _resolve_val_list(cfg: Cfg) -> str | None:
    """Absolute path to val.txt from the dataset data.yaml (path + val)."""
    try:
        doc = yaml.safe_load(Path(cfg["data"]["yaml"]).read_text())
        val = Path(doc["val"])
        return str(val if val.is_absolute() else Path(doc["path"]) / val)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not resolve val list: %s", exc)
        return None


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

        # validation
        nw = cfg.get_path("train.num_workers", 4)
        val_list = _resolve_val_list(cfg)
        self.val_loader = None
        if val_list and Path(val_list).exists():
            self.val_loader = build_loader(
                val_list, self.imgsz, cfg.get_path("eval.eval_batch", 16),
                with_labels=True, shuffle=False, num_workers=nw,
                limit=limit, drop_last=False)
            self.evaluator = DetectionEvaluator(
                self.imgsz, self.device, conf=cfg.get_path("eval.conf", 0.001))
            self.eval_interval = cfg.get_path("eval.interval", 1)
            self.select_by = cfg.get_path("eval.select_by", "map50")
        else:
            log.warning("val list not found (%s) — training without validation metrics", val_list)
        self.best_metric = -1.0

        params = list(self.f1.parameters()) + list(self.f2.parameters())
        self.optimizer = torch.optim.SGD(
            params, lr=cfg["train"]["lr0"], momentum=0.9, weight_decay=5e-4)
        log.info("Trainer ready | device=%s labeled_batches=%d unlabeled_batches=%d val=%s",
                 self.device, len(self.labeled), len(self.unlabeled),
                 len(self.val_loader) if self.val_loader else "off")

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

    # -- validation ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> dict | None:
        """Evaluate BOTH nets on val; log metrics; return the better net's stats
        tagged with which net won (used to pick best.pt)."""
        if self.val_loader is None:
            return None
        m1 = self.evaluator.evaluate(self.f1, self.val_loader)
        m2 = self.evaluator.evaluate(self.f2, self.val_loader)
        for tag, m in (("f1", m1), ("f2", m2)):
            log.info("epoch %d val %s | mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f",
                     epoch, tag, m["map50"], m["map5095"], m["precision"], m["recall"])
        best_tag, best = ("f1", m1) if m1[self.select_by] >= m2[self.select_by] else ("f2", m2)
        best = dict(best); best["net"] = best_tag
        return best

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

            self.save(epoch)   # always keep *_last.pt

            last = epoch == epochs - 1
            if self.val_loader is not None and (last or epoch % self.eval_interval == 0):
                best = self.validate(epoch)
                if best and best[self.select_by] > self.best_metric:
                    self.best_metric = best[self.select_by]
                    self._save_best(epoch, best)

    def save(self, epoch: int) -> None:
        out = self._out_dir()
        for name, net in (("f1", self.f1), ("f2", self.f2)):
            torch.save({"model": net.model, "epoch": epoch}, out / f"{name}_last.pt")
        log.info("saved last checkpoints to %s", out)

    def _save_best(self, epoch: int, best: dict) -> None:
        out = self._out_dir()
        net = self.f1 if best["net"] == "f1" else self.f2
        torch.save({"model": net.model, "epoch": epoch, "metrics": best}, out / "best.pt")
        log.info("new best: %s %s=%.4f (epoch %d) -> best.pt",
                 best["net"], self.select_by, best[self.select_by], epoch)

    def _out_dir(self) -> Path:
        out = Path(self.cfg["train"]["project"]) / self.cfg["train"]["name"]
        out.mkdir(parents=True, exist_ok=True)
        return out
