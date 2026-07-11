# Endo-SemiS → YOLO26 : Semi-Supervised Polyp Detection

Semi-supervised **object detection** for colorectal polyps, adapting the
Endo-SemiS framework (originally pixel-level segmentation) to a YOLO26
detection backbone/head.

The design follows [`instruction/Endo-SemiS_YOLO26_detection_adaptation.md`](instruction/Endo-SemiS_YOLO26_detection_adaptation.md).
Every module below is annotated with the spec section it implements (`§N`).

## The idea in one paragraph

Two independent YOLO26 models (`f1`, `f2`) teach each other. On **labeled**
frames they train on ground truth *and* mutually align their features/logits.
On **unlabeled** frames each network's NMS-filtered boxes become pseudo-labels
for the other, but only after being filtered by a **per-box uncertainty**
score (DFL edge entropy + MC-Dropout classification entropy) against a
dynamically computed threshold. A joint pseudo-label set is fused by IoU
matching (keep the lower-uncertainty box per matched pair). Finally, on video,
**ByteTrack** enforces temporal consistency — interpolating missed detections
(FN) and suppressing isolated spurious ones (FP).

## Component map

| Spec | Component | Module |
|---|---|---|
| §2 | Cross-supervision | [`losses/cross_supervision.py`](src/endosemi/losses/cross_supervision.py) |
| §3.1 | Aleatoric (weak/strong aug) | [`data/augment.py`](src/endosemi/data/augment.py) |
| §3.2 | Epistemic uncertainty (DFL + MC-Dropout) | [`uncertainty/`](src/endosemi/uncertainty/) |
| §3.3 | Dynamic thresholding (per-batch/window) | [`uncertainty/thresholding.py`](src/endosemi/uncertainty/thresholding.py) |
| §4 | Joint pseudo-label (IoU matching) | [`losses/joint_pseudo.py`](src/endosemi/losses/joint_pseudo.py), [`matching/box_match.py`](src/endosemi/matching/box_match.py) |
| §5 | Multi-level mutual learning | [`losses/mutual_learning.py`](src/endosemi/losses/mutual_learning.py) |
| §6 | Total objective | [`losses/total.py`](src/endosemi/losses/total.py) |
| §7 | Spatiotemporal correction (ByteTrack) | [`temporal/st_correction.py`](src/endosemi/temporal/st_correction.py) |

## Data

YOLO detection format, 1 class (`polyp`). Configured in
[`data/label-data/v3_2_1/data.yaml`](data/label-data/v3_2_1/data.yaml):
17,820 train / 4,347 val frames, split by `patient_id` (`split.csv`) to
prevent patient leakage. Raw colonoscopy videos for the temporal component
live in [`data/unlabel-video/`](data/unlabel-video/).

## Environment

Use the `polyp` conda env (Python 3.12, torch 2.12 + MPS, ultralytics 8.4.51,
numpy/scipy):

```bash
conda activate polyp
```

## Quickstart

```bash
# 1. Carve the labeled set out of the training pool at a given label ratio
python scripts/prepare_ssl_split.py --config configs/endo_semi_yolo26.yaml --label-ratio 0.10

# 2a. Supervised baseline on the labeled subset (the number SSL must beat)
python scripts/train_baseline.py --config configs/endo_semi_yolo26.yaml

# 2b. Semi-supervised training (dual YOLO26 + cross-supervision)
python scripts/train.py --config configs/endo_semi_yolo26.yaml --device mps
#     add --limit 8 --device cpu for a fast smoke run

# 3. Evaluate, stratified by lesion size (§8)              [TODO: step 8]
python scripts/eval.py --config configs/endo_semi_yolo26.yaml --weights runs/exp/f1_last.pt

# 4. Video inference + ByteTrack temporal correction (§7)  [TODO: step 7]
python scripts/infer_video.py --weights runs/exp/f1_last.pt --source data/unlabel-video/
```

## Two YOLO26 facts that change the spec (verified, not assumed)

The spec was written before YOLO26's head was known and hedged accordingly.
Confirmed against ultralytics 8.4.x:

1. **YOLO26 has no DFL** (`reg_max == 1`, single-value box regression). The
   spec's "free" DFL box-edge entropy (§3.2) is therefore **unavailable** — from
   step 2, per-box coordinate uncertainty must come from MC-Dropout, or the box
   term is dropped and only classification uncertainty is used. Flag this as a
   deviation from the spec.
2. **YOLO26 is end-to-end / NMS-free** (E2ELoss, one2one branch). Inference
   returns final `(x1,y1,x2,y2,conf,cls)` boxes directly, so §2's "run NMS to
   get B~(x)" collapses to a confidence threshold on the model's own output.

## Implementation order (from §9)

The scaffold is deliberately buildable in the spec's suggested order — each
step is a switch in the config so you can validate incrementally:

1. Dual YOLO26 + plain cross-supervision (no uncertainty) vs. supervised baseline
2. DFL box uncertainty (free) + MC-Dropout cls uncertainty (K=5)
3. Per-batch dynamic thresholding
4. IoU box matching for joint pseudo-labels — **unit-test in isolation first**
5. Mutual learning on labeled data (GT-anchor matching)
6. Combine + tune weights on validation
7. ByteTrack temporal correction
8. Evaluate at 10% label ratio first, then sweep

## Status

| Step (§9) | What | State |
|---|---|---|
| 1 | Dual YOLO26 + plain cross-supervision vs supervised baseline | **done, runs on real data** |
| 2 | MC-Dropout cls uncertainty (DFL path N/A — see above) | scaffolded + unit-tested |
| 3 | Per-batch dynamic thresholding | scaffolded + unit-tested |
| 4 | IoU box matching + joint pseudo-labels | scaffolded + unit-tested |
| 5 | Mutual learning on labeled data | interface only |
| 6 | Combined objective + weight tuning | wired for step 1 |
| 7 | ByteTrack temporal correction | scaffolded + unit-tested |
| 8 | Size-stratified evaluation | interface only |

**Step 1 is implemented and verified**: `models/dual_yolo.py`,
`data/polyp_dataset.py`, `losses/cross_supervision.py`, `engine/trainer.py`,
`engine/baseline.py`. Smoke-tested end-to-end (finite losses, cross-supervision
active, backward/step succeed). Steps 2-8 have working, unit-tested building
blocks; they attach at the `STEP N+ HOOK` markers in `engine/trainer.py`.

Run the tests: `PYTHONPATH=src python -m pytest tests/` (27 pass; they cover
matching, thresholding, uncertainty math, joint fusion, temporal flagging, and
the patient-level split — none require torch).
