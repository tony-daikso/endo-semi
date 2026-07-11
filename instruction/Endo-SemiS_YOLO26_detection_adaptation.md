# Endo-SemiS → YOLO26 Detection: Adaptation Spec

> Adapts the Endo-SemiS semi-supervised framework (originally pixel-level segmentation)
> to an **object detection** setting using YOLO26 as the backbone/head.
> Companion doc to `Endo-SemiS_technical_spec.md` (segmentation version).

---

## 0. Why This Isn't a 1:1 Port

Segmentation gives a dense per-pixel probability map — uncertainty and pseudo-labels are naturally pixelwise (`⊙` masking works directly). Detection gives a **sparse, variable-length set of boxes** (class, x, y, w, h, confidence) after NMS. Every component that relied on pixel-alignment needs a discrete-matching equivalent instead. The table below is the core translation:

| Segmentation concept | Detection (YOLO26) equivalent |
|---|---|
| Pixel probability map `P_i` | Per-box confidence score |
| Pixel entropy map `U_i` | Per-box uncertainty score (scalar) |
| `ỹ_i` (binarized mask) | NMS-filtered box set `{(cls, x,y,w,h, conf)}` |
| `U_i^b` pixel mask (keep/discard) | Per-box keep/discard filter before using as pseudo-label |
| Joint fusion via `M ⊙ P1 + (1-M) ⊙ P2` | IoU-based box matching, then pick lower-uncertainty box per matched pair |
| Decoder logit alignment (MSE) | Detection head output alignment on matched predictions |
| Frame-level `R_n` (foreground pixel count) | Frame-level object count / track continuity (ByteTrack) |

---

## 1. Architecture

```
Input x --> [YOLO26 backbone] --> [Neck/FPN] --> [Detection head] --> boxes: (cls, x, y, w, h, conf)
```

- Two independent YOLO26 models (`f1`, `f2`), same architecture, different init, no weight sharing — same as segmentation version.
- If YOLO26 keeps a **Distribution Focal Loss (DFL)** style box head (as in YOLOv8–v11), each box edge is predicted as a discrete probability distribution over bins rather than a single regressed number. **This is a gift for uncertainty estimation** — you get a natural per-edge entropy without needing MC-Dropout for box uncertainty (see §3.2).
- Insert MC-Dropout in the detection head (classification branch, and box branch if not using DFL) for epistemic uncertainty on the classification/objectness side.

---

## 2. Component 1 — Cross-Supervision (detection version)

For input `x`, run NMS on each network's raw predictions to get filtered box sets `B1(x)`, `B2(x)`. Each network is supervised using the **other network's boxes as pseudo-ground-truth**:

```
L_det_cross(x) = L_det(f1(x), B2_tilde(x)) + L_det(f2(x), B1_tilde(x))
```

- `L_det` = your normal YOLO26 detection loss (classification + DFL/box regression + objectness, whatever YOLO26's default is)
- `B_i_tilde(x)` = NMS-filtered, confidence-thresholded predictions from network i, treated as target boxes

This is a direct swap: same idea as CPS, just the "loss" and "target format" change from BCE-on-masks to detection-loss-on-boxes.

---

## 3. Component 2 — Uncertainty-Guided Pseudo-Label (detection version)

### 3.1 Aleatoric uncertainty (weak-to-strong, unchanged in spirit)
- Weak augmentation = geometric only (flip, scale, crop) → transform box coordinates accordingly, generate pseudo-boxes
- Strong augmentation = weak + intensity/color transforms (brightness, blur, noise — these don't move box coordinates, so pseudo-boxes from the weak view directly supervise the strong view's prediction)
- CutMix for detection needs care: mixing regions can create partial/occluded boxes — either drop boxes whose visible area falls below an IoU-with-original threshold after mixing, or use Mosaic (already standard in YOLO training) instead of CutMix, which is more detection-native.

### 3.2 Epistemic uncertainty

**If YOLO26 retains DFL-style box regression:**
```python
# per predicted box, per edge (left/top/right/bottom)
# DFL outputs a distribution over discrete bins for each edge
def dfl_edge_uncertainty(dfl_probs):  # dfl_probs: [n_bins] softmax distribution
    return entropy(dfl_probs)  # same entropy formula as segmentation

box_uncertainty = mean([dfl_edge_uncertainty(e) for e in box.four_edges])
```
No MC-Dropout forward passes needed for box coordinate uncertainty — it falls out of the existing DFL distribution for free.

**For classification/objectness confidence (still need MC-Dropout):**
```python
K = 5
cls_probs = [f_i(x, dropout=True).cls_prob for _ in range(K)]  # per matched box across passes
P_i_cls = mean(cls_probs)
U_i_cls = mean([entropy(p) for p in cls_probs])
```

**Combined per-box uncertainty** (simple weighted combination, tune the weight):
```python
U_box = alpha * U_i_cls + (1 - alpha) * box_uncertainty   # e.g. alpha = 0.5 as starting point
```

### 3.3 Dynamic thresholding (same rule, applied per-box instead of per-pixel)

```python
def dynamic_threshold(U_all_boxes_in_image):
    mu = U_all_boxes_in_image.mean()
    sigma = U_all_boxes_in_image.std()
    p95 = np.percentile(U_all_boxes_in_image, 95)
    return min(mu + sigma, p95)

T = dynamic_threshold(U_box_array)  # computed per-image across its own detected boxes
keep_mask = U_box_array < T
pseudo_boxes_uc = [b for b, keep in zip(pseudo_boxes, keep_mask) if keep]
```

**Caveat vs. segmentation:** with pixels you always have thousands of samples to compute μ/σ/P95 reliably. With boxes, a sparse-detection image (e.g. 1–3 nodules) gives you a near-meaningless mean/std over 1–3 values. **Recommendation:** accumulate uncertainty statistics over a mini-batch or a running window of recent images (not per-single-image) to get a stable threshold, then apply that shared threshold to filter that image's boxes. This is a genuine adaptation, not in the original paper — flag it as an experimental design choice if reproducing/publishing.

---

## 4. Component 3 — Joint Pseudo-Label Supervision (detection version)

This is the part requiring the most real redesign, because segmentation's "pick lower-uncertainty at each pixel" only works when both networks agree on *where* the pixel is. Detection needs explicit matching first.

```python
def joint_pseudo_labels(boxes1, U1, boxes2, U2, iou_thresh=0.5):
    matches, unmatched1, unmatched2 = iou_match(boxes1, boxes2, iou_thresh)  # Hungarian or greedy IoU matching

    joint_boxes = []
    for (b1, b2) in matches:
        # pick the box from whichever network has lower uncertainty
        joint_boxes.append(b1 if U1[b1] < U2[b2] else b2)

    # Unmatched boxes: only one network detected it — no cross-validation available.
    # Conservative default: require a STRICTER confidence/uncertainty threshold to admit these,
    # since there's no second-network agreement backing them up.
    for b in unmatched1 + unmatched2:
        if uncertainty(b) < strict_threshold:  # e.g. 0.5x the normal per-batch threshold
            joint_boxes.append(b)

    return joint_boxes
```

Then apply the same per-image dynamic filtering (§3.3) to `joint_boxes` to get `B_j_uc`.

### Full cross detection loss (weak-strong)
```
L_det_cross(x_u, x_u_s) =
      L_det(f1(x_u),   B2_uc)  +  L_det(f2(x_u),   B1_uc)     # uncertainty-guided cross-supervision
    + L_det(f1(x_u_s), Bj_uc)  +  L_det(f2(x_u_s), Bj_uc)     # joint pseudo-label supervision
```

---

## 5. Component 4 — Multi-Level Mutual Learning (detection version)

Applied only to labeled data `x_l`.

| Level | What to align | Loss |
|---|---|---|
| Backbone/neck features | `f1_backbone`, `f2_backbone` (e.g., FPN output feature maps) | SSIM or feature-map MSE |
| Detection head — classification logits | Per-anchor/per-cell class logits | Symmetric KL (only on matched anchor positions across the two networks) |
| Detection head — box regression | DFL distribution or regressed box params, on **matched anchors only** | MSE on matched predictions |

```
L_m(x_l) = L_ssim(f1_backbone, f2_backbone)
         + 0.5 * [ KL(p1_cls || p2_cls) + KL(p2_cls || p1_cls) ]   # over matched anchor positions
         + 2 * L_mse(box1_matched, box2_matched)
```

**Key difference from segmentation:** pixel-level mutual learning aligns every spatial location automatically since both networks produce the same H×W grid. For detection, the classification/box alignment can only be applied where **both networks produce an anchor/cell prediction for the same ground-truth object** on labeled data (since you have `y_l` here, use GT-anchor assignment, not prediction-to-prediction matching, to know which anchors correspond across the two networks).

---

## 6. Total Objective

```
L(x_l) = L_s(x_l) + 0.5 * L_det_cross(x_l) + 0.5 * L_m(x_l)
L(x_u) = 0.5 * L_det_cross(x_u, x_u_s)
```

Same weighting scheme as the segmentation paper — start here, then tune based on your validation set (the 0.5/0.5 split was tuned for their segmentation task, not guaranteed optimal for detection).

---

## 7. Component 5 — Spatiotemporal (ST) Correction — Detection Version

This maps naturally onto **tracking**, and you already have ByteTrack integration experience from the polyp project — reuse it directly.

Instead of pixel-count `R_n`, use:
- `N_n` = number of detected objects in frame n
- Track continuity from ByteTrack: does a track exist in frame n-1 and n+1 but not n (missed detection → FN), or does a track appear for exactly one frame in isolation (spurious → FP)?

```python
# FP frame: a track/detection exists in frame n but has no continuation in n-1 or n+1
is_FP = (track_id_in(n) is not None) and (track_id_in(n) not in tracks(n-1)) and (track_id_in(n) not in tracks(n+1))

# FN frame: a track exists in both n-1 and n+1 but has no matching detection in n
is_FN = (track_id in tracks(n-1)) and (track_id in tracks(n+1)) and (track_id not in tracks(n))
```

- Correction for FN frames: **interpolate the missing box** from `n-1` and `n+1` (linear interpolation of box center/size, or use the tracker's own Kalman-filter predicted state — ByteTrack already does this internally, so you may get this correction closer to "for free" than the segmentation paper's separate `f_st` network).
- Correction for FP frames: simply suppress the isolated detection.
- This is arguably a **cleaner and cheaper** solution than training a separate `f_st` correction network from scratch, since a tracker already does most of the temporal-consistency reasoning for detection tasks. Consider this as your primary path rather than replicating the paper's dedicated ST network 1:1.

---

## 8. Practical Notes for Your Thyroid Nodule / Polyp Projects

- **Nodule size stratification (from Endo-SemiS Table 4):** the paper found SSL gains were largest for large lesions and that small lesions remained the hardest — worth stratifying your own validation by nodule size the same way, given your existing size-stratified analysis work.
- **Multi-nodule class imbalance / Copy-Paste augmentation:** the aleatoric branch (§3.1) needs to be compatible with your existing Copy-Paste augmentation — order of operations matters: apply Copy-Paste as part of "weak" augmentation (it changes object placement/count) rather than folding it into the strong/intensity branch, so pseudo-label boxes still correspond to weak-view geometry before intensity perturbation.
- **SAHI sliced inference:** if you're using SAHI, uncertainty and NMS must be computed **after** slice merging, not per-slice — otherwise the per-image dynamic threshold (§3.3) will be computed on an incomplete, tile-local box population and can misbehave badly on edge-of-tile detections.

---

## 9. Suggested Implementation Order (detection-specific)

1. Two independent YOLO26 models + plain cross-supervision (box pseudo-labels, no uncertainty) → validate against your existing supervised YOLO26 baseline
2. Extract DFL-based box uncertainty (free, no extra passes) + MC-Dropout classification uncertainty (K=5)
3. Implement per-batch (not per-single-image) dynamic thresholding — test stability on your dataset's typical detections-per-image count first
4. Implement IoU-based box matching for joint pseudo-labels (§4) — this is the most bug-prone part; unit test the matching function in isolation before wiring into the loss
5. Add mutual learning on labeled data (GT-anchor-based matching, not prediction matching)
6. Combine into total objective, tune weighting on validation set
7. Wire in ByteTrack-based temporal correction (§7) using your existing ByteTrack integration
8. Evaluate at 10% label ratio first to match the paper's headline comparison point, then sweep other ratios
