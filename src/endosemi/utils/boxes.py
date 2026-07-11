"""Box geometry helpers shared across matching (§4), thresholding (§3.3) and
spatiotemporal correction (§7). Pure numpy/torch-agnostic where practical.

Box conventions
---------------
- ``xyxy``  : (x1, y1, x2, y2) absolute pixels.
- ``xywh``  : (cx, cy, w, h)   YOLO normalized [0,1] center form.
All matrices are shaped ``(N, 4)``.
"""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray, img_w: float = 1.0, img_h: float = 1.0) -> np.ndarray:
    """YOLO (cx,cy,w,h) normalized -> (x1,y1,x2,y2). Pass image size to denormalize."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywh(boxes: np.ndarray, img_w: float = 1.0, img_h: float = 1.0) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1)


def box_area(boxes_xyxy: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def box_iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU. Returns ``(len(a), len(b))``. Empty inputs -> empty matrix.

    Used by the IoU matcher (§4) and by ByteTrack-style association (§7).
    """
    a = np.asarray(a_xyxy, dtype=float).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)
