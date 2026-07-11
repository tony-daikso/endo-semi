"""Component 5 — spatiotemporal correction via ByteTrack (§7).

The spec recommends reusing tracking rather than training a separate ST network:
a tracker already does the temporal-consistency reasoning.

Rules (per track, across frames n-1, n, n+1):
  FP frame: a detection/track exists in frame n but has NO continuation in
            n-1 AND n+1  -> isolated spurious -> SUPPRESS.
  FN frame: a track exists in BOTH n-1 and n+1 but has NO matching detection in
            n            -> missed detection -> INTERPOLATE the box.

Interpolation:
  - 'linear': linear interp of box center/size between n-1 and n+1.
  - 'kalman': use ByteTrack's own Kalman-predicted state for frame n (closer to
    "free", since ByteTrack maintains this internally).

SAHI note (§8): if sliced inference is used, run uncertainty + NMS AFTER slice
merging — never per-slice — or the per-image dynamic threshold misbehaves on
edge-of-tile detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def flag_fp_fn(tracks_prev: set[int], tracks_cur: set[int], tracks_next: set[int]):
    """Classify track ids at frame n given neighbor-frame track id sets.

    Returns (fp_ids, fn_ids):
      fp_ids: in cur but not in prev and not in next  (isolated -> suppress)
      fn_ids: in prev and next but not in cur          (gap -> interpolate)
    """
    fp_ids = {t for t in tracks_cur if t not in tracks_prev and t not in tracks_next}
    fn_ids = {t for t in (tracks_prev & tracks_next) if t not in tracks_cur}
    return fp_ids, fn_ids


def interpolate_box(box_prev_xyxy, box_next_xyxy, method: str = "linear"):
    """Estimate the frame-n box for an FN gap. `linear` = midpoint of center/size.

    (Kalman path is handled inside the corrector, which has tracker state.)
    """
    a = np.asarray(box_prev_xyxy, dtype=float).ravel()
    b = np.asarray(box_next_xyxy, dtype=float).ravel()
    if method != "linear":
        raise ValueError("use SpatioTemporalCorrector for kalman interpolation")
    return (a + b) / 2.0


@dataclass
class SpatioTemporalCorrector:
    """Runs ByteTrack over a video's per-frame detections and applies FP/FN
    corrections. Used at inference (§7); optionally as an offline pseudo-label
    cleaner before feeding video pseudo-labels back into training."""

    tracker_cfg: str = "bytetrack.yaml"
    fn_interpolation: str = "kalman"
    fp_suppress_isolated: bool = True
    _tracker: object = field(default=None, repr=False)

    def __post_init__(self):
        # TODO(hook): instantiate ultralytics' ByteTrack from self.tracker_cfg.
        pass

    def run(self, per_frame_detections: list) -> list:
        """
        per_frame_detections: list over frames of DetOutput.
        returns: corrected list over frames (FPs suppressed, FN boxes inserted).
        """
        # TODO(hook):
        #   1. feed each frame's boxes to ByteTrack -> per-frame track ids.
        #   2. for each frame n, call flag_fp_fn on (n-1, n, n+1) track sets.
        #   3. suppress fp_ids; for fn_ids, insert interpolate_box (linear) or
        #      the tracker's Kalman-predicted state (fn_interpolation='kalman').
        raise NotImplementedError
