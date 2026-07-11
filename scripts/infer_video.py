#!/usr/bin/env python3
"""Video inference + ByteTrack spatiotemporal correction (§7).

    python scripts/infer_video.py --weights runs/.../best.pt --source data/unlabel-video/

For each video: detect per frame -> ByteTrack -> suppress isolated FPs and
interpolate FN gaps -> write corrected boxes / annotated video.

Can also be used offline to clean video-derived pseudo-labels before feeding
them back into training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from endosemi.config import Cfg
from endosemi.models import YOLO26Wrapper
from endosemi.temporal import SpatioTemporalCorrector


def iter_videos(source: str):
    p = Path(source)
    if p.is_dir():
        yield from sorted(p.glob("*.MOV")) + sorted(p.glob("*.mp4"))
    else:
        yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/endo_semi_yolo26.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file or directory")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    model = YOLO26Wrapper(cfg["model"]["arch"], weights=args.weights,
                          imgsz=cfg["model"]["imgsz"])
    corrector = SpatioTemporalCorrector(
        tracker_cfg=cfg["temporal"]["tracker"],
        fn_interpolation=cfg["temporal"]["fn_interpolation"],
        fp_suppress_isolated=cfg["temporal"]["fp_suppress_isolated"],
    )

    for video in iter_videos(args.source):
        print(f"[infer] {video}")
        # TODO(hook): decode frames; per-frame model.predict; collect DetOutputs;
        # corrected = corrector.run(dets); write annotated video / json.


if __name__ == "__main__":
    main()
