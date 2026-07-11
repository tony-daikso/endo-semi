"""Endo-SemiS → YOLO26 semi-supervised polyp detection.

Package layout mirrors the adaptation spec
(instruction/Endo-SemiS_YOLO26_detection_adaptation.md):

    data/         §3.1 labeled/unlabeled datasets, weak/strong augmentation
    models/       §1   dual YOLO26 + MC-Dropout head
    uncertainty/  §3.2 DFL entropy + MC-Dropout, §3.3 dynamic thresholding
    matching/     §4   IoU / Hungarian box matching
    losses/       §2,4,5,6 cross-supervision, joint pseudo, mutual learning, total
    temporal/     §7   ByteTrack spatiotemporal correction
    engine/            training loop + size-stratified evaluation
    utils/             box ops, logging
"""

__version__ = "0.1.0"
