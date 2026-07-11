"""DetectionEvaluator correctness (needs torch + ultralytics -> run in `polyp` env).

    PYTHONPATH=src python -m pytest tests/test_detection_metrics.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from endosemi.engine.evaluator import DetectionEvaluator
from endosemi.models.dual_yolo import DetOutput

IMGSZ = 640
_GTS = [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]


def _batch():
    return {
        "img": torch.zeros(2, 3, IMGSZ, IMGSZ),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(_GTS),
    }


def _xywhn_to_xyxy(g):
    cx, cy, w, h = g
    s = IMGSZ
    return [(cx - w / 2) * s, (cy - h / 2) * s, (cx + w / 2) * s, (cy + h / 2) * s]


class _PerfectNet:
    def predict(self, imgs, conf):
        return [DetOutput(np.array([_xywhn_to_xyxy(g)], dtype=np.float32),
                          np.array([0.9]), np.array([0.0])) for g in _GTS]


class _EmptyNet:
    def predict(self, imgs, conf):
        return [DetOutput(np.zeros((0, 4)), np.zeros(0), np.zeros(0)) for _ in imgs]


class _WrongLocationNet:
    def predict(self, imgs, conf):
        return [DetOutput(np.array([[0.0, 0.0, 20.0, 20.0]], dtype=np.float32),
                          np.array([0.9]), np.array([0.0])) for _ in imgs]


@pytest.fixture
def evaluator():
    return DetectionEvaluator(IMGSZ, "cpu", conf=0.001)


def test_perfect_predictions_score_one(evaluator):
    m = evaluator.evaluate(_PerfectNet(), [_batch()])
    assert m["map50"] > 0.99
    assert m["recall"] > 0.99
    assert m["precision"] > 0.99


def test_empty_predictions_score_zero(evaluator):
    m = evaluator.evaluate(_EmptyNet(), [_batch()])
    assert m["recall"] == 0.0
    assert m["map50"] == 0.0


def test_wrong_location_scores_zero(evaluator):
    m = evaluator.evaluate(_WrongLocationNet(), [_batch()])
    assert m["map50"] < 0.01
