"""§3.3 dynamic thresholding."""
import numpy as np

from endosemi.uncertainty.thresholding import base_threshold, DynamicThreshold


def test_base_threshold_formula():
    u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    T = base_threshold(u)
    mu, sigma = u.mean(), u.std()
    p95 = np.percentile(u, 95)
    assert np.isclose(T, min(mu + sigma, p95))


def test_empty_population_admits_nothing():
    assert base_threshold(np.array([])) == float("inf")


def test_batch_mode_uses_batch_population():
    dt = DynamicThreshold(mode="batch")
    batch = np.array([0.1, 0.9, 0.2, 0.8])
    T = dt.compute(np.array([0.1]), batch_uncertainties=batch)
    assert np.isclose(T, base_threshold(batch))


def test_window_mode_accumulates():
    dt = DynamicThreshold(mode="window", window_size=100)
    assert dt.compute(np.array([0.5])) == float("inf")   # empty window
    dt.update(np.array([0.1, 0.2, 0.3]))
    dt.update(np.array([0.4, 0.5]))
    T = dt.compute(np.array([0.5]))
    assert np.isclose(T, base_threshold(np.array([0.1, 0.2, 0.3, 0.4, 0.5])))


def test_keep_mask():
    u = np.array([0.1, 0.9, 0.5])
    mask = DynamicThreshold.keep_mask(u, T=0.6)
    assert mask.tolist() == [True, False, True]
