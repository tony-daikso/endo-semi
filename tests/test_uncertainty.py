"""§3.2 DFL edge entropy, MC-Dropout cls uncertainty, and combination."""
import numpy as np

from endosemi.uncertainty.dfl_entropy import dfl_edge_entropy, box_dfl_uncertainty
from endosemi.uncertainty.mc_dropout import cls_mc_uncertainty
from endosemi.uncertainty.combine import combine_box_uncertainty


def test_peaked_distribution_low_entropy():
    peaked = np.array([0.98, 0.01, 0.005, 0.005])
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    assert dfl_edge_entropy(peaked) < dfl_edge_entropy(uniform)


def test_uniform_matches_log_n():
    uniform = np.full(8, 1 / 8)
    assert np.isclose(dfl_edge_entropy(uniform), np.log(8), atol=1e-6)


def test_box_dfl_uncertainty_mean_over_edges():
    # 1 box, 4 edges, 4 bins: two peaked edges, two uniform edges
    peaked = [0.97, 0.01, 0.01, 0.01]
    uniform = [0.25, 0.25, 0.25, 0.25]
    box = np.array([[peaked, uniform, peaked, uniform]])  # (1,4,4)
    u = box_dfl_uncertainty(box)
    assert u.shape == (1,)
    expected = np.mean([dfl_edge_entropy(np.array(peaked)),
                        dfl_edge_entropy(np.array(uniform))] * 2)
    assert np.isclose(u[0], expected)


def test_cls_mc_uncertainty_shapes():
    # 3 boxes, K=5 passes, 2 classes
    samples = np.random.default_rng(0).dirichlet([1, 1], size=(3, 5))
    P, U = cls_mc_uncertainty(samples)
    assert P.shape == (3, 2)
    assert U.shape == (3,)


def test_combine_alpha_endpoints():
    u_cls = np.array([1.0, 2.0])
    u_dfl = np.array([3.0, 4.0])
    np.testing.assert_allclose(combine_box_uncertainty(u_cls, u_dfl, alpha=1.0), u_cls)
    np.testing.assert_allclose(combine_box_uncertainty(u_cls, u_dfl, alpha=0.0), u_dfl)
    np.testing.assert_allclose(
        combine_box_uncertainty(u_cls, u_dfl, alpha=0.5), (u_cls + u_dfl) / 2)
