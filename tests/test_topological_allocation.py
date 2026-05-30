import numpy as np

from src.core.topological_allocation import (
    betti_numbers_at,
    correlation_distance_matrix,
    optimize_topological_risk_parity,
    topological_risk_parity_report,
    vietoris_rips_h0,
)


def _synthetic_returns(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 400
    # two correlated blocks + one independent asset
    block = rng.normal(size=(t, 1))
    a = block + 0.1 * rng.normal(size=(t, 1))
    b = block + 0.1 * rng.normal(size=(t, 1))
    c = rng.normal(size=(t, 1))
    return np.hstack([a, b, c])


def test_weights_are_simplex():
    returns = _synthetic_returns()
    w = optimize_topological_risk_parity(returns, max_homology_dimension=2)
    assert w.shape == (3,)
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)


def test_h0_persistence_pairs_count():
    returns = _synthetic_returns()
    dist = correlation_distance_matrix(returns)
    pairs, edges = vietoris_rips_h0(dist)
    finite = [p for p in pairs if np.isfinite(p.death)]
    assert len(finite) == dist.shape[0] - 1
    assert edges.shape[0] == dist.shape[0] - 1


def test_betti_zero_collapses_to_one_at_large_radius():
    returns = _synthetic_returns()
    dist = correlation_distance_matrix(returns)
    betti = betti_numbers_at(dist, radius=dist.max() + 1.0, max_dim=2)
    assert betti[0] == 1


def test_report_contains_betti():
    returns = _synthetic_returns()
    report = topological_risk_parity_report(returns)
    assert 0 in report.betti_numbers
    assert report.filtration_radius >= 0.0
