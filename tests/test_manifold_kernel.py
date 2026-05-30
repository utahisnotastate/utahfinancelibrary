import numpy as np

from src.models.manifold_kernel import (
    compute_ricci_flow_covariance,
    ricci_curvature_proxy,
    ricci_flow_curvature_field,
)


def _noisy_cov(seed: int = 0, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, n))
    return np.cov(x, rowvar=False)


def test_ricci_flow_preserves_spd_and_trace():
    cov = _noisy_cov()
    flowed = compute_ricci_flow_covariance(cov, flow_duration=0.5, manifold_dimension=6)
    assert flowed.shape == cov.shape
    eig = np.linalg.eigvalsh(flowed)
    assert np.all(eig > 0)
    assert np.isclose(np.trace(flowed), np.trace(cov), rtol=1e-3)


def test_ricci_flow_reduces_eigenvalue_spread():
    cov = _noisy_cov()
    before = np.std(np.log(np.linalg.eigvalsh(cov)))
    flowed = compute_ricci_flow_covariance(cov, flow_duration=2.0, manifold_dimension=6)
    after = np.std(np.log(np.linalg.eigvalsh(flowed)))
    assert after <= before + 1e-9


def test_zero_duration_is_identity_like():
    cov = _noisy_cov()
    flowed = compute_ricci_flow_covariance(cov, flow_duration=0.0, manifold_dimension=6)
    assert np.allclose(flowed, 0.5 * (cov + cov.T), atol=1e-8)


def test_curvature_field_shapes():
    cov = _noisy_cov()
    times, scalars = ricci_flow_curvature_field(cov, flow_duration=1.0, manifold_dimension=6, samples=5)
    assert times.shape == (5,)
    assert scalars.shape == (5,)
