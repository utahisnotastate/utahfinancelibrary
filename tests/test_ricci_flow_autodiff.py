import numpy as np
import pytest

from src.models.riemannian_geometry import is_jax_available
from src.models.manifold_kernel import compute_ricci_flow_covariance

JAX = is_jax_available()


def test_require_jax_flag_enforced_when_missing():
    # When JAX is present this should pass through; when absent it must raise.
    cov = np.cov(np.random.default_rng(0).normal(size=(100, 4)), rowvar=False)
    if JAX:
        out = compute_ricci_flow_covariance(cov, 0.5, 4, require_jax=True)
        assert out.shape == cov.shape
    else:
        with pytest.raises(RuntimeError):
            compute_ricci_flow_covariance(cov, 0.5, 4, require_jax=True)


@pytest.mark.skipif(not JAX, reason="JAX required for autodiff metric-field flow")
def test_api_does_not_call_numpy_proxy():
    """The denoising API must route through exact autodiff, never the proxy."""
    import src.models.manifold_kernel as mk

    cov = np.cov(np.random.default_rng(1).normal(size=(150, 5)), rowvar=False)

    def _boom(_metric):
        raise AssertionError("ricci_curvature_proxy must not be in the flow path")

    original = mk.ricci_curvature_proxy
    mk.ricci_curvature_proxy = _boom
    try:
        out = mk.compute_ricci_flow_covariance(cov, 1.0, 5)
    finally:
        mk.ricci_curvature_proxy = original
    assert out.shape == cov.shape


@pytest.mark.skipif(not JAX, reason="JAX required for autodiff metric-field flow")
def test_autodiff_flow_denoises_strong_anisotropy():
    """A spiked covariance is contracted toward its bulk (eigenvalue spread down)."""
    rng = np.random.default_rng(7)
    n = 5
    spectrum = np.array([4.0, 1.2, 1.0, 0.9, 0.8]) + rng.normal(scale=0.2, size=n)
    spectrum = np.clip(spectrum, 0.2, None)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    cov = (q * spectrum) @ q.T

    before = np.std(np.log(np.linalg.eigvalsh(cov)))
    flowed = compute_ricci_flow_covariance(cov, flow_duration=2.0, manifold_dimension=n)
    after = np.std(np.log(np.linalg.eigvalsh(flowed)))
    assert after < before  # strict denoising
    assert np.isclose(np.trace(flowed), np.trace(cov), rtol=1e-6)


@pytest.mark.skipif(not JAX, reason="JAX required for autodiff metric-field flow")
def test_normalized_ricci_flow_uniformizes_curvature():
    import jax.numpy as jnp
    from src.models.manifold_kernel import (
        gaussian_conformal_metric_family,
        ricci_flow_metric_field,
    )

    fam, _ = gaussian_conformal_metric_family(np.eye(2))
    xs = jnp.array([[0.3, 0.2], [0.1, -0.3], [-0.2, 0.25], [0.25, 0.3], [-0.15, -0.1]])
    # single-mode perturbation in the linear (stable) regime
    theta0 = np.array([0.0, 0.0, 0.0, 0.12, 0.0, 0.0])
    _, history = ricci_flow_metric_field(
        fam, theta0, xs, flow_duration=1.5, n_steps=25, max_step=0.02
    )
    assert history[-1] <= history[0] + 1e-9
