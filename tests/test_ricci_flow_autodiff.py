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
