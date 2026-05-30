import math

import pytest

from src.models.riemannian_geometry import is_jax_available

pytestmark = pytest.mark.skipif(not is_jax_available(), reason="JAX required for exact tensor calculus")


def test_sphere_scalar_curvature_exact():
    import jax.numpy as jnp
    from src.models.riemannian_geometry import scalar_curvature, ricci_tensor

    for r in (1.0, 2.0, 3.0):
        def g(x, r=r):
            theta = x[0]
            return jnp.array([[r**2, 0.0], [0.0, (r**2) * jnp.sin(theta) ** 2]])

        x = jnp.array([0.9, 0.2])
        R = float(scalar_curvature(g, x))
        assert R == pytest.approx(2.0 / r**2, abs=1e-5)
        Ric = ricci_tensor(g, x)
        assert bool(jnp.allclose(Ric, g(x) / r**2, atol=1e-5))


def test_flat_metric_zero_curvature():
    import jax.numpy as jnp
    from src.models.riemannian_geometry import scalar_curvature

    def g(x):
        return jnp.eye(2)

    assert float(scalar_curvature(g, jnp.array([0.3, -0.4]))) == pytest.approx(0.0, abs=1e-8)


def test_require_jax_message_present():
    # smoke: function exists and is importable
    from src.models.riemannian_geometry import christoffel_symbols

    assert callable(christoffel_symbols)
