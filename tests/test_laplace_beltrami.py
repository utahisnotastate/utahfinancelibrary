import math

import pytest

from src.models.riemannian_geometry import is_jax_available

pytestmark = pytest.mark.skipif(not is_jax_available(), reason="JAX required for continuous generator")


def test_principal_eigenvalue_flat_unit_interval():
    import jax.numpy as jnp
    from src.core.risk_supervisor import principal_eigenvalue_laplace_beltrami

    g = lambda x: jnp.array(1.0)
    b = lambda x: jnp.array(0.0)
    lam0 = principal_eigenvalue_laplace_beltrami(g, b, (0.0, 1.0))
    # -1/2 d^2/dx^2 on [0,1], Dirichlet -> 0.5 * pi^2
    assert lam0 == pytest.approx(0.5 * math.pi**2, rel=1e-3)


def test_principal_eigenvalue_scales_with_domain():
    import jax.numpy as jnp
    from src.core.risk_supervisor import principal_eigenvalue_laplace_beltrami

    g = lambda x: jnp.array(1.0)
    b = lambda x: jnp.array(0.0)
    lam0 = principal_eigenvalue_laplace_beltrami(g, b, (0.0, 2.0))
    assert lam0 == pytest.approx(0.5 * (math.pi / 2) ** 2, rel=1e-3)


def test_feynman_kac_bound_monotone():
    from src.core.risk_supervisor import feynman_kac_drawdown_bound

    b_short = feynman_kac_drawdown_bound(2.0, 1.0)
    b_long = feynman_kac_drawdown_bound(2.0, 5.0)
    assert b_long < b_short  # longer horizon => tighter survival decay


def test_continuous_veto_logic():
    import jax.numpy as jnp
    from src.core.risk_supervisor import apply_continuous_spectral_cvar_veto

    g = lambda x: jnp.array(1.0)
    b = lambda x: jnp.array(0.0)
    # wide drawdown domain + short horizon => high exceedance bound => veto
    assert apply_continuous_spectral_cvar_veto(g, b, (0.0, 3.0), 0.99, 0.5) is True
    # tight domain + long horizon => safe
    assert apply_continuous_spectral_cvar_veto(g, b, (0.0, 0.5), 0.95, 5.0) is False
