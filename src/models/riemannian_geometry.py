"""
JAX-exact Riemannian geometry via continuous autodifferentiation.

This module computes the Christoffel symbols, Riemann curvature tensor, Ricci
tensor and scalar curvature of an arbitrary metric field ``g: R^d -> SPD(d)``
using ``jax`` automatic differentiation. Metric derivatives ``∂_k g_{ij}`` are
obtained to machine precision — there are **no finite differences** and hence no
``O(h^2)`` truncation error that could fracture the non-linear Ricci flow.

JAX is mandatory here. If JAX is unavailable these functions raise; there is no
NumPy fallback for tensor calculus by design (see module ``manifold_kernel`` for
the rationale: discrete finite differences corrupt curvature in higher dims).

Index conventions
-----------------
- ``metric_fn(x)`` returns ``g`` with ``g[i, j]`` = $g_{ij}(x)$.
- ``christoffel_symbols`` returns ``Gamma`` with ``Gamma[k, i, j]`` = $\\Gamma^k_{ij}$.
- ``riemann_tensor`` returns ``R`` with ``R[l, i, j, k]`` = $R^l{}_{ijk}$.
- ``ricci_tensor`` returns ``Ric`` with ``Ric[j, k]`` = $R_{jk}$.
"""

from __future__ import annotations

from typing import Callable

try:
    import jax
    from jax import config as _jax_config

    _jax_config.update("jax_enable_x64", True)  # double precision for exact tensor calculus
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    JAX_AVAILABLE = False


MetricField = Callable[["jnp.ndarray"], "jnp.ndarray"]


def _require_jax() -> None:
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "JAX is mandatory for exact Riemannian tensor calculus. "
            "Install with: pip install -e '.[jax]'. NumPy finite differences "
            "are forbidden here (O(h^2) truncation fractures Ricci flow)."
        )


def metric_jacobian(metric_fn: MetricField, x: "jnp.ndarray") -> "jnp.ndarray":
    """Return ``J[i, j, k] = ∂ g_{ij} / ∂ x_k`` exactly via forward-mode autodiff."""
    _require_jax()
    return jax.jacfwd(metric_fn)(x)


def christoffel_symbols(metric_fn: MetricField, x: "jnp.ndarray") -> "jnp.ndarray":
    r"""
    Christoffel symbols of the second kind:

    $$\Gamma^k_{ij} = \tfrac12 g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il}
        - \partial_l g_{ij}\right)$$

    Returns ``Gamma[k, i, j]``.
    """
    _require_jax()
    g = metric_fn(x)
    g_inv = jnp.linalg.inv(g)
    dg = metric_jacobian(metric_fn, x)  # dg[i, j, k] = d g_ij / d x_k

    # partial_i g_jl -> term1[i, j, l] = dg[j, l, i]
    di_gjl = jnp.einsum("jli->ijl", dg)
    dj_gil = jnp.einsum("ilj->ijl", dg)
    dl_gij = jnp.einsum("ijl->ijl", dg)
    combo = di_gjl + dj_gil - dl_gij  # [i, j, l]

    # Gamma^k_ij = 1/2 g^{kl} combo[i, j, l]
    gamma = 0.5 * jnp.einsum("kl,ijl->kij", g_inv, combo)
    return gamma


def riemann_tensor(metric_fn: MetricField, x: "jnp.ndarray") -> "jnp.ndarray":
    r"""
    Riemann curvature tensor (1,3):

    $$R^l{}_{ijk} = \partial_i \Gamma^l_{jk} - \partial_j \Gamma^l_{ik}
        + \Gamma^l_{im}\Gamma^m_{jk} - \Gamma^l_{jm}\Gamma^m_{ik}$$

    Returns ``R[l, i, j, k]``.
    """
    _require_jax()

    def gamma_fn(y: "jnp.ndarray") -> "jnp.ndarray":
        return christoffel_symbols(metric_fn, y)

    gamma = gamma_fn(x)  # [k, i, j]
    dgamma = jax.jacfwd(gamma_fn)(x)  # dgamma[k, i, j, m] = d Gamma^k_ij / d x_m

    # partial_i Gamma^l_jk -> A[l, i, j, k] = dgamma[l, j, k, i]
    di_g = jnp.einsum("ljki->lijk", dgamma)
    dj_g = jnp.einsum("likj->lijk", dgamma)

    # Gamma^l_im Gamma^m_jk -> B[l, i, j, k]
    quad1 = jnp.einsum("lim,mjk->lijk", gamma, gamma)
    quad2 = jnp.einsum("ljm,mik->lijk", gamma, gamma)

    # Convention chosen so that round spheres have positive Ricci/scalar curvature.
    return -(di_g - dj_g + quad1 - quad2)


def ricci_tensor(metric_fn: MetricField, x: "jnp.ndarray") -> "jnp.ndarray":
    r"""
    Ricci tensor by contracting the first and third indices of Riemann:

    $$R_{jk} = R^i{}_{jik}$$

    Returns ``Ric[j, k]``.
    """
    _require_jax()
    riem = riemann_tensor(metric_fn, x)  # R[l, i, j, k]
    # contract l == j-index position: R^i_{jik} -> sum over i of R[i, j, i, k]
    ric = jnp.einsum("ijik->jk", riem)
    return ric


def scalar_curvature(metric_fn: MetricField, x: "jnp.ndarray") -> "jnp.ndarray":
    r"""Scalar curvature $R = g^{jk} R_{jk}$."""
    _require_jax()
    g_inv = jnp.linalg.inv(metric_fn(x))
    ric = ricci_tensor(metric_fn, x)
    return jnp.einsum("jk,jk->", g_inv, ric)


def is_jax_available() -> bool:
    return JAX_AVAILABLE
