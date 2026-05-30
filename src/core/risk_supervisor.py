"""
Eigenmanifold CVaR — Spectral risk supervision with a Symplectic Veto.

Rather than sorting discrete historical scenarios (classic Mean-CVaR), we lift
the loss distribution onto a function space and bound tail risk analytically via
the spectrum of a discretised loss/Laplacian operator under Dirichlet boundary
conditions.

If the spectral radius of the loss operator exceeds the confidence boundary,
``apply_spectral_cvar_veto`` returns ``True`` (execution should halt).

Backend: NumPy (JAX optional).
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple

import numpy as np

from src.core._backend import as_array

logger = logging.getLogger(__name__)


def _discretize_operator(
    loss_operator: Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray,
    dirichlet_boundary_conditions: np.ndarray,
) -> np.ndarray:
    r"""
    Build a symmetric operator matrix on ``grid`` combining:

    - a 1D Dirichlet Laplacian $-\partial_{xx}$ (diffusion of the loss density),
    - a diagonal potential $V(x)$ given by sampling ``loss_operator`` on the grid.

    The Dirichlet conditions zero the eigenfunctions at the domain boundary,
    encoding "loss cannot escape past the confidence wall".
    """
    grid = as_array(grid).ravel()
    n = grid.shape[0]

    # Scale-stable (graph) Dirichlet Laplacian of the path: eigenvalues in [0, 4],
    # independent of grid spacing, so the loss potential governs the spectral wall
    # rather than discretisation artefacts.
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    potential = as_array(loss_operator(grid)).ravel()
    if potential.shape[0] != n:
        raise ValueError("loss_operator must map the grid to a same-length array")

    op = lap + np.diag(potential)

    bc = as_array(dirichlet_boundary_conditions).ravel()
    if bc.size >= 1:
        op[0, :] = 0.0
        op[0, 0] = float(bc[0]) if bc.size >= 1 else 1.0
    if bc.size >= 2:
        op[-1, :] = 0.0
        op[-1, -1] = float(bc[-1])
    return 0.5 * (op + op.T)


def spectral_radius(
    loss_operator: Callable[[np.ndarray], np.ndarray],
    grid: Optional[np.ndarray] = None,
    dirichlet_boundary_conditions: Optional[np.ndarray] = None,
) -> float:
    """Largest-magnitude eigenvalue of the discretised loss operator."""
    if grid is None:
        grid = np.linspace(-1.0, 1.0, 128)
    if dirichlet_boundary_conditions is None:
        dirichlet_boundary_conditions = np.array([0.0, 0.0])
    op = _discretize_operator(loss_operator, grid, dirichlet_boundary_conditions)
    eigvals = np.linalg.eigvalsh(op)
    return float(np.max(np.abs(eigvals)))


def _confidence_boundary(confidence_level: float, grid: np.ndarray) -> float:
    r"""
    Map a confidence level $\alpha \in (0,1)$ to a spectral wall.

    Uses the inverse-Gaussian tail quantile scaled by the domain size; a higher
    confidence (e.g. 0.99) sets a stricter (larger) admissible spectral radius
    boundary before veto.
    """
    alpha = min(max(confidence_level, 1e-6), 1 - 1e-6)
    # standard normal quantile via erfinv
    z = math.sqrt(2.0) * _erfinv(2.0 * alpha - 1.0)
    span = float(np.max(grid) - np.min(grid)) or 1.0
    base = (math.pi / span) ** 2  # ground-state Dirichlet Laplacian eigenvalue
    return base * (1.0 + abs(z))


def _erfinv(x: float) -> float:
    # Winitzki approximation (sufficient for confidence mapping)
    a = 0.147
    ln = math.log(1 - x * x) if abs(x) < 1 else -1e6
    term = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)


def apply_spectral_cvar_veto(
    loss_operator: Callable[[np.ndarray], np.ndarray],
    confidence_level: float,
    dirichlet_boundary_conditions,
    grid: Optional[np.ndarray] = None,
) -> bool:
    """
    Compute the maximum eigenvalue of the loss operator. If the spectral radius
    exceeds the confidence boundary, trigger the Symplectic Veto (return True).

    Parameters
    ----------
    loss_operator : callable
        Maps a grid array -> loss-density values (the potential $V(x)$).
    confidence_level : float
        Tail confidence, e.g. 0.95 / 0.99.
    dirichlet_boundary_conditions : array-like
        Boundary values [left, right] pinning the eigenfunctions.
    grid : np.ndarray, optional
        Evaluation grid; defaults to 128 points on [-1, 1].

    Returns
    -------
    bool
        ``True`` => veto execution (risk wall breached); ``False`` => admissible.
    """
    if grid is None:
        grid = np.linspace(-1.0, 1.0, 128)
    bc = as_array(dirichlet_boundary_conditions)
    radius = spectral_radius(loss_operator, grid, bc)
    boundary = _confidence_boundary(confidence_level, as_array(grid))
    veto = radius > boundary
    logger.info(
        "[SPECTRAL-CVAR] radius=%.4f boundary=%.4f veto=%s",
        radius,
        boundary,
        veto,
    )
    return bool(veto)


def spectral_cvar_diagnostics(
    loss_operator: Callable[[np.ndarray], np.ndarray],
    confidence_level: float,
    dirichlet_boundary_conditions,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, bool]:
    """Return (spectral_radius, confidence_boundary, veto) for inspection."""
    if grid is None:
        grid = np.linspace(-1.0, 1.0, 128)
    bc = as_array(dirichlet_boundary_conditions)
    radius = spectral_radius(loss_operator, grid, bc)
    boundary = _confidence_boundary(confidence_level, as_array(grid))
    return radius, boundary, bool(radius > boundary)


# --------------------------------------------------------------------------- #
# Continuous Laplace-Beltrami principal eigenvalue (Feynman-Kac drawdown bound)
# --------------------------------------------------------------------------- #
# JAX is mandatory for the continuous path: derivatives of the test functions
# and of the metric are taken analytically (machine precision), and the
# eigenproblem is solved by a *spectral* Galerkin projection onto a smooth basis
# that exactly satisfies the Dirichlet conditions — exponential (not O(h^2))
# convergence, unlike finite-difference matrices.
try:
    import jax
    from jax import config as _jax_config

    _jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _JAX_OK = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _JAX_OK = False


def _require_jax_continuous() -> None:
    if not _JAX_OK:
        raise RuntimeError(
            "The continuous Laplace-Beltrami drawdown bound requires JAX "
            "(exact autodiff of the generator). Install: pip install -e '.[jax]'."
        )


def principal_eigenvalue_laplace_beltrami(
    metric_fn: Callable,
    drift_fn: Callable,
    domain: Tuple[float, float],
    n_basis: int = 24,
    n_quad: int = 400,
) -> float:
    r"""
    Smallest Dirichlet eigenvalue $\lambda_0$ of $-\mathcal{L}$ on a 1D domain,

        $$\mathcal{L} = \tfrac12 \Delta_M + b(x)\cdot\nabla,$$

    where $\Delta_M$ is the Laplace-Beltrami operator of the (scalar) metric
    ``metric_fn(x) = g(x) > 0`` and ``drift_fn(x) = b(x)``.

    Method: Galerkin projection onto the sine basis
    $\varphi_k(x) = \sin\!\big(k\pi (x-a)/(b-a)\big)$ (vanishing on $\partial\Omega$,
    so Dirichlet conditions are satisfied exactly), with operator actions taken
    by JAX autodiff and inner products under the Riemannian measure
    $dV = \sqrt{g}\,dx$. Returns $\lambda_0 > 0$.
    """
    _require_jax_continuous()
    a, b = float(domain[0]), float(domain[1])
    L = b - a
    xs = jnp.linspace(a, b, n_quad + 2)[1:-1]
    w = (b - a) / (n_quad + 1)  # uniform quadrature weight

    ks = jnp.arange(1, n_basis + 1)

    def phi(k, x):
        return jnp.sin(k * jnp.pi * (x - a) / L)

    def laplace_beltrami(fn, x):
        # (1/sqrt(g)) d/dx( sqrt(g) * (1/g) * d fn/dx )
        def inner(y):
            g = metric_fn(y)
            dfn = jax.grad(fn)(y)
            return jnp.sqrt(g) * (1.0 / g) * dfn

        g = metric_fn(x)
        return (1.0 / jnp.sqrt(g)) * jax.grad(inner)(x)

    def generator(fn, x):
        return 0.5 * laplace_beltrami(fn, x) + drift_fn(x) * jax.grad(fn)(x)

    sqrt_g = jnp.sqrt(jax.vmap(metric_fn)(xs))  # measure

    # build basis value & -L action tables on the quad grid
    def phi_vec(x):
        return jax.vmap(lambda k: phi(k, x))(ks)

    Phi = jax.vmap(phi_vec)(xs)  # (n_quad, n_basis)

    def negL_phi_at(x):
        return jax.vmap(lambda k: -generator(lambda y: phi(k, y), x))(ks)

    NegLPhi = jax.vmap(negL_phi_at)(xs)  # (n_quad, n_basis)

    measure = (w * sqrt_g)[:, None]  # (n_quad,1)
    A = (Phi * measure).T @ NegLPhi  # stiffness <phi_k, -L phi_j>
    M = (Phi * measure).T @ Phi  # mass <phi_k, phi_j>

    A = 0.5 * (A + A.T)  # symmetrise (self-adjoint under dV when b=0)
    M = 0.5 * (M + M.T)

    # generalized eigenproblem A v = lambda M v
    M_inv = jnp.linalg.inv(M)
    eig = jnp.linalg.eigvals(M_inv @ A)
    eig_real = jnp.real(eig)
    lam0 = float(jnp.min(eig_real))
    return lam0


def feynman_kac_drawdown_bound(
    lambda0: float,
    horizon: float,
    eigenfunction_integral: float = 1.0,
) -> float:
    r"""
    Spectral (Feynman-Kac) upper bound on the drawdown-exceedance probability:

        $$\mathbb{P}\!\left(\sup_{0\le t\le T}\text{Drawdown}_t > \mathcal{D}_{max}\right)
            \le C\, e^{-\lambda_0 T}.$$

    ``eigenfunction_integral`` is the constant $C = \int_\Omega \phi_0\,dx$.
    """
    return float(eigenfunction_integral * math.exp(-lambda0 * horizon))


def drawdown_veto_from_tick_metric(
    covariation_matrix,
    weights,
    drawdown_limit: float,
    confidence_level: float,
    horizon: float,
    drift: float = 0.0,
    n_basis: int = 24,
) -> bool:
    r"""
    Pathwise-measured drawdown veto.

    Builds the 1D portfolio diffusion metric directly from the observed
    quadratic covariation, $g = w^\top \Sigma w$ (an $\mathcal{F}_t$-measurable
    observable, not an estimated parameter), and applies the continuous
    Laplace-Beltrami veto on the drawdown domain $[0, \mathcal{D}_{max}]$.

    ``covariation_matrix`` is the metric rate from
    :class:`src.core.tick_observer.QuadraticCovariationObserver`.
    """
    _require_jax_continuous()
    from src.core.tick_observer import drawdown_metric_from_covariation

    g_scalar = drawdown_metric_from_covariation(covariation_matrix, weights)
    g_scalar = max(g_scalar, 1e-12)

    metric_fn = lambda x: jnp.array(g_scalar)
    drift_fn = lambda x: jnp.array(float(drift))
    return apply_continuous_spectral_cvar_veto(
        metric_fn,
        drift_fn,
        (0.0, float(drawdown_limit)),
        confidence_level,
        horizon,
        n_basis=n_basis,
    )


def apply_continuous_spectral_cvar_veto(
    metric_fn: Callable,
    drift_fn: Callable,
    domain: Tuple[float, float],
    confidence_level: float,
    horizon: float,
    eigenfunction_integral: float = 1.0,
    n_basis: int = 24,
) -> bool:
    r"""
    Continuous Laplace-Beltrami veto. Computes the principal eigenvalue
    $\lambda_0$ of the portfolio generator on the drawdown domain and the
    Feynman-Kac bound $C e^{-\lambda_0 T}$. Vetoes (returns ``True``) when the
    bounded exceedance probability is greater than the tolerance
    $1 - \text{confidence\_level}$.

    This is an analytic supremum on the exit probability, not a histogram of past
    scenarios.
    """
    _require_jax_continuous()
    lam0 = principal_eigenvalue_laplace_beltrami(metric_fn, drift_fn, domain, n_basis=n_basis)
    bound = feynman_kac_drawdown_bound(lam0, horizon, eigenfunction_integral)
    tolerance = 1.0 - confidence_level
    veto = bound > tolerance
    logger.info(
        "[LAPLACE-BELTRAMI] lambda0=%.6f bound=%.6e tol=%.6e veto=%s",
        lam0,
        bound,
        tolerance,
        veto,
    )
    return bool(veto)
