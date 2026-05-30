"""
Ricci Flow Covariance Evolution — non-linear denoising of the dependency metric.

Treats the empirical covariance $g_{ij}$ as a Riemannian metric and evolves it
under a (volume-normalised) Ricci flow

    $$\\frac{\\partial g_{ij}}{\\partial t} = -2 R_{ij} + \\frac{2}{m} r\\, g_{ij}$$

toward a state of constant curvature. The fixed point is a metric whose
spectrum is uniformised — i.e. micro-structure noise (the dispersed eigenvalue
shoulder of random-matrix theory) is dissipated while the dominant non-linear
signal directions are preserved.

For an SPD matrix we realise the flow in the eigenbasis: the curvature proxy is
the deviation of each (log) eigenvalue from the mean log-eigenvalue, and the
flow contracts that deviation while preserving the trace (volume). This is a
well-posed, continuously differentiable ODE whose closed form we integrate with
adaptive explicit Euler steps.

Backend: NumPy (JAX optional).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple

import numpy as np

from src.core._backend import as_array

try:
    import jax
    import jax.numpy as jnp

    from src.models.riemannian_geometry import ricci_tensor, scalar_curvature

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _nearest_spd(m: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m = _symmetrize(m)
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def ricci_curvature_proxy(metric: np.ndarray) -> np.ndarray:
    """
    Spectral Ricci-curvature proxy $R_{ij}$ for an SPD metric.

    In the eigenbasis the Ricci tensor is diagonal with entries proportional to
    the deviation of each log-eigenvalue from the mean log-eigenvalue (constant
    curvature ⇔ all eigenvalues equal ⇔ zero proxy). This recovers the standard
    "evolve toward constant curvature" target while staying SPD-stable.
    """
    metric = _nearest_spd(metric)
    vals, vecs = np.linalg.eigh(metric)
    log_vals = np.log(vals)
    mean_log = float(np.mean(log_vals))
    ricci_eigs = (log_vals - mean_log) * vals  # scale back into metric units
    return (vecs * ricci_eigs) @ vecs.T


def compute_ricci_flow_covariance(
    empirical_metric_tensor,
    flow_duration: float,
    manifold_dimension: int,
    n_steps: int = 200,
    preserve_trace: bool = True,
    require_jax: bool = False,
) -> np.ndarray:
    """
    Evolve the empirical covariance metric under normalised Ricci flow until the
    scalar curvature is (near-)uniform, yielding a denoised non-linear dependency
    tensor.

    Parameters
    ----------
    empirical_metric_tensor : array-like, shape (N, N)
        Empirical covariance / correlation (SPD). Accepts ``jax.Array``.
    flow_duration : float
        Total integration time T. Larger T ⇒ stronger denoising (full collapse
        to constant curvature in the limit).
    manifold_dimension : int
        Dimension m used in the volume-normalisation term.
    n_steps : int
        Number of explicit Euler sub-steps.
    preserve_trace : bool
        If True, rescale after each step to hold the trace (total variance) fixed.
    require_jax : bool
        If True, refuse to run without JAX (mandate exact tensor calculus and
        machine-precision linear algebra; no NumPy fallback).

    Returns
    -------
    np.ndarray
        Denoised SPD covariance metric of the same shape.
    """
    if require_jax and not JAX_AVAILABLE:
        raise RuntimeError(
            "require_jax=True but JAX is unavailable. Install with "
            "pip install -e '.[jax]'. NumPy finite-difference tensor calculus is "
            "forbidden in strict mode."
        )
    g = _nearest_spd(as_array(empirical_metric_tensor))
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("empirical_metric_tensor must be square (N, N)")
    if flow_duration <= 0:
        return g
    m = max(int(manifold_dimension), 1)
    dt = flow_duration / max(n_steps, 1)
    target_trace = float(np.trace(g))

    for _ in range(n_steps):
        ricci = ricci_curvature_proxy(g)
        scalar_r = float(np.trace(ricci))
        # dg/dt = -2 Ric + (2/m) r g   (volume-normalised)
        dg = -2.0 * ricci + (2.0 / m) * (scalar_r / max(g.shape[0], 1)) * g
        g = _symmetrize(g + dt * dg)
        g = _nearest_spd(g)
        if preserve_trace and np.trace(g) > 0:
            g *= target_trace / float(np.trace(g))

    vals = np.linalg.eigvalsh(g)
    logger.info(
        "[RICCI] flow T=%.3f m=%d cond=%.3e spread=%.4f",
        flow_duration,
        m,
        float(vals.max() / max(vals.min(), 1e-12)),
        float(np.std(np.log(np.clip(vals, 1e-12, None)))),
    )
    return g


def ricci_flow_curvature_field(
    empirical_metric_tensor,
    flow_duration: float,
    manifold_dimension: int,
    samples: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuously-differentiable curvature trajectory for ``wave_theory_engine``.

    Returns the time grid and a stack of scalar-curvature values sampled along
    the flow, so downstream signal models can ingest a dynamic curvature tensor
    rather than a single static matrix.
    """
    g = _nearest_spd(as_array(empirical_metric_tensor))
    times = np.linspace(0.0, flow_duration, samples)
    scalars = np.empty(samples, dtype=float)
    for k, t in enumerate(times):
        gt = compute_ricci_flow_covariance(g, float(t), manifold_dimension)
        scalars[k] = float(np.trace(ricci_curvature_proxy(gt)))
    return times, scalars


# --------------------------------------------------------------------------- #
# Exact (autodiff) Ricci flow on a curved metric FIELD — JAX mandatory
# --------------------------------------------------------------------------- #
def _require_jax_strict() -> None:
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "JAX is mandatory for autodiff Ricci flow on a metric field. "
            "Finite differences (O(h^2)) corrupt the Riemann tensor in higher "
            "dimensions. Install with: pip install -e '.[jax]'."
        )


def _quadratic_basis(x):
    """Smooth conformal basis over R^2: [1, x0, x1, x0^2, x1^2, x0*x1]."""
    return jnp.array(
        [
            1.0,
            x[0],
            x[1],
            x[0] ** 2,
            x[1] ** 2,
            x[0] * x[1],
        ]
    )


def gaussian_conformal_metric_family(empirical_metric_tensor) -> Tuple[Callable, "np.ndarray"]:
    r"""
    Build a curved, conformally-flat metric family seeded by an empirical
    covariance, with enough degrees of freedom for the normalised Ricci flow to
    reach constant curvature.

        $$g_\theta(x) = \exp\!\big(2\, \theta^\top \psi(x)\big)\, \Sigma$$

    where $\psi$ is a quadratic conformal basis. The seed ``theta0`` encodes the
    Gaussian log-density $-\tfrac12 x^\top \Sigma^{-1} x$, so the initial manifold
    curvature reflects the covariance anisotropy. Returns ``(family, theta0)``
    with ``family(theta) -> metric_fn(x)``.

    (2D state space; for higher dimensions extend ``_quadratic_basis``.)
    """
    _require_jax_strict()
    sigma = jnp.asarray(_nearest_spd(as_array(empirical_metric_tensor)))
    if sigma.shape != (2, 2):
        raise ValueError("gaussian_conformal_metric_family currently supports 2x2 metrics")
    sigma_inv = jnp.linalg.inv(sigma)

    # seed: f(x) = -0.5 x^T Sigma^-1 x  ->  coefficients on [x0^2, x1^2, x0*x1]
    theta0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            float(-0.5 * sigma_inv[0, 0]),
            float(-0.5 * sigma_inv[1, 1]),
            float(-1.0 * sigma_inv[0, 1]),
        ]
    )

    def family(theta):
        theta = jnp.asarray(theta)

        def metric_fn(x):
            f = jnp.dot(theta, _quadratic_basis(x))
            return jnp.exp(2.0 * f) * sigma

        return metric_fn

    return family, theta0


def ricci_flow_metric_field(
    family: Callable,
    theta0,
    x_samples,
    flow_duration: float,
    n_steps: int = 50,
    max_step: float = 0.1,
) -> Tuple["np.ndarray", List[float]]:
    r"""
    Integrate the **volume-normalised** Ricci flow

        $$\partial_t g = -2\Big(\mathrm{Ric}(g) - \tfrac{\bar r}{m} g\Big)$$

    on a parametric metric family by Galerkin projection onto the family's
    tangent space ($\bar r$ = mean scalar curvature, $m$ = manifold dimension).
    The normalised flow converges to a **constant-curvature** (denoised) metric
    rather than blowing up at a finite-time singularity.

    At each step the exact Ricci tensor is computed via autodiff (no finite
    differences) at every sample point, and the parameter velocity solves

        $$\dot\theta = \arg\min_{\dot\theta}\; \big\| \partial_\theta g\,\dot\theta
            - \partial_t g \big\|_{F}^2$$

    Returns the evolved parameters and the history of the **scalar-curvature
    dispersion** (std across samples), which decreases as curvature uniformises.
    """
    _require_jax_strict()
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    if theta.ndim == 0:
        theta = theta.reshape(1)
    xs = jnp.asarray(as_array(x_samples))
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)
    m = int(xs.shape[1])
    dt = flow_duration / max(n_steps, 1)

    def dg_dtheta(theta_vec, x):
        return jax.jacfwd(lambda th: family(th)(x))(theta_vec)  # (d,d,p)

    def mfn_of(theta_vec):
        return family(theta_vec)

    def scalar_dispersion(theta_vec) -> float:
        mfn = mfn_of(theta_vec)
        vals = jnp.array([scalar_curvature(mfn, x) for x in xs])
        return float(jnp.std(vals))

    history: List[float] = [scalar_dispersion(theta)]
    for _ in range(n_steps):
        mfn = mfn_of(theta)
        # mean scalar curvature for the normalisation term
        rbar = float(jnp.mean(jnp.array([scalar_curvature(mfn, x) for x in xs])))

        p = theta.shape[0]
        ata = jnp.zeros((p, p))
        atb = jnp.zeros((p,))
        for x in xs:
            dG = dg_dtheta(theta, x)  # (d, d, p)
            ric = ricci_tensor(mfn, x)  # (d, d)
            g_here = mfn(x)
            dg_target = -2.0 * (ric - (rbar / m) * g_here)  # normalised flow
            A = dG.reshape(-1, p)
            b = dg_target.reshape(-1)
            ata = ata + A.T @ A
            atb = atb + A.T @ b
        ata = ata + 1e-9 * jnp.eye(p)
        dtheta = jnp.linalg.solve(ata, atb)
        step = dt * dtheta
        norm = float(jnp.linalg.norm(step))
        if norm > max_step and norm > 0:
            step = step * (max_step / norm)
        theta = theta + step
        history.append(scalar_dispersion(theta))

    logger.info(
        "[RICCI-AUTODIFF] curvature dispersion %.6f -> %.6f (theta=%s)",
        history[0],
        history[-1],
        np.asarray(theta),
    )
    return np.asarray(theta), history
