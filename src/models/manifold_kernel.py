"""
Ricci Flow Covariance Evolution — non-linear denoising of the dependency metric.

Treats the empirical covariance $g_{ij}$ as a Riemannian metric and evolves it
under a (volume-normalised) Ricci flow

    $$\\frac{\\partial g_{ij}}{\\partial t} = -2 R_{ij} + \\frac{2}{m} r\\, g_{ij}$$

toward a state of constant curvature. The fixed point is a metric whose
spectrum is uniformised — i.e. micro-structure noise (the dispersed eigenvalue
shoulder of random-matrix theory) is dissipated while the dominant non-linear
signal directions are preserved.

The primary API ``compute_ricci_flow_covariance`` is a **direct bridge to the
exact autodiff metric-field flow** (``ricci_flow_metric_field``). It builds a
curved Riemannian metric *field* whose curvature is induced by the empirical
covariance (a constant matrix is flat — Ricci flow only acts once the metric
varies in space), integrates the volume-normalised Ricci flow with Christoffel
symbols / Ricci tensor obtained by ``jax.jacfwd`` to machine precision, and
contracts the spectrum by the *measured* curvature-uniformization ratio. There
is **no NumPy finite-difference proxy in the API path** — discrete `O(h^2)`
tensor calculus is forbidden because it corrupts the non-linear flow PDE.

Backend: JAX mandatory for the flow (exact tensor calculus).
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
    Spectral Ricci-curvature **diagnostic** (standalone; NOT used by the flow API).

    In the eigenbasis this analytic surrogate is diagonal with entries
    proportional to the deviation of each log-eigenvalue from the mean
    log-eigenvalue (constant curvature ⇔ all eigenvalues equal). It is retained
    only as a cheap closed-form diagnostic. The denoising API
    (``compute_ricci_flow_covariance``) does **not** call it — it integrates the
    exact autodiff metric-field flow instead.
    """
    metric = _nearest_spd(metric)
    vals, vecs = np.linalg.eigh(metric)
    log_vals = np.log(vals)
    mean_log = float(np.mean(log_vals))
    ricci_eigs = (log_vals - mean_log) * vals  # scale back into metric units
    return (vecs * ricci_eigs) @ vecs.T


def _anisotropic_conformal_family(eigenvalues) -> Tuple[Callable, "jnp.ndarray", int]:
    r"""
    Diagonal, conformally-curved metric field seeded by a covariance spectrum.

        $$g(x) = \operatorname{diag}_i \exp\!\big(2(a_i + p_i\, r(x))\big),
            \qquad r(x) = \tfrac12\lVert x\rVert^2,$$

    with $a_i = \tfrac12\log\lambda_i$ (so $g(0) = \operatorname{diag}(\lambda)$)
    and $p_i = +\tfrac1{2\lambda_i}$ (precision-seeded, so the induced curvature
    reflects the spectral anisotropy). The normalised Ricci flow on $(a, p)$ —
    integrated with exact ``jax.jacfwd`` curvature — drives this field toward
    constant curvature. Returns ``(family, theta0, N)``.
    """
    _require_jax_strict()
    lam = jnp.asarray(eigenvalues)
    n = int(lam.shape[0])
    a0 = 0.5 * jnp.log(lam)
    p0 = 0.5 / lam
    theta0 = jnp.concatenate([a0, p0])

    def family(theta):
        th = jnp.asarray(theta)
        a = th[:n]
        p = th[n:]

        def metric_fn(x):
            r = 0.5 * jnp.sum(x ** 2)
            return jnp.diag(jnp.exp(2.0 * (a + p * r)))

        return metric_fn

    return family, theta0, n


def _curvature_sample_cloud(n_dim: int, radius: float = 0.35, seed: int = 0) -> np.ndarray:
    """Deterministic small point cloud at which curvature is sampled for the flow."""
    rng = np.random.default_rng(seed)
    n_pts = max(5, n_dim)
    return rng.normal(scale=radius, size=(n_pts, n_dim))


def compute_ricci_flow_covariance(
    empirical_metric_tensor,
    flow_duration: float,
    manifold_dimension: int,
    n_steps: int = 30,
    preserve_trace: bool = True,
    require_jax: bool = True,
) -> np.ndarray:
    r"""
    Denoise a covariance by the **exact autodiff** normalised Ricci flow.

    This is a direct bridge to ``ricci_flow_metric_field``: a curved metric field
    is built from the covariance spectrum (:func:`_anisotropic_conformal_family`),
    the volume-normalised Ricci flow is integrated with Christoffel/Ricci tensors
    from ``jax.jacfwd`` (no finite differences), and the spectrum is contracted
    toward its bulk mean by the **measured curvature-uniformization ratio**

        $$\gamma = \frac{\operatorname{std}_x R(T)}{\operatorname{std}_x R(0)}
            \in [0, 1],\qquad
          \log\lambda_i^{\text{denoised}} = \overline{\log\lambda}
            + \gamma\,(\log\lambda_i - \overline{\log\lambda}).$$

    As the flow drives the scalar curvature to a constant ($\gamma \to 0$), the
    spectrum collapses to the denoised constant-curvature manifold; at $T=0$
    ($\gamma = 1$) the input is returned unchanged.

    Parameters
    ----------
    empirical_metric_tensor : array-like, shape (N, N)
        Empirical covariance / correlation (SPD). Accepts ``jax.Array``.
    flow_duration : float
        Total integration time T. Larger T ⇒ stronger denoising.
    manifold_dimension : int
        Retained for API compatibility; the flow uses the metric dimension N.
    n_steps : int
        Number of normalised-flow sub-steps.
    preserve_trace : bool
        If True, rescale the output to hold the trace (total variance) fixed.
    require_jax : bool
        Default True. The exact autodiff flow is mandatory; there is no NumPy
        finite-difference fallback (O(h^2) corrupts the non-linear Ricci PDE).

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
    _require_jax_strict()

    raw = as_array(empirical_metric_tensor)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("empirical_metric_tensor must be square (N, N)")
    if flow_duration <= 0:
        return _symmetrize(np.asarray(raw, dtype=float))

    g_in = _nearest_spd(raw)
    vals, vecs = np.linalg.eigh(g_in)
    vals = np.clip(vals, 1e-10, None)

    family, theta0, n = _anisotropic_conformal_family(vals)
    xs = _curvature_sample_cloud(n)
    _, history = ricci_flow_metric_field(
        family, theta0, xs, float(flow_duration), n_steps=int(n_steps), max_step=0.05
    )

    disp0 = max(float(history[0]), 1e-12)
    gamma = float(np.clip(history[-1] / disp0, 0.0, 1.0))

    log_vals = np.log(vals)
    mean_log = float(np.mean(log_vals))
    denoised = np.exp(mean_log + gamma * (log_vals - mean_log))

    g_out = _nearest_spd((vecs * denoised) @ vecs.T)
    if preserve_trace and np.trace(g_out) > 0:
        g_out *= float(np.trace(g_in)) / float(np.trace(g_out))

    out_eig = np.linalg.eigvalsh(g_out)
    logger.info(
        "[RICCI-AUTODIFF] T=%.3f N=%d gamma=%.4f curv-disp %.4f->%.4f spread %.4f->%.4f",
        flow_duration,
        n,
        gamma,
        float(history[0]),
        float(history[-1]),
        float(np.std(log_vals)),
        float(np.std(np.log(np.clip(out_eig, 1e-12, None)))),
    )
    return g_out


def ricci_flow_curvature_field(
    empirical_metric_tensor,
    flow_duration: float,
    manifold_dimension: int,
    samples: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact scalar-curvature-dispersion trajectory along the autodiff Ricci flow.

    Runs a single exact metric-field flow and returns the time grid together with
    the (autodiff-measured) scalar-curvature dispersion resampled onto it, so
    downstream signal models can ingest a dynamic, continuously-uniformizing
    curvature signal rather than a single static matrix.
    """
    _require_jax_strict()
    g = _nearest_spd(as_array(empirical_metric_tensor))
    vals, _ = np.linalg.eigh(g)
    vals = np.clip(vals, 1e-10, None)
    family, theta0, n = _anisotropic_conformal_family(vals)
    xs = _curvature_sample_cloud(n)
    n_steps = max(int(samples) - 1, 1) * 4
    _, history = ricci_flow_metric_field(
        family, theta0, xs, float(flow_duration), n_steps=n_steps, max_step=0.05
    )
    times = np.linspace(0.0, flow_duration, samples)
    src_t = np.linspace(0.0, flow_duration, len(history))
    scalars = np.interp(times, src_t, np.asarray(history, dtype=float))
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
