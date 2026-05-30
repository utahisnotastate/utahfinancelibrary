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
from typing import Tuple

import numpy as np

from src.core._backend import as_array

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

    Returns
    -------
    np.ndarray
        Denoised SPD covariance metric of the same shape.
    """
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
