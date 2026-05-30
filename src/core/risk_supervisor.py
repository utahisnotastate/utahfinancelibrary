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
