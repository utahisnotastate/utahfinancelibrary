r"""
Pathwise metric-tensor observer — realized quadratic (co)variation from ticks.

The Riemannian metric governing the Laplace-Beltrami generator of the portfolio
manifold is the instantaneous quadratic covariation of the log-price paths:

    $$g_{ij}(t) = \frac{d}{dt}\,\langle X_i, X_j\rangle_t.$$

For a continuous semi-martingale the realized covariation

    $$[X_i, X_j]^{(\Delta)}_t = \sum_{k} \big(X_i(t_{k+1}) - X_i(t_k)\big)
        \big(X_j(t_{k+1}) - X_j(t_k)\big)$$

converges in probability (and along subsequences, almost surely) to the true
quadratic covariation $\langle X_i, X_j\rangle_t$ as the mesh $\Delta \to 0$
— **independent of the drift**. The metric is therefore an $\mathcal{F}_t$-measurable
observable read off the tick stream, not a free parameter requiring a look-back
average.

Real exchange ticks carry microstructure noise, which biases the naive realized
covariance upward as $\Delta \to 0$. The Two-Scale Realized Volatility (TSRV)
estimator of Zhang, Mykland & Aït-Sahalia (2005) removes this bias and is the
consistent, noise-robust realization of the same limit. Use it on raw ticks.

Backend: NumPy (no JAX needed for the estimator itself; the resulting metric
feeds the JAX generator / PINN downstream).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.core._backend import as_array

logger = logging.getLogger(__name__)


def _nearest_spd(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def log_price_increments(prices: np.ndarray) -> np.ndarray:
    """Return log-return increments ΔX of a (T, N) price path (positive prices)."""
    prices = as_array(prices)
    if prices.ndim == 1:
        prices = prices[:, None]
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive for log returns")
    logp = np.log(prices)
    return np.diff(logp, axis=0)


def realized_covariation(
    prices: np.ndarray,
    horizon: float = 1.0,
    annualize: bool = False,
) -> np.ndarray:
    r"""
    Realized quadratic covariation matrix $[X_i, X_j]_t$ from a price path.

    With ``horizon`` = total elapsed time $t$, the **metric tensor** (covariation
    *rate*) is returned: $g_{ij} = [X_i, X_j]_t / t$. Set ``horizon=1`` to obtain
    the raw realized covariation over the sample.
    """
    dx = log_price_increments(prices)
    qv = dx.T @ dx  # sum of outer products of increments
    g = qv / horizon if horizon > 0 else qv
    return _nearest_spd(g)


def two_scale_realized_covariance(
    prices: np.ndarray,
    n_subsamples: int = 5,
    horizon: float = 1.0,
) -> np.ndarray:
    r"""
    Noise-robust Two-Scale Realized Volatility/Covariance (TSRV).

    Combines the all-data (fast) realized covariation with the average of slow
    subsampled estimators to cancel the leading microstructure-noise bias:

        $$\widehat{\langle X\rangle}^{TSRV} = \widehat{\langle X\rangle}^{(slow)}
            - \frac{\bar n}{n}\,\widehat{\langle X\rangle}^{(all)}$$

    where ``n_subsamples`` sets the slow grid. Converges to the integrated
    covariation as the noise is removed. Returns the SPD metric rate.
    """
    prices = as_array(prices)
    if prices.ndim == 1:
        prices = prices[:, None]
    n_total = prices.shape[0] - 1
    if n_total < n_subsamples * 2:
        # not enough data to subsample meaningfully; fall back to raw estimator
        return realized_covariation(prices, horizon=horizon)

    fast = realized_covariation(prices, horizon=1.0)

    slow_acc = np.zeros_like(fast)
    counts = 0
    for offset in range(n_subsamples):
        sub = prices[offset::n_subsamples]
        if sub.shape[0] < 2:
            continue
        slow_acc += realized_covariation(sub, horizon=1.0)
        counts += 1
    slow = slow_acc / max(counts, 1)

    n_bar = n_total / n_subsamples
    tsrv = slow - (n_bar / n_total) * fast
    g = _nearest_spd(tsrv)
    return g / horizon if horizon > 0 else g


@dataclasses.dataclass
class _OnlineState:
    last_log_price: Optional[np.ndarray] = None
    covariation: Optional[np.ndarray] = None
    elapsed: float = 0.0
    n_increments: int = 0


class QuadraticCovariationObserver:
    r"""
    Online, pathwise metric observer. Ingests ticks as they arrive and maintains
    the running quadratic covariation $\langle X\rangle_t$ — **no look-back
    window average, no lag**. The instantaneous metric is the covariation rate
    $g(t) = \langle X\rangle_t / t$.

    Parameters
    ----------
    n_assets : int
        Dimension of the price vector.
    decay : float, optional
        If given in (0, 1], applies an exponential forgetting factor so the
        observer tracks the *instantaneous* metric of a time-varying volatility
        regime (Itô diffusion with stochastic vol) rather than the path average.
        ``decay=1.0`` (default) accumulates the exact pathwise covariation.
    """

    def __init__(self, n_assets: int, decay: float = 1.0) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.n_assets = n_assets
        self.decay = decay
        self._state = _OnlineState(
            covariation=np.zeros((n_assets, n_assets)),
            elapsed=0.0,
        )

    def ingest(self, log_price: np.ndarray, dt: float = 1.0) -> None:
        """Ingest one tick's log-price vector with elapsed time ``dt`` since prev."""
        lp = as_array(log_price).ravel()
        if lp.shape[0] != self.n_assets:
            raise ValueError("log_price dimension mismatch")
        st = self._state
        if st.last_log_price is not None:
            dx = lp - st.last_log_price
            outer = np.outer(dx, dx)
            if self.decay < 1.0:
                st.covariation = self.decay * st.covariation + outer
                st.elapsed = self.decay * st.elapsed + dt
            else:
                st.covariation = st.covariation + outer
                st.elapsed += dt
            st.n_increments += 1
        st.last_log_price = lp

    def ingest_prices(self, prices: np.ndarray, dt: float = 1.0) -> None:
        """Convenience: ingest a (T, N) positive-price path tick by tick."""
        prices = as_array(prices)
        if prices.ndim == 1:
            prices = prices[:, None]
        if np.any(prices <= 0):
            raise ValueError("prices must be strictly positive")
        for row in np.log(prices):
            self.ingest(row, dt=dt)

    @property
    def covariation(self) -> np.ndarray:
        """Accumulated quadratic covariation matrix (SPD-projected)."""
        return _nearest_spd(self._state.covariation)

    def metric_tensor(self) -> np.ndarray:
        r"""Instantaneous metric $g_{ij}(t) = \langle X_i, X_j\rangle_t / t$."""
        st = self._state
        if st.elapsed <= 0:
            return np.eye(self.n_assets)
        return _nearest_spd(st.covariation / st.elapsed)

    @property
    def n_increments(self) -> int:
        return self._state.n_increments


def metric_field_from_covariation(metric_matrix: np.ndarray):
    r"""
    Build a position-independent metric field ``g(x) -> Sigma`` from an observed
    covariation matrix, suitable for the autodiff Laplace-Beltrami generator and
    Ricci-flow geometry. (Position dependence enters via the conformal families
    in ``manifold_kernel``; this is the flat measured metric.)
    """
    sigma = _nearest_spd(as_array(metric_matrix))

    def metric_fn(x):  # x is a coordinate vector; metric is the measured tensor
        try:  # support both jax and numpy callers
            import jax.numpy as jnp

            return jnp.asarray(sigma)
        except ImportError:  # pragma: no cover
            return sigma

    return metric_fn


def drawdown_metric_from_covariation(metric_matrix: np.ndarray, weights: np.ndarray) -> float:
    r"""
    Project the observed covariation onto a portfolio's weights to obtain the
    scalar diffusion coefficient of the **portfolio value** process — the 1D
    metric $g = w^\top \Sigma w$ used by the Laplace-Beltrami drawdown bound.
    """
    sigma = _nearest_spd(as_array(metric_matrix))
    w = as_array(weights).ravel()
    return float(w @ sigma @ w)
