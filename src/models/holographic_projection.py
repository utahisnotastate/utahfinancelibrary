r"""
AdS/CFT-inspired holographic order-book embedding.

This module borrows the *geometry* of the AdS/CFT correspondence — a bulk space
of negative (hyperbolic) curvature whose boundary carries a lower-dimensional
field — as a **representation** of limit-order-book (LOB) depth. The deep book
(liquidity resting away from the mid) is the "bulk"; a single contemporaneous
microstructure pressure scalar on the mid is the "boundary".

Concretely, each price level at fractional distance $z = |p - p_\text{mid}|/p_\text{mid}$
from the mid is treated as a radial coordinate in a Poincaré-style hyperbolic
metric $ds^2 = (dz^2 + dx^2)/z^2$. Resting size is weighted by a near-boundary
kernel $w(z) = e^{-z/R}$ and aggregated into a bounded order-flow imbalance.

What this is and is NOT
-----------------------
- It IS a deterministic, contemporaneous transform of the **currently observed**
  book into a bounded pressure feature and a hyperbolic-depth descriptor.
- It is NOT an oracle. It does not "see the future", does not solve Einstein
  field equations, and carries no claim that the boundary moves before the bulk.
  There is no look-ahead: every output is a function of information available at
  the observation instant. Treat the output as one microstructure feature among
  many, to be validated out-of-sample.

Backend: NumPy (JAX-compatible via ``src.core._backend``).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from src.core._backend import as_array
from src.core.protocol_economics import YieldSplit, protocol_yield_split


@dataclasses.dataclass(frozen=True)
class HolographicState:
    pressure: float  # bounded order-flow imbalance in [-1, 1]
    bulk_depth_entropy: float  # dispersion of the hyperbolic mass distribution
    bid_boundary_mass: float
    ask_boundary_mass: float


def _level_distances(level_prices: np.ndarray, mid_price: float) -> np.ndarray:
    if mid_price <= 0:
        raise ValueError("mid_price must be positive")
    z = np.abs(level_prices - mid_price) / mid_price
    return np.clip(z, 1e-9, None)


def holographic_order_book_pressure(
    level_prices,
    bid_sizes,
    ask_sizes,
    mid_price: float,
    curvature_radius: float = 0.01,
) -> HolographicState:
    r"""
    Project a snapshot order book onto the conformal boundary as a bounded
    pressure scalar.

    Parameters
    ----------
    level_prices : array-like, shape (L,)
        Price of each book level (bids and asks share the grid; sizes are zero
        where a side has no resting volume).
    bid_sizes, ask_sizes : array-like, shape (L,)
        Resting size on each side at the corresponding level.
    mid_price : float
        Current mid price (the conformal boundary location).
    curvature_radius : float
        AdS curvature radius $R$ of the near-boundary weighting kernel
        $w(z) = e^{-z/R}$. Smaller $R$ ⇒ only liquidity very close to the mid
        contributes (sharper boundary).

    Returns
    -------
    HolographicState
        ``pressure`` in [-1, 1] (positive = bid/buy pressure), plus the bulk
        depth entropy and the boundary masses.
    """
    prices = as_array(level_prices)
    bids = as_array(bid_sizes)
    asks = as_array(ask_sizes)
    if not (prices.shape == bids.shape == asks.shape):
        raise ValueError("level_prices, bid_sizes, ask_sizes must share shape")
    if curvature_radius <= 0:
        raise ValueError("curvature_radius must be positive")

    z = _level_distances(prices, mid_price)
    w = np.exp(-z / curvature_radius)

    bid_mass = float(np.sum(w * bids))
    ask_mass = float(np.sum(w * asks))
    total = bid_mass + ask_mass
    pressure = 0.0 if total <= 0 else (bid_mass - ask_mass) / total

    # entropy of the (normalised) hyperbolic mass distribution over levels:
    # high entropy = depth spread across the bulk, low = concentrated at boundary.
    mass = w * (bids + asks)
    s = float(np.sum(mass))
    if s > 0:
        p = mass / s
        p = p[p > 0]
        bulk_entropy = float(-np.sum(p * np.log(p)))
    else:
        bulk_entropy = 0.0

    return HolographicState(
        pressure=float(np.clip(pressure, -1.0, 1.0)),
        bulk_depth_entropy=bulk_entropy,
        bid_boundary_mass=bid_mass,
        ask_boundary_mass=ask_mass,
    )


def holographic_alpha_with_split(
    holographic_pressure: float,
    notional: float,
    expected_edge_bps: float = 1.0,
    apply_split: bool = True,
) -> Optional[YieldSplit]:
    r"""
    Convert a (validated, out-of-sample) pressure reading into a candidate yield
    figure and optionally route it through the transparent protocol split.

    This is illustrative bookkeeping only: ``expected_edge_bps`` must come from
    your own backtest, not from this module. The split is opt-in and fully
    removable — the pressure computation above does not depend on it.
    """
    raw_yield = float(holographic_pressure) * float(notional) * (expected_edge_bps * 1e-4)
    if not apply_split:
        return None
    return protocol_yield_split(np.asarray(raw_yield))
