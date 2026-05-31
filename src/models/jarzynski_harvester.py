r"""
Jarzynski / Crooks non-equilibrium free-energy estimator.

The Jarzynski equality and the Crooks fluctuation theorem are genuine, rigorously
proven results of stochastic thermodynamics:

.. math::
    \langle e^{-\beta W} \rangle = e^{-\beta \Delta F},
    \qquad
    \frac{P_F(W)}{P_R(-W)} = e^{\beta (W - \Delta F)} .

They let you recover an **equilibrium** free-energy difference :math:`\Delta F`
from an ensemble of **non-equilibrium** work measurements :math:`W` collected
along driven trajectories. This module applies that machinery to a stream of
"work-like" quantities derived from order-flow pressure, to estimate a regime
free-energy gap and to quantify how far the flow is from equilibrium.

What this IS / is NOT (read before using)
------------------------------------------
- It **is** a consistent estimator of :math:`\Delta F` (Jarzynski), a measurement
  of the average dissipated work :math:`\langle W_{\text{diss}}\rangle =
  \langle W\rangle - \Delta F \ge 0`, and of the Kullback-Leibler divergence
  between the forward and time-reversed work distributions
  (:math:`\beta\langle W_{\text{diss}}\rangle = D_{\mathrm{KL}}(P_F\|P_R)`,
  the Kawai-Parrondo-Van den Broeck relation).
- Individual trajectories **can** have :math:`W < \Delta F` — these are real,
  finite-probability *microscopic fluctuations* (transient "second-law
  violations" in the Evans-Searles sense). The module reports their size and
  frequency as a **non-equilibrium diagnostic**.
- It is **NOT** a risk-free-arbitrage engine. A single fluctuation is not a
  tradable, repeatable, frictionless profit; the ensemble still obeys
  :math:`\langle W\rangle \ge \Delta F`. Any downstream "yield" figure must be
  validated against execution cost and out-of-sample, exactly like any other
  signal. We do not claim the Second Law of Thermodynamics can be monetised.

Backend: NumPy (JAX-compatible via ``src.core._backend``).
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np

from src.core._backend import as_array
from src.core.protocol_economics import enforce_universal_tithe


def log_mean_exp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(mean(exp(x)))`` along ``axis``."""
    x = as_array(x)
    m = np.max(x, axis=axis, keepdims=True)
    out = np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(x - m), axis=axis))
    return out


def jarzynski_free_energy(forward_work, inverse_temperature_beta: float, axis: int = -1) -> np.ndarray:
    r"""
    Estimate the free-energy difference :math:`\Delta F` from forward work samples
    via the Jarzynski equality :math:`\Delta F = -\tfrac1\beta\log\langle
    e^{-\beta W}\rangle`.

    Uses a stable log-mean-exp. The estimator is biased low for finite samples
    (a known property of the Jarzynski estimator); use enough trajectories.
    """
    if inverse_temperature_beta <= 0:
        raise ValueError("inverse_temperature_beta must be positive")
    w = as_array(forward_work)
    return -(1.0 / inverse_temperature_beta) * log_mean_exp(-inverse_temperature_beta * w, axis=axis)


def average_dissipated_work(forward_work, inverse_temperature_beta: float, axis: int = -1) -> np.ndarray:
    r"""
    Average dissipated work :math:`\langle W_{\text{diss}}\rangle = \langle W
    \rangle - \Delta F \ge 0`. Non-negativity is the (ensemble) second law and is
    guaranteed by Jensen's inequality for the Jarzynski estimator.
    """
    w = as_array(forward_work)
    delta_f = jarzynski_free_energy(w, inverse_temperature_beta, axis=axis)
    return np.mean(w, axis=axis) - delta_f


def work_distribution_kl_divergence(forward_work, reverse_work, bins: int = 64) -> float:
    r"""
    Kullback-Leibler divergence :math:`D_{\mathrm{KL}}(P_F(W)\,\|\,P_R(-W))`
    between the forward work distribution and the reflected reverse-protocol work
    distribution, estimated from shared histograms.

    By Crooks, this divergence equals :math:`\beta\langle W_{\text{diss}}\rangle`
    and is therefore non-negative — a clean scalar measure of irreversibility
    (distance from equilibrium) of the driving protocol.
    """
    f = as_array(forward_work).ravel()
    r = -as_array(reverse_work).ravel()  # Crooks compares P_F(W) with P_R(-W)
    lo = float(min(f.min(), r.min()))
    hi = float(max(f.max(), r.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pf, _ = np.histogram(f, bins=edges, density=True)
    pr, _ = np.histogram(r, bins=edges, density=True)
    width = edges[1] - edges[0]
    pf = pf * width
    pr = pr * width
    eps = 1e-12
    mask = pf > eps
    return float(np.sum(pf[mask] * np.log((pf[mask] + eps) / (pr[mask] + eps))))


@dataclasses.dataclass(frozen=True)
class FluctuationHarvest:
    """Result of a non-equilibrium fluctuation analysis on a work ensemble."""

    delta_free_energy: float
    mean_dissipated_work: float
    kl_divergence: float
    transient_violation_fraction: float  # fraction of paths with W < delta_F
    fluctuation_functional: np.ndarray  # per-path relu(delta_F - W) >= 0
    net: np.ndarray
    humanitarian: np.ndarray
    protocol: np.ndarray


def extract_fluctuation_arbitrage(
    forward_work_trajectories,
    reverse_work_trajectories,
    inverse_temperature_beta: float,
) -> FluctuationHarvest:
    r"""
    Quantify transient non-equilibrium fluctuations in an order-flow "work"
    ensemble and route the (positive) fluctuation functional through the
    transparent universal tithe.

    Steps:

    1. :math:`\Delta F` via the Jarzynski equality.
    2. Per-trajectory fluctuation functional :math:`\max(\Delta F - W, 0)` — the
       magnitude of transient excursions below the free-energy gap. Their
       frequency is reported as ``transient_violation_fraction``.
    3. The irreversibility scale :math:`D_{\mathrm{KL}}(P_F\|P_R)` and the mean
       dissipated work, which bound the ensemble behaviour.

    The fluctuation functional is a **diagnostic magnitude**, not a guaranteed
    payoff. It is passed through :func:`enforce_universal_tithe` only so that, if
    a validated strategy *does* monetise these fluctuations downstream, the
    accounting split is already wired and auditable.
    """
    if inverse_temperature_beta <= 0:
        raise ValueError("inverse_temperature_beta must be positive")
    fwd = as_array(forward_work_trajectories)
    rev = as_array(reverse_work_trajectories)

    delta_f = float(jarzynski_free_energy(fwd, inverse_temperature_beta))
    fluct = np.maximum(delta_f - fwd, 0.0)
    violation_fraction = float(np.mean(fwd < delta_f))
    mean_diss = float(np.mean(fwd) - delta_f)
    kl = work_distribution_kl_divergence(fwd, rev)

    net, humanitarian, protocol = enforce_universal_tithe(fluct)
    return FluctuationHarvest(
        delta_free_energy=delta_f,
        mean_dissipated_work=mean_diss,
        kl_divergence=kl,
        transient_violation_fraction=violation_fraction,
        fluctuation_functional=np.asarray(fluct),
        net=net,
        humanitarian=humanitarian,
        protocol=protocol,
    )
