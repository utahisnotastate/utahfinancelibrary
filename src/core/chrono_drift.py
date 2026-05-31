r"""
Malliavin / Skorokhod calculus — pathwise sensitivity tooling.

Malliavin calculus and the Skorokhod (anticipating) integral are genuine,
standard objects in stochastic analysis. Their established, defensible use in
quantitative finance is **computing sensitivities (Greeks)** and decomposing the
risk of a payoff with respect to perturbations of the driving noise — see
Fournié et al. (1999), "Applications of Malliavin calculus to Monte-Carlo methods
in finance".

READ THIS BEFORE USING
----------------------
The Skorokhod integral is *anticipating*: evaluating it requires the **entire**
Brownian path, including its future. That is exactly why it is **not** a trading
signal. You cannot observe tomorrow's path today, so nothing here extracts future
price information into the present, and there is no "advanced wave" that bleeds
drift backward in time — that would be look-ahead bias, the canonical way to
produce a backtest that is fraudulent out of sample.

These functions are for:
- computing Greeks via Malliavin weights on **simulated** complete paths, and
- decomposing the risk of a payoff on **historical, already-realised** paths
  (scenario analysis), where the full path is legitimately known after the fact.

Backend: NumPy.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from src.core._backend import as_array


def malliavin_derivative_terminal_gbm(
    s0: float,
    sigma: float,
    drift: float,
    horizon: float,
    n_steps: int,
) -> np.ndarray:
    r"""
    Malliavin derivative $D_t S_T$ of the terminal value of a geometric Brownian
    motion $dS = \mu S\,dt + \sigma S\,dW$.

    Closed form: $D_t S_T = \sigma\, S_T$ for $t \le T$ (constant in $t$ up to
    $T$). Returned on the time grid as a length-``n_steps`` vector of the
    coefficient $\sigma S_T$ in expectation (here the deterministic expectation
    $\mathbb{E}[S_T] = S_0 e^{\mu T}$ is used for a reproducible reference).
    """
    if horizon <= 0 or n_steps < 1:
        raise ValueError("horizon>0 and n_steps>=1 required")
    expected_st = s0 * np.exp(drift * horizon)
    coeff = sigma * expected_st
    return np.full(n_steps, coeff, dtype=np.float64)


def skorokhod_integral(integrand: np.ndarray, brownian_increments: np.ndarray, malliavin_trace: np.ndarray) -> float:
    r"""
    Discrete Skorokhod integral $\delta(u)$ of a (possibly anticipating) integrand.

    $$\delta(u) = \sum_i u_i\,\Delta W_i - \sum_i D_i u_i,$$

    where the second sum is the trace of the Malliavin derivative of the
    integrand (the divergence correction). For an **adapted** integrand
    $D_i u_i = 0$ and the Skorokhod integral reduces to the Itô sum
    $\sum_i u_i \Delta W_i$ — verified in the tests.
    """
    u = as_array(integrand).ravel()
    dW = as_array(brownian_increments).ravel()
    trace = as_array(malliavin_trace).ravel()
    if not (u.shape == dW.shape == trace.shape):
        raise ValueError("integrand, increments and malliavin_trace must share shape")
    ito_part = float(np.sum(u * dW))
    correction = float(np.sum(trace))
    return ito_part - correction


def malliavin_delta_european_call(
    s0: float,
    strike: float,
    sigma: float,
    rate: float,
    horizon: float,
    n_paths: int = 100_000,
    seed: int = 0,
) -> float:
    r"""
    Monte-Carlo **delta** of a European call via the Malliavin weight method.

    $$\Delta = e^{-rT}\,\mathbb{E}\!\left[ \text{payoff}\cdot
        \frac{W_T}{S_0\,\sigma\,T} \right].$$

    This is a real, drift-free estimator of $\partial C/\partial S_0$; it
    converges to the Black-Scholes delta. It uses only the simulated terminal
    state — a legitimate, non-anticipating use of the Malliavin weight.
    """
    if s0 <= 0 or sigma <= 0 or horizon <= 0:
        raise ValueError("s0, sigma, horizon must be positive")
    rng = np.random.default_rng(seed)
    w_t = rng.normal(0.0, np.sqrt(horizon), size=n_paths)
    s_t = s0 * np.exp((rate - 0.5 * sigma ** 2) * horizon + sigma * w_t)
    payoff = np.maximum(s_t - strike, 0.0)
    weight = w_t / (s0 * sigma * horizon)
    return float(np.exp(-rate * horizon) * np.mean(payoff * weight))


@dataclasses.dataclass(frozen=True)
class PathRiskDecomposition:
    total_quadratic_variation: float
    per_step_contribution: np.ndarray


def decompose_realized_path_risk(log_prices) -> PathRiskDecomposition:
    r"""
    Decompose the risk of an **already-realised** (historical) log-price path into
    per-step quadratic-variation contributions.

    This is a backward-looking attribution on a complete path — no future
    information is used to act, it merely explains where realised variance came
    from. Returns total quadratic variation and the per-step contributions.
    """
    x = as_array(log_prices).ravel()
    if x.size < 2:
        raise ValueError("need at least two observations")
    dx = np.diff(x)
    contrib = dx ** 2
    return PathRiskDecomposition(
        total_quadratic_variation=float(np.sum(contrib)),
        per_step_contribution=contrib,
    )
