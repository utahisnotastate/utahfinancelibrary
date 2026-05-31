import math

import numpy as np

from src.core.chrono_drift import (
    decompose_realized_path_risk,
    malliavin_delta_european_call,
    malliavin_derivative_terminal_gbm,
    skorokhod_integral,
)


def test_skorokhod_reduces_to_ito_for_adapted_integrand():
    rng = np.random.default_rng(0)
    n = 50
    dW = rng.normal(0.0, 0.1, size=n)
    u = rng.normal(size=n)  # adapted integrand => zero Malliavin trace
    trace = np.zeros(n)
    delta = skorokhod_integral(u, dW, trace)
    assert np.isclose(delta, float(np.sum(u * dW)))


def test_skorokhod_correction_subtracts_trace():
    dW = np.ones(3) * 0.1
    u = np.array([1.0, 2.0, 3.0])
    trace = np.array([0.5, 0.5, 0.5])
    expected = float(np.sum(u * dW)) - 1.5
    assert np.isclose(skorokhod_integral(u, dW, trace), expected)


def test_malliavin_derivative_shape_and_sign():
    d = malliavin_derivative_terminal_gbm(s0=100.0, sigma=0.2, drift=0.05, horizon=1.0, n_steps=10)
    assert d.shape == (10,)
    assert np.all(d > 0)  # sigma * E[S_T] > 0


def test_malliavin_delta_matches_black_scholes():
    s0, k, sigma, r, t = 100.0, 100.0, 0.2, 0.01, 1.0
    delta_mc = malliavin_delta_european_call(s0, k, sigma, r, t, n_paths=400_000, seed=1)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    bs_delta = 0.5 * (1.0 + math.erf(d1 / np.sqrt(2.0)))
    assert abs(delta_mc - bs_delta) < 0.02


def test_path_risk_decomposition_sums_to_total_qv():
    x = np.log(np.array([100.0, 101.0, 99.5, 100.2]))
    decomp = decompose_realized_path_risk(x)
    assert np.isclose(decomp.total_quadratic_variation, float(np.sum(decomp.per_step_contribution)))
