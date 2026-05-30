import numpy as np

from src.core.tick_observer import (
    QuadraticCovariationObserver,
    drawdown_metric_from_covariation,
    realized_covariation,
    two_scale_realized_covariance,
)


def _simulate_gbm_paths(true_cov: np.ndarray, n_steps: int, dt: float, seed: int = 0):
    """Simulate correlated GBM log-price paths with known instantaneous covariance."""
    rng = np.random.default_rng(seed)
    n = true_cov.shape[0]
    L = np.linalg.cholesky(true_cov)
    increments = (rng.normal(size=(n_steps, n)) @ L.T) * np.sqrt(dt)
    log_prices = np.cumsum(increments, axis=0)
    prices = np.exp(log_prices)  # start near 1
    return prices


def test_realized_covariation_converges_as_dt_shrinks():
    true_cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    # finer sampling (more steps, smaller dt) over the same horizon t=1 => closer
    coarse = realized_covariation(
        _simulate_gbm_paths(true_cov, n_steps=250, dt=1 / 250, seed=1), horizon=1.0
    )
    fine = realized_covariation(
        _simulate_gbm_paths(true_cov, n_steps=20000, dt=1 / 20000, seed=1), horizon=1.0
    )
    err_coarse = np.linalg.norm(coarse - true_cov)
    err_fine = np.linalg.norm(fine - true_cov)
    assert err_fine < err_coarse
    assert err_fine < 0.02  # near machine/statistical exactness at high frequency


def test_two_scale_reduces_microstructure_bias():
    true_cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    prices = _simulate_gbm_paths(true_cov, n_steps=8000, dt=1 / 8000, seed=2)
    # add multiplicative microstructure noise to the observed prices
    rng = np.random.default_rng(2)
    noisy = prices * (1.0 + 0.001 * rng.normal(size=prices.shape))

    naive = realized_covariation(noisy, horizon=1.0)
    tsrv = two_scale_realized_covariance(noisy, n_subsamples=10, horizon=1.0)

    # naive estimator is biased upward by noise; TSRV is closer to the truth
    naive_err = abs(np.trace(naive) - np.trace(true_cov))
    tsrv_err = abs(np.trace(tsrv) - np.trace(true_cov))
    assert tsrv_err < naive_err


def test_online_observer_matches_batch():
    true_cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    prices = _simulate_gbm_paths(true_cov, n_steps=2000, dt=1 / 2000, seed=3)

    obs = QuadraticCovariationObserver(n_assets=2)
    obs.ingest_prices(prices, dt=1 / 2000)

    # compare the raw accumulated covariation (rate normalisation aside): the
    # online outer-product accumulation equals the batch dx^T dx exactly
    batch = realized_covariation(prices, horizon=1.0)
    online = obs.covariation
    assert np.allclose(online, batch, atol=1e-10)
    assert obs.n_increments == prices.shape[0] - 1


def test_drift_independence():
    # quadratic variation is drift-invariant: adding a deterministic trend
    # must not change the realized covariation (up to higher-order terms)
    true_cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    prices = _simulate_gbm_paths(true_cov, n_steps=10000, dt=1 / 10000, seed=4)
    trend = np.exp(np.linspace(0, 0.5, prices.shape[0]))[:, None]
    drifted = prices * trend

    base = realized_covariation(prices, horizon=1.0)
    with_drift = realized_covariation(drifted, horizon=1.0)
    assert np.linalg.norm(base - with_drift) < 1e-3


def test_drawdown_metric_projection():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    g = drawdown_metric_from_covariation(cov, w)
    assert np.isclose(g, float(w @ cov @ w), atol=1e-9)
    assert g > 0
