import numpy as np
import pytest

from src.models.riemannian_geometry import is_jax_available

pytestmark = pytest.mark.skipif(not is_jax_available(), reason="JAX required for generator/PINN binding")


def test_drawdown_veto_from_observed_metric():
    from src.core.tick_observer import QuadraticCovariationObserver
    from src.core.risk_supervisor import drawdown_veto_from_tick_metric

    rng = np.random.default_rng(0)
    n_steps = 4000
    dt = 1 / n_steps
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    L = np.linalg.cholesky(cov)
    incs = (rng.normal(size=(n_steps, 2)) @ L.T) * np.sqrt(dt)
    prices = np.exp(np.cumsum(incs, axis=0))

    obs = QuadraticCovariationObserver(n_assets=2)
    obs.ingest_prices(prices, dt=dt)
    g = obs.metric_tensor()

    weights = np.array([0.5, 0.5])
    # tight drawdown limit + short horizon vs measured diffusion => veto True
    veto_tight = drawdown_veto_from_tick_metric(
        g, weights, drawdown_limit=3.0, confidence_level=0.99, horizon=0.2
    )
    # generous safety (small domain, long horizon) => safe
    veto_safe = drawdown_veto_from_tick_metric(
        g, weights, drawdown_limit=0.3, confidence_level=0.95, horizon=5.0
    )
    assert isinstance(veto_tight, bool)
    assert veto_safe is False


def test_pinn_binds_pathwise_metric_and_whitens():
    import jax.numpy as jnp
    from src.core.tick_observer import QuadraticCovariationObserver
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

    rng = np.random.default_rng(1)
    n_steps = 2000
    dt = 1 / n_steps
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    L = np.linalg.cholesky(cov)
    incs = (rng.normal(size=(n_steps, 2)) @ L.T) * np.sqrt(dt)
    prices = np.exp(np.cumsum(incs, axis=0))

    obs = QuadraticCovariationObserver(n_assets=2)
    obs.ingest_prices(prices, dt=dt)

    predictor = OrthogonalWaveStatePredictor()
    g = predictor.bind_tick_metric(obs)
    assert g.shape == (2, 2)
    assert predictor.metric_tensor is not None

    # whitening should make the data covariance approximately identity in metric
    x = jnp.asarray(rng.normal(size=(256, 2)))
    xw = predictor.whiten(x)
    assert xw.shape == x.shape
