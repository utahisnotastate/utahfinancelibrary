import numpy as np

from src.core.topological_allocation import (
    betti_number_divergence_test,
    rolling_betti0,
)


def _crash_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)
    T = 260
    n = 9

    def blocks():
        f = [rng.normal(size=(T, 1)) for _ in range(3)]
        cols = (
            [f[0] + 0.3 * rng.normal(size=(T, 1)) for _ in range(3)]
            + [f[1] + 0.3 * rng.normal(size=(T, 1)) for _ in range(3)]
            + [f[2] + 0.3 * rng.normal(size=(T, 1)) for _ in range(3)]
        )
        return np.hstack(cols)

    calm = blocks()
    market = rng.normal(size=(T, 1))
    # persistent block structure + dominant systemic market mode
    crash = blocks() + 2.5 * np.tile(market, (1, n))
    return np.vstack([calm, crash])


def test_baseline_collapses_trp_separates():
    returns = _crash_dataset()
    rep = betti_number_divergence_test(
        returns, window=60, radius_quantile=0.35, n_market_factors=1, step=20
    )
    assert rep.baseline_collapsed is True
    assert rep.trp_maintained_separation is True
    assert rep.baseline_b0.min() <= 1
    assert rep.trp_b0.min() > 1


def test_rolling_betti0_shapes():
    returns = _crash_dataset()
    traj = rolling_betti0(returns, window=60, radius_quantile=0.3, step=30)
    assert traj.ndim == 1
    assert np.all(traj >= 1)
