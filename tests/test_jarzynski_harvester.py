import numpy as np

from src.models.jarzynski_harvester import (
    average_dissipated_work,
    extract_fluctuation_arbitrage,
    jarzynski_free_energy,
    log_mean_exp,
    work_distribution_kl_divergence,
)


def test_log_mean_exp_matches_naive():
    x = np.array([0.1, 0.5, -0.3, 2.0])
    assert np.isclose(log_mean_exp(x), np.log(np.mean(np.exp(x))))


def test_free_energy_exact_for_constant_work():
    # W constant => <e^{-bW}> = e^{-bW} => Delta F = W
    w = np.full(1000, 3.0)
    assert np.isclose(jarzynski_free_energy(w, inverse_temperature_beta=2.0), 3.0)


def test_average_dissipated_work_nonnegative():
    rng = np.random.default_rng(0)
    w = rng.normal(1.0, 0.5, size=5000)
    assert average_dissipated_work(w, inverse_temperature_beta=1.0) >= -1e-9


def test_kl_divergence_nonnegative_and_zero_for_identical():
    rng = np.random.default_rng(1)
    f = rng.normal(0.0, 1.0, size=20000)
    # reverse work reflected equals forward => KL ~ 0
    kl_same = work_distribution_kl_divergence(f, -f)
    assert kl_same >= -1e-9
    assert kl_same < 0.05
    g = rng.normal(3.0, 1.0, size=20000)
    assert work_distribution_kl_divergence(f, g) >= 0.0


def test_fluctuation_functional_nonnegative_and_tithe_conserves():
    rng = np.random.default_rng(2)
    fwd = rng.normal(1.0, 1.0, size=4000)
    rev = rng.normal(-1.0, 1.0, size=4000)
    res = extract_fluctuation_arbitrage(fwd, rev, inverse_temperature_beta=1.0)
    assert np.all(res.fluctuation_functional >= 0.0)
    assert 0.0 <= res.transient_violation_fraction <= 1.0
    recon = res.net + res.humanitarian + res.protocol
    assert np.allclose(recon, res.fluctuation_functional)
    # humanitarian = 10% of the (non-negative) functional
    assert np.allclose(res.humanitarian, res.fluctuation_functional * 0.10)
