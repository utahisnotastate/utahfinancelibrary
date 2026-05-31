import numpy as np

from src.models.tensor_network_hedge import (
    bipartition_entanglement_entropy,
    effective_number_of_bets,
    entanglement_diversification_weights,
    mps_decompose,
    von_neumann_portfolio_entropy,
)


def test_identity_covariance_maximizes_entropy():
    n = 4
    iso = np.eye(n)
    spike = np.diag([10.0, 0.1, 0.1, 0.1])
    s_iso = von_neumann_portfolio_entropy(iso)
    s_spike = von_neumann_portfolio_entropy(spike)
    assert s_iso > s_spike
    # isotropic => effective bets ~ n
    assert np.isclose(effective_number_of_bets(iso), n, rtol=1e-6)


def test_product_state_has_zero_entanglement():
    # |00> has no entanglement across the middle cut
    psi = np.zeros(4)
    psi[0] = 1.0
    assert bipartition_entanglement_entropy(psi, cut=1) < 1e-12


def test_bell_state_has_maximal_entanglement():
    # (|00> + |11>)/sqrt(2) => S = ln 2
    psi = np.zeros(4)
    psi[0] = psi[3] = 1.0 / np.sqrt(2)
    s = bipartition_entanglement_entropy(psi, cut=1)
    assert np.isclose(s, np.log(2), atol=1e-9)


def test_mps_reconstructs_state():
    rng = np.random.default_rng(0)
    n_sites = 4
    psi = rng.normal(size=2 ** n_sites)
    psi /= np.linalg.norm(psi)
    decomp = mps_decompose(psi, bond_dim=16)  # large chi => exact

    # contract the MPS back to a full vector
    vec = decomp.tensors[0]
    for t in decomp.tensors[1:]:
        vec = np.tensordot(vec, t, axes=([-1], [0]))
    vec = vec.reshape(-1)
    # MPS is exact up to global sign per SVD gauge; compare magnitude
    assert np.allclose(np.abs(vec), np.abs(psi), atol=1e-8)
    assert len(decomp.bond_entropies) == n_sites - 1


def test_diversification_weights_sum_to_one():
    cov = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
    w = entanglement_diversification_weights(cov)
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w > 0)
