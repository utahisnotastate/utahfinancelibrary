r"""
Tensor-network (MPS) entanglement diagnostics for portfolio diversification.

Quantum many-body physics measures how "entangled" a state is via the Von
Neumann entropy of a subsystem's reduced density matrix. We borrow that
machinery as a **diversification spectrum** for an asset universe:

- Treat the normalised covariance $\rho = \Sigma / \operatorname{tr}\Sigma$ as a
  density matrix. Its Von Neumann entropy $S(\rho) = -\operatorname{tr}(\rho\log\rho)$
  measures how many *independent* risk directions are active. $e^{S}$ is an
  effective number of bets (low $S$ ⇒ one factor dominates, i.e. fragile).
- Encode a portfolio "state vector" over $N$ assets and compute the
  **bipartition entanglement entropy** via a Matrix Product State (MPS)
  decomposition (sequential SVD with bond-dimension truncation $\chi$). The bond
  entropies are the genuine entanglement entropies across each cut.

Honest scope
------------
Entanglement/spectral entropy is a *concentration / diversification* measure. It
quantifies how broadly risk is spread across independent modes. It does **not**
manufacture a deterministic hedge in which "one asset must rise when another
falls" — no allocation can guarantee that, and correlations do move toward 1 in a
crash. High entanglement entropy means the portfolio is not dominated by a single
collapsing mode; it is a robustness diagnostic, not an immunity theorem.

Backend: NumPy.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np

from src.core._backend import as_array


def _entropy_from_spectrum(probs: np.ndarray) -> float:
    p = probs[probs > 1e-15]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def von_neumann_portfolio_entropy(covariance) -> float:
    r"""
    Von Neumann entropy $S(\rho)$ of $\rho = \Sigma/\operatorname{tr}\Sigma$.

    Equals the spectral (eigenvalue) entropy of the normalised covariance. Larger
    ⇒ risk spread across more independent modes.
    """
    cov = as_array(covariance)
    cov = 0.5 * (cov + cov.T)
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    if total <= 0:
        return 0.0
    return _entropy_from_spectrum(vals / total)


def effective_number_of_bets(covariance) -> float:
    r"""$e^{S(\rho)}$ — the effective number of independent risk directions."""
    return float(np.exp(von_neumann_portfolio_entropy(covariance)))


def bipartition_entanglement_entropy(state_vector, cut: int) -> float:
    r"""
    Entanglement entropy of an $N$-qubit-style state across a bipartition.

    ``state_vector`` has length $2^N$ (will be L2-normalised). ``cut`` is the
    number of leading sites in subsystem $A$. Returns $S(\rho_A)$ via the SVD of
    the reshaped amplitude matrix (Schmidt decomposition).
    """
    psi = as_array(state_vector).ravel()
    n_amp = psi.size
    n_sites = int(round(np.log2(n_amp)))
    if 2 ** n_sites != n_amp:
        raise ValueError("state_vector length must be a power of 2")
    if not (1 <= cut < n_sites):
        raise ValueError("cut must satisfy 1 <= cut < n_sites")
    norm = np.linalg.norm(psi)
    if norm <= 0:
        return 0.0
    psi = psi / norm
    mat = psi.reshape(2 ** cut, 2 ** (n_sites - cut))
    sv = np.linalg.svd(mat, compute_uv=False)
    schmidt = sv ** 2  # already normalised since psi is unit norm
    return _entropy_from_spectrum(schmidt)


@dataclasses.dataclass(frozen=True)
class MPSDecomposition:
    tensors: List[np.ndarray]  # rank-3 site tensors (left, phys, right)
    bond_entropies: List[float]  # entanglement entropy at each internal bond
    bond_dimensions: List[int]


def mps_decompose(state_vector, bond_dim: int = 8) -> MPSDecomposition:
    r"""
    Left-canonical Matrix Product State decomposition of a $2^N$ state vector via
    sequential SVD, truncated to bond dimension $\chi$ = ``bond_dim``.

    Returns the site tensors, the entanglement entropy at each bond (computed
    from the *untruncated* Schmidt spectrum), and the realised bond dimensions.
    """
    psi = as_array(state_vector).ravel().astype(np.float64)
    n_amp = psi.size
    n_sites = int(round(np.log2(n_amp)))
    if 2 ** n_sites != n_amp:
        raise ValueError("state_vector length must be a power of 2")
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm

    tensors: List[np.ndarray] = []
    bond_entropies: List[float] = []
    bond_dims: List[int] = []

    residual = psi.reshape(1, n_amp)  # (left_bond, rest)
    left_bond = 1
    for site in range(n_sites - 1):
        rows = left_bond * 2
        residual = residual.reshape(rows, -1)
        u, s, vh = np.linalg.svd(residual, full_matrices=False)

        full_spectrum = (s ** 2)
        ssum = float(np.sum(full_spectrum))
        if ssum > 0:
            bond_entropies.append(_entropy_from_spectrum(full_spectrum / ssum))
        else:
            bond_entropies.append(0.0)

        chi = min(bond_dim, s.size)
        u_t = u[:, :chi]
        s_t = s[:chi]
        vh_t = vh[:chi, :]

        tensors.append(u_t.reshape(left_bond, 2, chi))
        bond_dims.append(chi)
        residual = (np.diag(s_t) @ vh_t)
        left_bond = chi

    tensors.append(residual.reshape(left_bond, 2, 1))
    return MPSDecomposition(tensors=tensors, bond_entropies=bond_entropies, bond_dimensions=bond_dims)


def entanglement_diversification_weights(covariance) -> np.ndarray:
    r"""
    Diversification heuristic: weight assets to flatten the risk spectrum.

    Allocates inversely to each asset's projection onto the dominant
    (systemic) eigenvector, then normalises. This raises the portfolio's Von
    Neumann entropy (spreads risk off the collapsing mode). It is a heuristic
    diversifier, **not** a guaranteed hedge.
    """
    cov = as_array(covariance)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    dominant = vecs[:, int(np.argmax(vals))]
    load = np.abs(dominant)
    inv = 1.0 / np.clip(load, 1e-6, None)
    w = inv / np.sum(inv)
    return w
