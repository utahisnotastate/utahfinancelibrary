"""
Topological Risk Parity (TRP) — Persistent Homology allocation.

Replaces Euclidean dendrogram clustering (HRP/NCO) with a Vietoris-Rips
filtration over the asset-return correlation geometry. Capital is allocated
inversely to each asset's participation in persistent topological structure
(connected-component merges and 1-cycles / contagion loops).

Backend: NumPy (JAX optional). Inputs accept any array-like, incl. ``jax.Array``.

Notation
--------
- $b_0, b_1, \\dots$  : Betti numbers (components, loops, voids)
- $W_p$               : p-Wasserstein distance between persistence barcodes
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core._backend import as_array

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Persistence primitives
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class PersistencePair:
    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def correlation_distance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Build the classic Mantegna correlation distance $d = \\sqrt{2(1-\\rho)}$.

    ``returns`` has shape (T, N): T observations, N assets.
    """
    returns = as_array(returns)
    if returns.ndim != 2:
        raise ValueError("returns must be 2D (time, assets)")
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    dist = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def vietoris_rips_h0(distance: np.ndarray) -> Tuple[List[PersistencePair], np.ndarray]:
    """
    Exact H0 persistence via single-linkage (Kruskal) over the distance matrix.

    Returns the H0 barcode and the merge edges (N-1 of them) used, where each
    edge records which assets merged and at what filtration radius.
    """
    distance = as_array(distance)
    n = distance.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    weights = distance[iu, ju]
    order = np.argsort(weights, kind="mergesort")

    uf = _UnionFind(n)
    pairs: List[PersistencePair] = []
    merge_edges: List[Tuple[int, int, float]] = []
    for idx in order:
        i, j, w = int(iu[idx]), int(ju[idx]), float(weights[idx])
        if uf.union(i, j):
            # a component is born at 0 and dies when absorbed at radius w
            pairs.append(PersistencePair(dimension=0, birth=0.0, death=w))
            merge_edges.append((i, j, w))
            if len(merge_edges) == n - 1:
                break
    # one component persists forever
    pairs.append(PersistencePair(dimension=0, birth=0.0, death=float("inf")))
    return pairs, np.array(merge_edges, dtype=float) if merge_edges else np.empty((0, 3))


def betti_numbers_at(distance: np.ndarray, radius: float, max_dim: int = 2) -> Dict[int, int]:
    """
    Betti numbers of the Vietoris-Rips 1-/2-skeleton at a filtration ``radius``.

    - $b_0$ = connected components (exact)
    - $b_1$ = cycle rank of the 1-skeleton graph: $E - V + b_0$ (graph homology)
    - $b_2$ = approximated via filled-triangle deficit (count of empty triangular
      cavities), a tractable surrogate for 2D voids.
    """
    distance = as_array(distance)
    n = distance.shape[0]
    adj = (distance <= radius) & ~np.eye(n, dtype=bool)

    uf = _UnionFind(n)
    edges = 0
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu, ju):
        if adj[i, j]:
            edges += 1
            uf.union(int(i), int(j))
    b0 = len({uf.find(k) for k in range(n)})
    b1 = max(edges - n + b0, 0)

    betti = {0: b0, 1: b1}
    if max_dim >= 2:
        # count closed triangles; 2-voids surrogate = (#open triples that are
        # pairwise connected but not all three) capped to be non-negative.
        triangles = 0
        open_cavities = 0
        idxs = np.arange(n)
        for a in idxs:
            nbrs = np.where(adj[a])[0]
            nbrs = nbrs[nbrs > a]
            for bi in range(len(nbrs)):
                for ci in range(bi + 1, len(nbrs)):
                    b, c = int(nbrs[bi]), int(nbrs[ci])
                    if adj[b, c]:
                        triangles += 1
                    else:
                        open_cavities += 1
        betti[2] = max(open_cavities - triangles, 0)
    return betti


def _persistence_diagram_to_points(pairs: List[PersistencePair], dimension: int) -> np.ndarray:
    pts = [
        (p.birth, p.death)
        for p in pairs
        if p.dimension == dimension and np.isfinite(p.death)
    ]
    return np.array(pts, dtype=float) if pts else np.empty((0, 2))


def wasserstein_barcode_distance(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
    order: int = 1,
) -> float:
    """
    1D optimal-transport (sorted-persistence) Wasserstein distance between the
    persistence values of two diagrams, padded against the diagonal (death=birth).
    """
    a = np.sort((diagram_a[:, 1] - diagram_a[:, 0])) if diagram_a.size else np.array([])
    b = np.sort((diagram_b[:, 1] - diagram_b[:, 0])) if diagram_b.size else np.array([])
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    a = np.pad(a, (0, m - len(a)), constant_values=0.0)
    b = np.pad(b, (0, m - len(b)), constant_values=0.0)
    return float(np.power(np.sum(np.abs(a - b) ** order), 1.0 / order))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class TopologicalAllocation:
    weights: np.ndarray
    betti_numbers: Dict[int, int]
    barcode_wasserstein: float
    filtration_radius: float


def _asset_topological_loadings(
    distance: np.ndarray,
    merge_edges: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Per-asset topological load: longer/late merges and cycle participation imply
    the asset bridges otherwise-disjoint regions (systemic contagion routes) and
    is therefore penalized (lower weight).
    """
    n = distance.shape[0]
    load = np.ones(n, dtype=float)

    # late single-linkage merges = bridging asset → higher load
    if merge_edges.size:
        for i, j, w in merge_edges:
            contribution = w
            load[int(i)] += contribution
            load[int(j)] += contribution

    # cycle participation at the chosen radius
    adj = (distance <= radius) & ~np.eye(n, dtype=bool)
    degree = adj.sum(axis=1).astype(float)
    # high-degree nodes inside loops carry contagion; fold in gently
    load += 0.5 * degree
    return load


def optimize_topological_risk_parity(
    tensor_data,
    max_homology_dimension: int = 2,
    wasserstein_penalty: float = 0.01,
    filtration_quantile: float = 0.5,
) -> np.ndarray:
    """
    Compute optimal weights by penalizing assets that form persistent topological
    holes (contagion cycles) in the return manifold, steering the portfolio
    barcode toward a maximally dissipated (uniform-risk) state.

    Parameters
    ----------
    tensor_data : array-like, shape (T, N)
        Asset return tensor (time x assets). Accepts ``jax.Array`` or ndarray.
    max_homology_dimension : int
        Highest Betti dimension to compute (0..2 supported).
    wasserstein_penalty : float
        Strength of the barcode-uniformity regularizer applied to weights.
    filtration_quantile : float
        Quantile of pairwise distances used as the Betti evaluation radius.

    Returns
    -------
    np.ndarray
        Long-only weights summing to 1.
    """
    returns = as_array(tensor_data)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("tensor_data must be (time, assets) with >= 2 assets")

    n = returns.shape[1]
    distance = correlation_distance_matrix(returns)

    pairs, merge_edges = vietoris_rips_h0(distance)
    finite = distance[np.triu_indices(n, k=1)]
    radius = float(np.quantile(finite, filtration_quantile)) if finite.size else 0.0

    betti = betti_numbers_at(distance, radius, max_dim=max_homology_dimension)

    load = _asset_topological_loadings(distance, merge_edges, radius)

    # inverse-topological-load allocation (risk parity in topological metric)
    inv = 1.0 / np.maximum(load, 1e-9)
    weights = inv / inv.sum()

    # Wasserstein regularizer: pull the realized H0 barcode toward a uniform
    # (perfectly dissipated) target barcode, nudging weights toward 1/N.
    diagram = _persistence_diagram_to_points(pairs, dimension=0)
    if diagram.size:
        target_persist = np.full(diagram.shape[0], np.mean(diagram[:, 1] - diagram[:, 0]))
        target = np.column_stack([np.zeros_like(target_persist), target_persist])
        w_dist = wasserstein_barcode_distance(diagram, target)
    else:
        w_dist = 0.0

    uniform = np.full(n, 1.0 / n)
    blend = np.clip(wasserstein_penalty * w_dist, 0.0, 1.0)
    weights = (1.0 - blend) * weights + blend * uniform
    weights = weights / weights.sum()

    logger.info(
        "[TRP] Betti=%s radius=%.4f W1=%.4f blend=%.3f",
        betti,
        radius,
        w_dist,
        blend,
    )
    return weights


def topological_risk_parity_report(
    tensor_data,
    max_homology_dimension: int = 2,
    wasserstein_penalty: float = 0.01,
) -> TopologicalAllocation:
    """Rich variant returning weights plus Betti numbers and diagnostics."""
    returns = as_array(tensor_data)
    n = returns.shape[1]
    distance = correlation_distance_matrix(returns)
    finite = distance[np.triu_indices(n, k=1)]
    radius = float(np.quantile(finite, 0.5)) if finite.size else 0.0
    betti = betti_numbers_at(distance, radius, max_dim=max_homology_dimension)
    pairs, _ = vietoris_rips_h0(distance)
    diagram = _persistence_diagram_to_points(pairs, 0)
    if diagram.size:
        target_p = np.full(diagram.shape[0], np.mean(diagram[:, 1] - diagram[:, 0]))
        target = np.column_stack([np.zeros_like(target_p), target_p])
        w = wasserstein_barcode_distance(diagram, target)
    else:
        w = 0.0
    weights = optimize_topological_risk_parity(
        returns, max_homology_dimension, wasserstein_penalty
    )
    return TopologicalAllocation(
        weights=weights,
        betti_numbers=betti,
        barcode_wasserstein=w,
        filtration_radius=radius,
    )
