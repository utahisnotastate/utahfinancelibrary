"""k-sunflower disjoint capital routing — transfinite sieve topology layer.

Also hosts the Navier-Stokes liquidity-routing API, which models capital
reallocation as an incompressible fluid flow (viscosity = market impact,
pressure gradient = alpha signal) instead of a static L1 turnover penalty.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from src.core._backend import as_array

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CapitalNode:
    asset_id: str
    venue_id: str
    liquidity_volume: float
    yield_velocity: float
    entropy_drag: float

    @property
    def efficiency_score(self) -> float:
        return self.yield_velocity / (self.entropy_drag + 1e-9)


class UtahTransfiniteSieve:
    """
    Partitions liquidity nodes into disjoint petals (size <= k_bound)
    sorted by risk-adjusted yield efficiency.
    """

    def __init__(self, k_bound: int = 3) -> None:
        if k_bound < 1:
            raise ValueError("k_bound must be >= 1")
        self.k_bound = k_bound

    def apply_sunflower_lemma(self, capital_nodes: List[CapitalNode]) -> List[Set[CapitalNode]]:
        disjoint_routes: List[Set[CapitalNode]] = []
        current_petal: Set[CapitalNode] = set()

        ordered = sorted(capital_nodes, key=lambda n: n.efficiency_score, reverse=True)
        for node in ordered:
            if len(current_petal) < self.k_bound:
                current_petal.add(node)
            else:
                disjoint_routes.append(current_petal)
                current_petal = {node}

        if current_petal:
            disjoint_routes.append(current_petal)

        return disjoint_routes

    def k_sunflower_intersection_count(self, petals: Iterable[Set[CapitalNode]]) -> int:
        """Count pairwise intersections of (asset_id, venue_id) keys across petals — expect 0."""
        keys_per_petal = [
            {(n.asset_id, n.venue_id) for n in petal} for petal in petals
        ]
        intersections = 0
        for i, a in enumerate(keys_per_petal):
            for b in keys_per_petal[i + 1 :]:
                intersections += len(a & b)
        return intersections

    def execute_netting_matrix(self, nodes: List[CapitalNode]) -> dict:
        logger.info("[UTAH-SIEVE] Ingesting %d global liquidity nodes...", len(nodes))
        start = time.perf_counter()

        petals = self.apply_sunflower_lemma(nodes)
        intersection_count = self.k_sunflower_intersection_count(petals)

        summaries = []
        for idx, petal in enumerate(petals):
            total_liquidity = sum(n.liquidity_volume for n in petal)
            core_yield = sum(n.yield_velocity for n in petal) / len(petal)
            msg = (
                f"Routed Petal {idx}: {total_liquidity:.2f} units "
                f"at {core_yield * 100:.2f}% velocity."
            )
            logger.info("  -> %s", msg)
            summaries.append(
                {
                    "petal_index": idx,
                    "total_liquidity": total_liquidity,
                    "core_yield": core_yield,
                    "node_count": len(petal),
                }
            )

        elapsed = time.perf_counter() - start
        logger.info("[UTAH-SIEVE] Routing resolved in %.8f seconds.", elapsed)

        return {
            "petals": petals,
            "summaries": summaries,
            "elapsed_seconds": elapsed,
            "k_sunflower_intersection_count": intersection_count,
        }


# --------------------------------------------------------------------------- #
# Navier-Stokes Liquidity Routing
# --------------------------------------------------------------------------- #
def calculate_navier_stokes_rebalance_flow(
    current_weights,
    target_manifold,
    market_viscosity_tensor,
    kinematic_constraints: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Solve the (discretised, over-asset) momentum equation for capital flow.

    We treat the weight vector as a density on a graph of assets and seek a
    velocity field $v$ moving mass from ``current_weights`` toward
    ``target_manifold`` while (a) respecting viscous market impact and
    (b) conserving total capital (incompressibility / divergence-free flow).

    Momentum (steady-state, low-Reynolds Stokes limit):

        $$\nu\, L\, v = -\nabla p + f, \qquad \mathbf{1}^\top v = 0$$

    where $L$ is the graph Laplacian (diffusive coupling), $\nu$ the viscosity
    (slippage), $f = (\text{target} - \text{current})$ the alpha-driven body
    force, and $p$ the pressure enforcing mass conservation.

    Parameters
    ----------
    current_weights : array-like, shape (N,)
    target_manifold : array-like, shape (N,)
    market_viscosity_tensor : array-like, scalar, (N,) or (N, N)
        Per-asset / pairwise slippage. Scalar or vector is treated as diagonal.
    kinematic_constraints : dict
        Optional keys: ``max_velocity`` (clip speed), ``coupling`` (graph
        Laplacian weight, default 1.0).

    Returns
    -------
    (velocity_field, pressure_gradient) : Tuple[np.ndarray, np.ndarray]
        ``velocity_field`` (dv/dt over assets, mass-conserving) and the
        ``pressure_gradient`` that routes the arbitrage.
    """
    w0 = as_array(current_weights).ravel()
    wt = as_array(target_manifold).ravel()
    n = w0.shape[0]
    if wt.shape[0] != n:
        raise ValueError("current_weights and target_manifold must match length")

    visc = as_array(market_viscosity_tensor)
    if visc.ndim == 0:
        visc_mat = np.eye(n) * float(visc)
    elif visc.ndim == 1:
        visc_mat = np.diag(visc)
    else:
        visc_mat = visc
    visc_mat = visc_mat + 1e-9 * np.eye(n)

    coupling = float(kinematic_constraints.get("coupling", 1.0))
    # graph Laplacian for a fully-connected asset network (diffusive coupling)
    laplacian = coupling * (n * np.eye(n) - np.ones((n, n)))

    body_force = wt - w0  # alpha pressure gradient (desired transport)

    # Operator A = viscosity * Laplacian  (+ regularisation for invertibility)
    a_op = visc_mat @ laplacian + 1e-6 * np.eye(n)

    # Pressure solves the projection making the flow divergence-free (sum v = 0).
    # Solve A v = f - grad p with constraint 1^T v = 0 via a saddle-point system.
    ones = np.ones((n, 1))
    kkt = np.block([[a_op, ones], [ones.T, np.zeros((1, 1))]])
    rhs = np.concatenate([body_force, np.zeros(1)])
    sol, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
    velocity = sol[:n]
    lagrange_pressure = float(sol[n])

    # pressure gradient field = alpha force minus realised viscous transport
    pressure_gradient = body_force - a_op @ velocity

    max_v = kinematic_constraints.get("max_velocity")
    if max_v is not None:
        speed = np.linalg.norm(velocity)
        if speed > max_v and speed > 0:
            velocity = velocity * (float(max_v) / speed)

    dissipation = float(velocity @ (visc_mat @ velocity))
    logger.info(
        "[NAVIER] |v|=%.4f dissipation=%.6e p0=%.4e",
        float(np.linalg.norm(velocity)),
        dissipation,
        lagrange_pressure,
    )
    return velocity, pressure_gradient
