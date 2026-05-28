"""k-sunflower disjoint capital routing — transfinite sieve topology layer."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Iterable, List, Set

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
