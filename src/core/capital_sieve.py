"""Continuous capital leakage auditor — capital-sieve / Utah-Prime-Sieve drag layer."""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AssetPosition:
    ticker: str
    venue_id: str
    allocation_volume: float
    idle_since_timestamp: int
    yield_rate: float


@dataclasses.dataclass(frozen=True)
class AllocationIntent:
    target_ticker: str
    source_venue: str
    destination_venue: str
    transfer_volume: float
    expected_leakage_reduction: float


class FinancialSieveEngine(abc.ABC):
    """Abstract baseline for multi-asset corporate and hedge fund auditing manifolds."""

    @abc.abstractmethod
    def audit_portfolio_drag(self, positions: List[AssetPosition]) -> List[AllocationIntent]:
        pass


class AutonomousAuditor(FinancialSieveEngine):
    """Detects cross-venue yield drag and emits rebalancing intents."""

    def __init__(
        self,
        efficiency_threshold: float,
        venue_risk_matrix: Dict[str, float],
    ) -> None:
        self.efficiency_threshold = efficiency_threshold
        self.venue_risk_matrix = venue_risk_matrix

    def _risk_adjusted_yield(self, position: AssetPosition) -> float:
        risk = self.venue_risk_matrix.get(position.venue_id, 0.5)
        return position.yield_rate * (1.0 - risk)

    def audit_portfolio_drag(self, positions: List[AssetPosition]) -> List[AllocationIntent]:
        intents: List[AllocationIntent] = []
        asset_groups: Dict[str, List[AssetPosition]] = {}

        for pos in positions:
            asset_groups.setdefault(pos.ticker, []).append(pos)

        for ticker, venues in asset_groups.items():
            if len(venues) < 2:
                continue

            optimal = max(venues, key=self._risk_adjusted_yield)

            for current in venues:
                if current.venue_id == optimal.venue_id:
                    continue

                yield_differential = optimal.yield_rate - current.yield_rate
                capital_drag = current.allocation_volume * yield_differential

                if capital_drag > self.efficiency_threshold:
                    intent = AllocationIntent(
                        target_ticker=ticker,
                        source_venue=current.venue_id,
                        destination_venue=optimal.venue_id,
                        transfer_volume=current.allocation_volume,
                        expected_leakage_reduction=capital_drag,
                    )
                    intents.append(intent)
                    logger.info(
                        "[AUDIT ALERT] Capital leakage in %s at %s. Drag: %.4f",
                        ticker,
                        current.venue_id,
                        capital_drag,
                    )

        return intents
