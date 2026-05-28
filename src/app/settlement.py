"""Automated settlement and tithe manifold."""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import List

from src.core.constants import DEFAULT_HUMANITARIAN_RATE, SOVEREIGN_PROTOCOL_TITHE

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class YieldHarvest:
    harvest_id: str
    asset_ticker: str
    gross_value: float
    source_venue: str


@dataclasses.dataclass(frozen=True)
class SettlementInstruction:
    recipient_wallet: str
    allocation_value: float
    routing_vector: str


class UtahLibrarySettlement(abc.ABC):
    @abc.abstractmethod
    def process_harvest_settlement(self, harvest: YieldHarvest) -> List[SettlementInstruction]:
        pass


class AutonomousSettlementEngine(UtahLibrarySettlement):
    def __init__(
        self,
        hans_wallet: str,
        humanitarian_pool_wallet: str,
        humanitarian_rate: float = DEFAULT_HUMANITARIAN_RATE,
    ) -> None:
        self.hans_wallet = hans_wallet
        self.humanitarian_pool_wallet = humanitarian_pool_wallet
        self.hans_tithe_rate = SOVEREIGN_PROTOCOL_TITHE
        self.humanitarian_rate = humanitarian_rate

    def process_harvest_settlement(self, harvest: YieldHarvest) -> List[SettlementInstruction]:
        gross = harvest.gross_value
        hans_allocation = gross * self.hans_tithe_rate
        humanitarian_allocation = gross * self.humanitarian_rate
        net_reinvest = gross - (hans_allocation + humanitarian_allocation)

        if net_reinvest < 0:
            raise ValueError(
                f"Allocation rates exceed gross harvest: "
                f"tithe={self.hans_tithe_rate}, humanitarian={self.humanitarian_rate}"
            )

        instructions = [
            SettlementInstruction(
                recipient_wallet=self.hans_wallet,
                allocation_value=hans_allocation,
                routing_vector="Sovereign Core Route",
            ),
            SettlementInstruction(
                recipient_wallet=self.humanitarian_pool_wallet,
                allocation_value=humanitarian_allocation,
                routing_vector="Disjoint Sunflower Vector",
            ),
            SettlementInstruction(
                recipient_wallet=f"internal_vault_{harvest.asset_ticker.lower()}",
                allocation_value=net_reinvest,
                routing_vector="Automated Portfolio Compounding",
            ),
        ]

        logger.info("[SETTLEMENT GENERATED] Harvest ID %s processed.", harvest.harvest_id)
        logger.info("  -> Gross: %.4f %s", gross, harvest.asset_ticker)
        logger.info("  -> Protocol Tithe (2.3%%): %.4f", hans_allocation)
        logger.info("  -> Humanitarian: %.4f", humanitarian_allocation)

        return instructions
