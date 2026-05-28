"""
Alpha orchestrator — event-driven execution hooks with orthogonal risk guardrails.

Full HFT runtime is out of scope here; this module defines the risk-isolated intent
interface consumed by the netting and settlement layers.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"


@dataclasses.dataclass(frozen=True)
class ExecutionIntent:
    strategy_id: str
    symbol: str
    side: str
    quantity: float
    limit_price: Optional[float] = None


@dataclasses.dataclass
class RiskGuardrail:
    max_notional: float
    max_orders_per_second: int
    circuit_state: CircuitState = CircuitState.CLOSED

    def validate(self, intent: ExecutionIntent) -> bool:
        if self.circuit_state == CircuitState.OPEN:
            logger.warning("[CIRCUIT BREAKER] Rejecting intent %s", intent.strategy_id)
            return False
        notional = intent.quantity * (intent.limit_price or 0.0)
        if intent.limit_price and notional > self.max_notional:
            logger.warning("[RISK] Notional %.2f exceeds cap %.2f", notional, self.max_notional)
            return False
        return True

    def trip_circuit(self) -> None:
        self.circuit_state = CircuitState.OPEN
        logger.error("[CIRCUIT BREAKER] Execution layer halted.")


class AlphaOrchestrator:
    """Routes validated intents to an injected executor (simulation-friendly)."""

    def __init__(self, guardrail: RiskGuardrail) -> None:
        self.guardrail = guardrail
        self._queue: List[ExecutionIntent] = []

    def submit(self, intent: ExecutionIntent) -> bool:
        if not self.guardrail.validate(intent):
            return False
        self._queue.append(intent)
        logger.info("[ORCHESTRATOR] Queued %s %s %s", intent.side, intent.quantity, intent.symbol)
        return True

    def drain(self, executor: Callable[[ExecutionIntent], None]) -> int:
        count = 0
        while self._queue:
            intent = self._queue.pop(0)
            executor(intent)
            count += 1
        return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    guard = RiskGuardrail(max_notional=1_000_000.0, max_orders_per_second=100)
    orch = AlphaOrchestrator(guard)
    orch.submit(
        ExecutionIntent(
            strategy_id="demo_mean_revert",
            symbol="USD",
            side="BUY",
            quantity=1000.0,
            limit_price=1.0,
        )
    )
    orch.drain(lambda i: logger.info("[EXEC] Filled %s", i.symbol))


if __name__ == "__main__":
    main()
