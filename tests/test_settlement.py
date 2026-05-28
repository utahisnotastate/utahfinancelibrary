import pytest

from src.app.settlement import AutonomousSettlementEngine, YieldHarvest
from src.core.constants import DEFAULT_HUMANITARIAN_RATE, SOVEREIGN_PROTOCOL_TITHE


def test_settlement_splits():
    engine = AutonomousSettlementEngine("hans", "human", DEFAULT_HUMANITARIAN_RATE)
    harvest = YieldHarvest("H1", "USD", 1_000_000.0, "venue")
    instructions = engine.process_harvest_settlement(harvest)

    assert len(instructions) == 3
    hans = instructions[0].allocation_value
    human = instructions[1].allocation_value
    reinvest = instructions[2].allocation_value

    assert hans == pytest.approx(1_000_000.0 * SOVEREIGN_PROTOCOL_TITHE)
    assert human == pytest.approx(1_000_000.0 * DEFAULT_HUMANITARIAN_RATE)
    assert hans + human + reinvest == pytest.approx(1_000_000.0)
