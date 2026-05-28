import pytest

from src.core.adelic_clearing import (
    AdelicClearinghouseEngine,
    AtomicTrade,
    SettlementPhase,
    VaultSnapshot,
)


def test_zero_collateral_when_solvency_holds():
    engine = AdelicClearinghouseEngine()
    engine.register_vault(VaultSnapshot("buyer", {"USD": 10_000_000.0}))
    engine.register_vault(VaultSnapshot("seller", {"WETH": 100.0}))

    trade = AtomicTrade("t1", "buyer", "seller", "WETH", "USD", 10.0, 2500.0)
    result = engine.attempt_atomic_settlement(trade)

    assert result.zero_collateral
    assert result.phase == SettlementPhase.ATOMIC_SETTLED
    assert engine.get_vault("buyer").available("WETH") == 10.0
    assert engine.get_vault("seller").available("USD") == 25_000.0


def test_collateral_required_when_insolvent():
    engine = AdelicClearinghouseEngine()
    engine.register_vault(VaultSnapshot("buyer", {"USD": 1.0}))
    engine.register_vault(VaultSnapshot("seller", {"WETH": 100.0}))

    trade = AtomicTrade("t2", "buyer", "seller", "WETH", "USD", 10.0, 2500.0)
    result = engine.attempt_atomic_settlement(trade)

    assert not result.global_passed
    assert result.required_collateral > 0
    assert result.phase == SettlementPhase.REJECTED
