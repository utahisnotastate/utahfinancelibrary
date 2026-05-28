"""
Hasse-Minkowski daemon — Adelic Clearinghouse Bypass entrypoint.

Demonstrates zero-collateral atomic settlement when local-global verification passes.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from src.core.adelic_clearing import (
    AdelicClearinghouseEngine,
    AtomicTrade,
    VaultSnapshot,
)

logger = logging.getLogger(__name__)


def _demo_ledger() -> AdelicClearinghouseEngine:
    engine = AdelicClearinghouseEngine()
    engine.register_vault(
        VaultSnapshot(
            vault_id="fund_alpha",
            balances={"USD": 600_000_000.0, "WETH": 0.0},
        )
    )
    engine.register_vault(
        VaultSnapshot(
            vault_id="fund_beta",
            balances={"USD": 50_000_000.0, "WETH": 10_000.0},
        )
    )
    return engine


def run_demo(json_output: bool = False) -> Dict[str, Any]:
    engine = _demo_ledger()

    # Legacy rail: $500M margin would be locked at DTCC for this notional.
    trade = AtomicTrade(
        trade_id="T-ADELIC-2026-001",
        buyer_vault_id="fund_alpha",
        seller_vault_id="fund_beta",
        base_asset="WETH",
        quote_asset="USD",
        quantity=1_000.0,
        price=2_500.0,
    )

    result = engine.attempt_atomic_settlement(trade)
    collateral_avoided = engine.total_locked_collateral_avoided() if result.zero_collateral else 0.0

    payload: Dict[str, Any] = {
        "trade_id": trade.trade_id,
        "global_passed": result.global_passed,
        "zero_collateral": result.zero_collateral,
        "required_collateral": result.required_collateral,
        "settlement_hash": result.settlement_hash,
        "phase": result.phase.value,
        "legacy_margin_avoided_usd": collateral_avoided,
        "local_field_pass_count": sum(1 for r in result.local_results if r.passed),
        "local_field_total": len(result.local_results),
    }

    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        print("\n--- Adelic Clearinghouse Bypass ---")
        print(f"Trade: {trade.trade_id}")
        print(f"Global verification: {result.global_passed}")
        print(f"Collateral required: ${result.required_collateral:,.2f}")
        print(f"Legacy margin avoided (demo): ${collateral_avoided:,.2f}")
        print(f"Settlement hash: {result.settlement_hash[:32]}...")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Hasse-Minkowski zero-collateral settlement daemon")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_demo(json_output=args.json)


if __name__ == "__main__":
    main()
