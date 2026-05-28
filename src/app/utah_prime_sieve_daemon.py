"""
Utah-Prime-Sieve execution runtime — drag audit, sunflower routing, settlement pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List

from src.core.capital_sieve import AssetPosition, AutonomousAuditor
from src.core.constants import DEFAULT_HUMANITARIAN_RATE, SOVEREIGN_PROTOCOL_TITHE
from src.core.sunflower_router import CapitalNode, UtahTransfiniteSieve
from src.core.utah_verification_manifold import InvarianceValidationLattice
from src.app.settlement import AutonomousSettlementEngine, YieldHarvest
from src.app.hasse_minkowski_daemon import run_demo as run_adelic_demo
from src.models.wave_theory_engine import TickSample, WaveTelemetryEngine

logger = logging.getLogger(__name__)


def _demo_positions() -> List[AssetPosition]:
    return [
        AssetPosition("USD", "prime_custody_01", 10_000_000.0, 1716900000, 0.042),
        AssetPosition("USD", "high_velocity_venue_02", 5_000_000.0, 1716910000, 0.051),
        AssetPosition("USD", "yield_aggregator_03", 2_000_000.0, 1716915000, 0.058),
    ]


def _demo_capital_nodes() -> List[CapitalNode]:
    return [
        CapitalNode("USDC", "utah_dex_01", 500_000.0, 0.05, 0.001),
        CapitalNode("USDT", "prime_broker_A", 1_200_000.0, 0.04, 0.005),
        CapitalNode("DAI", "yield_farm_sigma", 300_000.0, 0.08, 0.02),
        CapitalNode("WETH", "utah_darkpool", 2_500_000.0, 0.06, 0.0001),
    ]


def _demo_risk_profiles() -> Dict[str, float]:
    return {
        "prime_custody_01": 0.02,
        "high_velocity_venue_02": 0.15,
        "yield_aggregator_03": 0.30,
        "utah_dex_01": 0.10,
        "prime_broker_A": 0.05,
        "yield_farm_sigma": 0.25,
        "utah_darkpool": 0.08,
    }


async def run_pipeline(
    efficiency_threshold: float = 500.0,
    k_bound: int = 4,
    json_output: bool = False,
) -> Dict[str, Any]:
    risk = _demo_risk_profiles()

    auditor = AutonomousAuditor(
        efficiency_threshold=efficiency_threshold,
        venue_risk_matrix=risk,
    )
    migration_intents = auditor.audit_portfolio_drag(_demo_positions())

    sieve = UtahTransfiniteSieve(k_bound=k_bound)
    netting = sieve.execute_netting_matrix(_demo_capital_nodes())

    settlement = AutonomousSettlementEngine(
        hans_wallet="0xUtahHansSovereignVaultKeyManifold",
        humanitarian_pool_wallet="0xProHumanitarianAbundanceMatrixNode",
        humanitarian_rate=DEFAULT_HUMANITARIAN_RATE,
    )
    harvest = YieldHarvest(
        harvest_id="H-2026-OMNIBUS-01",
        asset_ticker="USD",
        gross_value=2_500_000.0,
        source_venue="utah_darkpool_alpha",
    )
    settlements = settlement.process_harvest_settlement(harvest)

    bound, sieve_valid = InvarianceValidationLattice().verify_adelic_sieve_interval(1000.0)
    audit = InvarianceValidationLattice().execute_omnibus_system_audit(
        {
            "tithe_rate": SOVEREIGN_PROTOCOL_TITHE,
            "imaginary_eigenvalue_sum": 0.0,
            "k_sunflower_intersection_count": netting["k_sunflower_intersection_count"],
            "adelic_sieve_valid": sieve_valid,
        }
    )

    adelic = run_adelic_demo(json_output=json_output)
    wave = WaveTelemetryEngine(capacity=128)
    for i, price in enumerate([100.0, 100.5, 99.8, 100.2, 101.0]):
        wave.ingest(TickSample(timestamp_ns=i, price=price, volume=1.0))

    alpha_signal: Dict[str, Any] = {"wave_z_score": wave.z_score(101.0), "pinn": "skipped"}
    try:
        from src.models.pinn_jax_runtime import (
            JAX_AVAILABLE,
            OrthogonalWaveStatePredictor,
            PINNTrainConfig,
        )

        if JAX_AVAILABLE:
            import numpy as np

            rng = np.random.default_rng(0)
            x = rng.normal(size=(64, 4)).astype(np.float32)
            y = (0.2 * x[:, 0] + 0.1 * x[:, 1]).astype(np.float32)[:, None]
            _, pred = OrthogonalWaveStatePredictor().train(
                x, y, PINNTrainConfig(steps=20, hidden_dim=16)
            )
            alpha_signal = {
                "wave_z_score": wave.z_score(101.0),
                "pinn_sample_pred": pred(x[:1]).tolist(),
            }
    except Exception as exc:  # pragma: no cover
        alpha_signal["pinn_error"] = str(exc)

    result: Dict[str, Any] = {
        "migration_intents": [intent.__dict__ for intent in migration_intents],
        "netting_summaries": netting["summaries"],
        "k_sunflower_intersection_count": netting["k_sunflower_intersection_count"],
        "settlements": [s.__dict__ for s in settlements],
        "adelic_bound_sample": bound,
        "omnibus_audit": audit,
        "adelic_clearing": adelic,
        "alpha_signal": alpha_signal,
    }

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nExecution pipeline complete. Intents: {len(migration_intents)}")
        print(f"Omnibus audit: {audit}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Utah-Prime-Sieve daemon")
    parser.add_argument("--threshold", type=float, default=500.0)
    parser.add_argument("--k-bound", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(
        run_pipeline(
            efficiency_threshold=args.threshold,
            k_bound=args.k_bound,
            json_output=args.json,
        )
    )


if __name__ == "__main__":
    main()
