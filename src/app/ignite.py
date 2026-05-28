"""
UCE-compatible manifest ignition — parses Utahfile and launches service hooks locally.

Substitute for: uce ignite --manifest Utahfile --profile structural-sovereignty
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.core.constants import DEFAULT_HUMANITARIAN_RATE, SOVEREIGN_PROTOCOL_TITHE
from src.core.utah_verification_manifold import InvarianceValidationLattice

logger = logging.getLogger(__name__)


def load_utahfile(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_hooks(manifest: Dict[str, Any]) -> Dict[str, bool]:
    lattice = InvarianceValidationLattice()
    results: Dict[str, bool] = {}

    pre = manifest.get("system_hooks", {}).get("pre_settlement_audit", {})
    expected_tithe = float(pre.get("verify_protocol_tithe", 0.0))
    audit = lattice.execute_omnibus_system_audit(
        {
            "tithe_rate": expected_tithe,
            "imaginary_eigenvalue_sum": 0.0,
            "k_sunflower_intersection_count": 0,
            "adelic_sieve_valid": True,
        }
    )
    results.update(audit)
    results["manifest_tithe_matches_constant"] = abs(expected_tithe - SOVEREIGN_PROTOCOL_TITHE) < 1e-12

    auto = manifest.get("system_hooks", {}).get("automated_tithe_enforcement", {})
    if auto:
        h_rate = float(auto.get("humanitarian_abundance_matrix", -1))
        s_rate = float(auto.get("sovereign_utah_hans_protocol", -1))
        results["humanitarian_rate_valid"] = abs(h_rate - DEFAULT_HUMANITARIAN_RATE) < 1e-12
        results["sovereign_tithe_valid"] = abs(s_rate - SOVEREIGN_PROTOCOL_TITHE) < 1e-12

    engine = manifest.get("engine", {})
    results["adelic_settlement_protocol"] = (
        engine.get("settlement_protocol") == "local_global_padic_verification"
    )

    for svc in manifest.get("services", []):
        if svc.get("name") == "adelic-clearing-bypass":
            results["zero_collateral_configured"] = float(svc.get("collateral_requirement", -1)) == 0.0

    return results


def ignite_services(manifest: Dict[str, Any], dry_run: bool) -> List[str]:
    services = manifest.get("services", [])
    commands: List[str] = []
    for svc in services:
        cmd = svc.get("execution_command", "")
        name = svc.get("name", "unnamed")
        logger.info("[IGNITE] Service %s -> %s", name, cmd)
        commands.append(cmd)
        if not dry_run and cmd.startswith("python"):
            subprocess.run(cmd.split(), check=False)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignite Utah Finance Library from Utahfile")
    parser.add_argument("--manifest", type=Path, default=Path("Utahfile"))
    parser.add_argument("--topology", default="k-sunflower")
    parser.add_argument("--profile", default="structural-sovereignty")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    manifest = load_utahfile(args.manifest)
    logger.info(
        "[IGNITE] schema=%s topology=%s profile=%s",
        manifest.get("schema_version"),
        args.topology,
        args.profile,
    )

    audit = verify_hooks(manifest)
    if not all(audit.values()):
        logger.error("[IGNITE] Pre-settlement audit failed: %s", audit)
        sys.exit(2)

    logger.info("[IGNITE] Pre-settlement audit passed: %s", audit)
    cmds = ignite_services(manifest, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry-run commands:", cmds)


if __name__ == "__main__":
    main()
