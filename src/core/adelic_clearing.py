"""
Hasse-Minkowski zero-collateral settlement — Adelic Clearinghouse Bypass.

Implements a local-global solvency model inspired by the Hasse principle:
if a trade is solvent in the real (balance) field and across configured p-adic
localizations, it may settle atomically without central clearing margin.

This is a programmatic settlement *verification* layer — not a replacement for
regulated clearing without legal/compliance review in production.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default primes for local (_p-adic) consistency checks.
DEFAULT_PRIMES: Tuple[int, ...] = (2, 3, 5, 7, 11, 13)


@dataclasses.dataclass(frozen=True)
class VaultSnapshot:
    vault_id: str
    balances: Dict[str, float]  # asset -> quantity

    def available(self, asset: str) -> float:
        return float(self.balances.get(asset, 0.0))


@dataclasses.dataclass(frozen=True)
class AtomicTrade:
    trade_id: str
    buyer_vault_id: str
    seller_vault_id: str
    base_asset: str  # asset sold by seller
    quote_asset: str  # asset paid by buyer
    quantity: float
    price: float

    @property
    def quote_notional(self) -> float:
        return self.quantity * self.price


class SettlementPhase(enum.Enum):
    PENDING = "pending"
    ADELIC_VERIFIED = "adelic_verified"
    ATOMIC_SETTLED = "atomic_settled"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class LocalFieldResult:
    prime: int
    buyer_solvent: bool
    seller_solvent: bool

    @property
    def passed(self) -> bool:
        return self.buyer_solvent and self.seller_solvent


@dataclasses.dataclass(frozen=True)
class HasseVerificationResult:
    trade_id: str
    real_field_passed: bool
    local_results: Tuple[LocalFieldResult, ...]
    global_passed: bool
    required_collateral: float
    settlement_hash: str
    phase: SettlementPhase
    notional_settled: float = 0.0

    @property
    def zero_collateral(self) -> bool:
        return self.global_passed and self.required_collateral == 0.0


def _to_scaled_int(value: float, scale: int = 1_000_000) -> int:
    return int(round(value * scale))


class HasseMinkowskiVerifier:
    """
    Verifies counterparty solvency locally (mod p) and globally (real balances).

    Global pass: all local fields pass AND real-field balances cover obligations.
    """

    def __init__(
        self,
        primes: Sequence[int] = DEFAULT_PRIMES,
        amount_scale: int = 1_000_000,
    ) -> None:
        self.primes = tuple(primes)
        self.amount_scale = amount_scale

    def _local_obligation_ok(
        self,
        available: float,
        obligation: float,
        prime: int,
    ) -> bool:
        """Local solvency: real dominance, plus modular consistency on scaled integers."""
        if obligation <= 0:
            return True
        if available + 1e-12 >= obligation:
            return True
        a = _to_scaled_int(available, self.amount_scale)
        o = _to_scaled_int(obligation, self.amount_scale)
        return (a - o) % prime == 0

    def _real_field_check(
        self,
        buyer: VaultSnapshot,
        seller: VaultSnapshot,
        trade: AtomicTrade,
    ) -> bool:
        buyer_ok = buyer.available(trade.quote_asset) >= trade.quote_notional
        seller_ok = seller.available(trade.base_asset) >= trade.quantity
        return buyer_ok and seller_ok

    def _local_field_checks(
        self,
        buyer: VaultSnapshot,
        seller: VaultSnapshot,
        trade: AtomicTrade,
    ) -> Tuple[LocalFieldResult, ...]:
        results: List[LocalFieldResult] = []
        for p in self.primes:
            buyer_ok = self._local_obligation_ok(
                buyer.available(trade.quote_asset),
                trade.quote_notional,
                p,
            )
            seller_ok = self._local_obligation_ok(
                seller.available(trade.base_asset),
                trade.quantity,
                p,
            )
            results.append(
                LocalFieldResult(prime=p, buyer_solvent=buyer_ok, seller_solvent=seller_ok)
            )
        return tuple(results)

    @staticmethod
    def _settlement_hash(trade: AtomicTrade, global_passed: bool) -> str:
        payload = {
            "trade_id": trade.trade_id,
            "buyer": trade.buyer_vault_id,
            "seller": trade.seller_vault_id,
            "base": trade.base_asset,
            "quote": trade.quote_asset,
            "qty": trade.quantity,
            "price": trade.price,
            "global_passed": global_passed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def verify(
        self,
        trade: AtomicTrade,
        buyer: VaultSnapshot,
        seller: VaultSnapshot,
    ) -> HasseVerificationResult:
        real_ok = self._real_field_check(buyer, seller, trade)
        local = self._local_field_checks(buyer, seller, trade)
        local_ok = all(r.passed for r in local)
        global_passed = real_ok and local_ok
        collateral = 0.0 if global_passed else max(trade.quote_notional, trade.quantity)

        phase = (
            SettlementPhase.ADELIC_VERIFIED
            if global_passed
            else SettlementPhase.REJECTED
        )
        digest = self._settlement_hash(trade, global_passed)

        if global_passed:
            logger.info(
                "[ADELIC] Trade %s verified — zero collateral settlement permitted.",
                trade.trade_id,
            )
        else:
            logger.warning(
                "[ADELIC] Trade %s failed local-global checks. Collateral required: %.2f",
                trade.trade_id,
                collateral,
            )

        return HasseVerificationResult(
            trade_id=trade.trade_id,
            real_field_passed=real_ok,
            local_results=local,
            global_passed=global_passed,
            required_collateral=collateral,
            settlement_hash=digest,
            phase=phase,
            notional_settled=0.0,
        )


class AdelicClearinghouseEngine:
    """Executes atomic settlement after Hasse verification (in-memory ledger)."""

    def __init__(self, verifier: Optional[HasseMinkowskiVerifier] = None) -> None:
        self.verifier = verifier or HasseMinkowskiVerifier()
        self._vaults: Dict[str, VaultSnapshot] = {}
        self._audit_trail: List[HasseVerificationResult] = []

    def register_vault(self, vault: VaultSnapshot) -> None:
        self._vaults[vault.vault_id] = vault

    def get_vault(self, vault_id: str) -> VaultSnapshot:
        if vault_id not in self._vaults:
            raise KeyError(f"Unknown vault: {vault_id}")
        return self._vaults[vault_id]

    def attempt_atomic_settlement(self, trade: AtomicTrade) -> HasseVerificationResult:
        buyer = self.get_vault(trade.buyer_vault_id)
        seller = self.get_vault(trade.seller_vault_id)
        result = self.verifier.verify(trade, buyer, seller)
        self._audit_trail.append(result)

        if not result.global_passed:
            return result

        # Atomic swap: execution and settlement are the same event.
        new_buyer_bal = dict(buyer.balances)
        new_seller_bal = dict(seller.balances)

        new_buyer_bal[trade.quote_asset] = (
            new_buyer_bal.get(trade.quote_asset, 0.0) - trade.quote_notional
        )
        new_buyer_bal[trade.base_asset] = (
            new_buyer_bal.get(trade.base_asset, 0.0) + trade.quantity
        )

        new_seller_bal[trade.base_asset] = (
            new_seller_bal.get(trade.base_asset, 0.0) - trade.quantity
        )
        new_seller_bal[trade.quote_asset] = (
            new_seller_bal.get(trade.quote_asset, 0.0) + trade.quote_notional
        )

        self._vaults[trade.buyer_vault_id] = VaultSnapshot(trade.buyer_vault_id, new_buyer_bal)
        self._vaults[trade.seller_vault_id] = VaultSnapshot(trade.seller_vault_id, new_seller_bal)

        settled = HasseVerificationResult(
            trade_id=result.trade_id,
            real_field_passed=result.real_field_passed,
            local_results=result.local_results,
            global_passed=True,
            required_collateral=0.0,
            settlement_hash=result.settlement_hash,
            phase=SettlementPhase.ATOMIC_SETTLED,
            notional_settled=trade.quote_notional,
        )
        self._audit_trail[-1] = settled
        logger.info("[ADELIC] Atomic settlement committed hash=%s", settled.settlement_hash[:16])
        return settled

    @property
    def audit_trail(self) -> List[HasseVerificationResult]:
        return list(self._audit_trail)

    def total_locked_collateral_avoided(self) -> float:
        return sum(
            t.notional_settled
            for t in self._audit_trail
            if t.phase == SettlementPhase.ATOMIC_SETTLED
        )
