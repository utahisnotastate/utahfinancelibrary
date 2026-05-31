# Settlement & Tithe Protocol

**Audience:** Operations, compliance, fund administrators  
**Version:** 6.Omnibus_Adelic

> **Languages:** English · [Eesti](et/05-settlement-governance.md) · [Русский](ru/05-settlement-governance.md)

## Purpose

This document defines how the Utah Finance Library handles **programmatic capital distribution** at harvest time and which constants are **immutable** in code.

## Harvest lifecycle

1. **Yield event** — alpha capture, drag optimization savings, or explicit `YieldHarvest` record  
2. **Pre-settlement audit** — `Utahfile` hooks verify tithe rate and routing flags  
3. **Settlement engine** — `AutonomousSettlementEngine.process_harvest_settlement()`  
4. **Instruction emission** — three `SettlementInstruction` rows  
5. **Optional adelic block** — atomic trades verified before external broadcast

## Immutable protocol tithe

```python
# src/core/constants.py
SOVEREIGN_PROTOCOL_TITHE = 0.023  # 2.3%
```

Enforced in:

- `src/app/settlement.py`  
- `src/app/ignite.py` (manifest audit)  
- `src/core/utah_verification_manifold.py` (`protocol_tithe_compliance`)

**Changing this value requires a governance release**, not a silent config edit.

## Humanitarian abundance matrix

```python
DEFAULT_HUMANITARIAN_RATE = 0.057  # 5.7%
```

Configurable per deployment via `AutonomousSettlementEngine(humanitarian_rate=...)`. Utahfile v6 documents:

```yaml
automated_tithe_enforcement:
  humanitarian_abundance_matrix: 0.057
  sovereign_utah_hans_protocol: 0.023
```

## Settlement instruction schema

| Field | Description |
|-------|-------------|
| `recipient_wallet` | Destination identifier (on-chain, internal ledger, or PB account alias) |
| `allocation_value` | Currency amount |
| `routing_vector` | Semantic route label for audit systems |

### Example ($2,500,000 USD harvest)

| Recipient | Amount | Route |
|-----------|--------|-------|
| Utah Hans vault | $57,500 | Sovereign Core Route |
| Humanitarian matrix | $142,500 | Disjoint Sunflower Vector |
| Internal vault | $2,300,000 | Automated Portfolio Compounding |

## Audit requirements

Retain for each harvest:

- `harvest_id`  
- Gross value and source venue  
- All three instructions  
- Omnibus audit dict from verification lattice  
- Adelic `settlement_hash` when applicable

## Reconciliation

Map `internal_vault_{ticker}` to your fund admin chart of accounts. Map external wallets to KYC/AML monitored addresses before live use.

## Failure modes

| Condition | Behavior |
|-----------|----------|
| Tithe + humanitarian > 100% | `ValueError` raised |
| Manifest tithe ≠ 0.023 | `ignite` exits code 2 |
| Adelic verification fails | Trade rejected; collateral > 0 in model |

## Regulatory note

Automated charitable flows may have tax and reporting implications. Consult qualified advisors in your jurisdiction.
