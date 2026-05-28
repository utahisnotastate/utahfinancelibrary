# Deploying Bounded Transfinite Infrastructure

**Audience:** Software engineers, DevOps, system architects  
**Version:** 6.Omnibus_Adelic

## Overview

The Utah Finance Library is a **multi-module Python monorepo** that replaces monolithic fund stacks with deterministic, testable components:

| Module | Path | Responsibility |
|--------|------|----------------|
| Capital sieve | `src/core/capital_sieve.py` | Cross-venue yield drag → rebalancing intents |
| Sunflower router | `src/core/sunflower_router.py` | Disjoint liquidity petals ($k$-bound partitions) |
| Verification lattice | `src/core/utah_verification_manifold.py` | Navier alignment, adelic sieve bound, omnibus audit |
| Adelic clearing | `src/core/adelic_clearing.py` | Hasse-Minkowski local-global settlement verification |
| Settlement | `src/app/settlement.py` | Harvest splits (tithe + humanitarian + reinvest) |
| PINN alpha | `src/models/pinn_jax_runtime.py` | JAX physics-informed wave predictor |
| Wave telemetry | `src/models/wave_theory_engine.py` | Bounded ring-buffer tick statistics |
| Vault | `src/app/vault_daemon.py` | Threshold-signed intent broadcast |
| Orchestrator | `src/app/alpha_orchestrator.py` | Risk guardrails + circuit breaker |

## Prerequisites

- Python 3.10+
- Windows: use `py -3` if `python` is not on PATH
- Optional JAX stack: `pip install -e ".[jax,dev]"`

## Installation

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Repository layout

```text
/
├── Utahfile                 # Deployment manifest (v6)
├── .cursorrules             # Cursor AI engineering mandate
├── src/
│   ├── core/                # Mathematical + financial kernels
│   ├── app/                 # Daemons and CLI entrypoints
│   └── models/              # JAX PINN + wave telemetry
├── tests/                   # pytest suite
└── docs/                    # This documentation set
```

## Utahfile v6

The `Utahfile` is the single deployment manifest. Key v6 additions:

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    collateral_requirement: 0.00
```

`src/app/ignite.py` validates:

- Pre-settlement tithe `0.023`
- Automated humanitarian rate `0.057`
- Zero collateral on adelic service
- Adelic settlement protocol string

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
```

## OrthogonalWaveStatePredictor (JAX / XLA)

The alpha engine lives in `src/models/pinn_jax_runtime.py`.

**Forward pass** uses orthogonal phase-lock activation:

$$\text{hidden} = \sin(z) \cdot e^{-z^2/2}, \quad z = x W_1 + b_1$$

**Loss** combines MSE with a vorticity penalty derived from the batch Jacobian:

$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda \cdot \mathbb{E}[|\text{trace}(J)|]$$

Compile for deployment:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor, PINNTrainConfig

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()
params, predict = predictor.train(market_x, alpha_y, PINNTrainConfig(steps=200))
```

## Adelic clearing integration

```python
from src.core.adelic_clearing import AdelicClearinghouseEngine, AtomicTrade, VaultSnapshot

engine = AdelicClearinghouseEngine()
engine.register_vault(VaultSnapshot("buyer", {"USD": 1e9}))
engine.register_vault(VaultSnapshot("seller", {"WETH": 500}))
result = engine.attempt_atomic_settlement(AtomicTrade(...))
assert result.zero_collateral  # when local-global checks pass
```

## Verification lattice

Run standalone:

```bash
python -m src.core.utah_verification_manifold
```

Checks include:

1. **Navier geometric depletion** — vorticity vs intermediate eigenvector of strain tensor
2. **Adelic sieve interval** — $Y(x) = \Theta(x \log x \log \log x)$ vs $O(x^2)$ legacy ceiling
3. **Omnibus audit** — tithe rate, spectral rigidity, disjoint routing

## Testing

```bash
pytest -q
pytest tests/test_pinn.py  # requires JAX extra
```

## External ecosystem

Designed to integrate with [github.com/utahisnotastate](https://github.com/utahisnotastate):

- `utahcontainerengine` — production unikernel runtime (local substitute: `ignite.py`)
- `wavetheory_5` — telemetry patterns mirrored in `wave_theory_engine.py`

## Production checklist

1. Replace demo vaults/positions with API feeds
2. Persist settlement hashes and audit trail to immutable storage
3. Wire `SovereignVault` to real TSS/HSM signing
4. Run compliance review before claiming zero clearing margin
5. Pin dependency versions in production images
