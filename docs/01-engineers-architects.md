# Deploying Bounded Transfinite Infrastructure

**Audience:** Software engineers, DevOps, system architects  
**Version:** 6.Omnibus_Adelic

> **Languages:** English · [Eesti](et/01-engineers-architects.md) · [Русский](ru/01-engineers-architects.md)

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
| Tick observer | `src/core/tick_observer.py` | Pathwise quadratic-covariation metric `g_ij(t)` |
| Manifold kernel | `src/models/manifold_kernel.py` | Exact-autodiff Ricci-flow covariance denoising |
| Riemannian geometry | `src/models/riemannian_geometry.py` | Christoffel / Riemann / Ricci via `jax.jacfwd` |
| Topological allocation | `src/core/topological_allocation.py` | Persistent-homology risk parity + Betti divergence |
| Risk supervisor | `src/core/risk_supervisor.py` | Spectral CVaR + Laplace-Beltrami drawdown bound |
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

## Continuous-time geometry stack (JAX mandatory)

The v6 geometry modules replace static covariance estimation with exact
continuous-time differential geometry. The Ricci-flow denoiser, the
Christoffel/Riemann/Ricci tensors, and the Laplace-Beltrami drawdown bound are
**JAX-only by design** — there is no NumPy finite-difference fallback, because
`O(h^2)` truncation corrupts the non-linear flow PDE.

```python
import numpy as np
from src.core.tick_observer import QuadraticCovariationObserver
from src.models.manifold_kernel import compute_ricci_flow_covariance
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

# 1. Measure the pathwise metric tensor from ticks (zero look-back lag)
obs = QuadraticCovariationObserver(n_assets=N)
obs.ingest_prices(price_path, dt=1 / len(price_path))
g = obs.metric_tensor()                       # g_ij(t) = d/dt <X_i, X_j>_t

# 2. Denoise via the exact-autodiff normalized Ricci flow (no proxy in the path)
denoised = compute_ricci_flow_covariance(g, flow_duration=1.0, manifold_dimension=N)

# 3. Absolute drawdown veto from the measured metric
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

Validate exactness against analytic benchmarks:

```bash
pytest tests/test_riemannian_geometry.py   # 2-sphere R = 2/r^2 to 1e-5
pytest tests/test_ricci_flow_autodiff.py   # proxy excluded; strict denoising
pytest tests/test_tick_observer.py         # dt->0 convergence, drift invariance
```

Full theory: [08-continuous-time-allocation.md](08-continuous-time-allocation.md)
and the [Ricci-flow stabilization theorem](09_Ricci_Flow_Stabilization.tex).

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
pytest tests/test_pinn.py                  # PINN alpha (requires JAX extra)
pytest tests/test_riemannian_geometry.py   # exact curvature (requires JAX extra)
pytest tests/test_manifold_kernel.py tests/test_ricci_flow_autodiff.py
pytest tests/test_tick_observer.py tests/test_laplace_beltrami.py
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
