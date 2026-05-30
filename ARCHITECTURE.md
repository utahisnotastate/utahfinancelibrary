# Architecture Overview

**Repository:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)  
**Version:** 6.Omnibus_Adelic

## System diagram

```mermaid
flowchart TB
    subgraph deploy [Deployment]
        UF[Utahfile v6]
        IG[ignite.py]
        UF --> IG
    end

    subgraph core [src/core]
        CS[capital_sieve]
        SR[sunflower_router]
        VL[utah_verification_manifold]
        AC[adelic_clearing]
    end

    subgraph app [src/app]
        PS[utah_prime_sieve_daemon]
        ST[settlement]
        HM[hasse_minkowski_daemon]
        AO[alpha_orchestrator]
        VD[vault_daemon]
    end

    subgraph models [src/models]
        WT[wave_theory_engine]
        PINN[pinn_jax_runtime]
        MK[manifold_kernel]
        RG[riemannian_geometry]
    end

    subgraph geo [Continuous-time geometry]
        TO[tick_observer]
        TA[topological_allocation]
        RS[risk_supervisor]
    end

    IG --> PS
    PS --> CS
    PS --> SR
    PS --> ST
    PS --> AC
    PS --> WT
    PS --> PINN
    HM --> AC

    TO -->|pathwise metric g_ij| MK
    TO -->|measured metric| RS
    TO -->|covariation| PINN
    MK --> RG
    TA --> RS
```

## Module responsibilities

| Layer | Module | Output |
|-------|--------|--------|
| FinOps audit | `AutonomousAuditor` | `AllocationIntent` list |
| Routing | `UtahTransfiniteSieve` | Disjoint capital petals |
| Settlement | `AutonomousSettlementEngine` | `SettlementInstruction` list |
| Adelic | `AdelicClearinghouseEngine` | Zero-collateral atomic ledger swap |
| Alpha | `OrthogonalWaveStatePredictor` | Wave-state predictions (JAX) |
| Risk | `AlphaOrchestrator` | Guarded execution queue |
| Custody | `SovereignVault` | Threshold-signed intents |
| Validation | `InvarianceValidationLattice` | Omnibus audit booleans |
| Pathwise metric | `QuadraticCovariationObserver` | Measured metric tensor `g_ij(t)` from ticks |
| Geometric denoising | `compute_ricci_flow_covariance` | Exact-autodiff Ricci-flow covariance |
| Exact curvature | `riemannian_geometry` | Christoffel / Riemann / Ricci via `jax.jacfwd` |
| Topological allocation | `optimize_topological_risk_parity` | Persistent-homology risk parity |
| Spectral risk wall | `feynman_kac_drawdown_bound` | Laplace-Beltrami drawdown supremum |

## Continuous-time geometry layer

The v6 geometry stack treats allocation as continuous-time differential geometry
rather than static quadratic programming. It **measures** the market's metric
tensor instead of modelling it:

- `tick_observer.py` — reads the pathwise quadratic covariation
  `g_ij(t) = d/dt ⟨X_i, X_j⟩_t` directly from the tick stream (zero look-back lag),
  with a noise-robust two-scale estimator for raw exchange feeds.
- `manifold_kernel.py` / `riemannian_geometry.py` — exact `jax.jacfwd` Ricci-flow
  covariance denoising; no finite differences.
- `topological_allocation.py` — persistent-homology risk parity and the
  Betti-number divergence crash diagnostic.
- `risk_supervisor.py` — Laplace-Beltrami principal eigenvalue → Feynman-Kac
  drawdown bound, consuming the measured metric.

See [docs/08-continuous-time-allocation.md](docs/08-continuous-time-allocation.md)
and [docs/09_Ricci_Flow_Stabilization.tex](docs/09_Ricci_Flow_Stabilization.tex).

## Data flow (demo pipeline)

1. Positions ingested → drag audit → migration intents  
2. Liquidity nodes → sunflower petals → netting summaries  
3. Yield harvest → tithe / humanitarian / reinvest splits  
4. Adelic trade → local-global verify → atomic settlement hash  
5. Optional PINN train/predict on synthetic features  

## Extension points

- Replace demo vaults with live PB/custody feeds  
- Persist `settlement_hash` to append-only storage  
- Wire `SovereignVault` to HSM/TSS providers  
- Deploy via [utahcontainerengine](https://github.com/utahisnotastate) when available  

See [docs/README.md](docs/README.md) for role-specific guides.
