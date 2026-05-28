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
    end

    IG --> PS
    PS --> CS
    PS --> SR
    PS --> ST
    PS --> AC
    PS --> WT
    PS --> PINN
    HM --> AC
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
