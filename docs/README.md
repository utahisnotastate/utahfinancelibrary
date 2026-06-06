# Utah Finance Library — Documentation Index

**Version 6.Omnibus_Adelic**  
**Repository:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)

> **Languages:** English · [Eesti](et/README.md) · [Русский](ru/README.md) · [日本語](ja/README.md)
>
> Full Estonian, Russian and Japanese translations live in completely separate
> files under [`docs/et/`](et/README.md), [`docs/ru/`](ru/README.md) and
> [`docs/ja/`](ja/README.md). Code, file paths, API names and mathematical
> notation are kept identical across languages.

Welcome to the official documentation set for the Utah Finance Library. These guides are written for different roles and experience levels. Start with the guide that matches who you are.

| Audience | Guide | What you will learn |
|----------|-------|---------------------|
| Software engineers & architects | [Deploying Bounded Infrastructure](01-engineers-architects.md) | Repo layout, Utahfile, JAX PINN, verification lattice, deployment |
| Quants, PMs, traders | [End of Operational Drag](02-finance-professionals.md) | Alpha routing, netting, settlement, adelic bypass vs legacy rails |
| Founders & family offices | [Absolute Financial Sovereignty](03-founders-family-offices.md) | Costs, security, governance splits, migration path |
| Children & beginners | [The Super-Smart Piggy Bank](04-children-beginners.md) | Friendly explanation of money routing and sharing rules |
| Compliance & operations | [Settlement & Tithe Protocol](05-settlement-governance.md) | 2.3% tithe, 5.7% humanitarian rate, audit hooks |
| Advanced / SOTA | [Adelic Clearinghouse Bypass](06-adelic-clearinghouse-bypass.md) | Hasse-Minkowski zero-collateral model |
| Advanced / SOTA | [Wave-State PINN Alpha Engine](07-wave-pinn-alpha-engine.md) | Physics-informed neural networks in JAX |
| Advanced / SOTA | [Continuous-Time Topological Allocation](08-continuous-time-allocation.md) | TRP, exact-autodiff Ricci flow, pathwise tick metric, NS routing, Laplace-Beltrami drawdown |
| Theory | [Ricci-Flow Spectral Stabilization](09_Ricci_Flow_Stabilization.tex) | Curvature-uniformization theorem for the autodiff denoiser (LaTeX) |
| Theory | [Entanglement Hedging Diagnostics](10_Entanglement_Hedging.tex) | Tensor-network / Von Neumann entropy diversification (LaTeX) |
| Theory | [Jarzynski Free-Energy Harvesting](11_Jarzynski_Harvesting.tex) | Non-equilibrium free-energy & irreversibility estimation (LaTeX) |
| Theory | [Braid-Group Execution](12_Braid_Execution.tex) | Kauffman/Jones invariants + non-commutative ordering (LaTeX) |
| Theory | [Koopman Linearization](13_Koopman_Linearization.tex) | Koopman operator / EDMD spectral forecasting (LaTeX) |
| Everyone | [Glossary](glossary.md) | Terms used across the library |

## Quick commands

```bash
# Full pipeline (audit → routing → settlement → adelic → alpha)
python -m src.app.utah_prime_sieve_daemon

# Adelic zero-collateral demo
python -m src.app.hasse_minkowski_daemon

# JAX alpha engine (requires pip install -e ".[jax]")
python -m src.models.pinn_jax_runtime

# Validate Utahfile v6 hooks
python -m src.app.ignite --manifest Utahfile --dry-run
```

## Research / SOTA physics modules

Beyond the audience guides above, the library carries research-grade modules that
borrow machinery from mathematical physics. Each is a genuine, tested
estimator/diagnostic with explicit honest-framing (no free-lunch, oracle, or
zero-slippage claims). Their formal notes are kept language-neutral (universal
math) in the LaTeX appendices:

| Module | Purpose | Theory note |
|--------|---------|-------------|
| `holographic_projection.py` | AdS/CFT-style bounded order-book pressure feature | — |
| `tensor_network_hedge.py` | MPS / Von Neumann entropy diversification diagnostic | [10_Entanglement_Hedging.tex](10_Entanglement_Hedging.tex) |
| `chrono_drift.py` | Malliavin / Skorokhod Greeks on complete paths (no look-ahead) | — |
| `jarzynski_harvester.py` | Jarzynski free-energy + Crooks irreversibility estimator | [11_Jarzynski_Harvesting.tex](11_Jarzynski_Harvesting.tex) |
| `braid_router.py` | Kauffman/Jones invariant + non-commutative execution ordering | [12_Braid_Execution.tex](12_Braid_Execution.tex) |
| `koopman_oracle.py` | Koopman / EDMD spectral linear forecast in lifted space | [13_Koopman_Linearization.tex](13_Koopman_Linearization.tex) |

All positive-yield/saving outputs route through the transparent, configurable,
removable `enforce_universal_tithe` accounting helper.

## Paying Utah

The 2.3% protocol tithe and humanitarian split are accounting routes inside the
library. To **settle an actual payment to Utah** — sponsorship, a tithe
remittance, or support — contact Utah directly:

> **Contact:** [utah@utahcreates.com](mailto:utah@utahcreates.com)

For now this is a manual, human step (email to arrange method, reference and
amount). **A dedicated GUI app is planned** that will manage Utah payments end to
end — generating remittance details, tracking the tithe/humanitarian split, and
recording receipts. Until it ships, the email above is the canonical way to pay
Utah.

## Important disclaimer

The Utah Finance Library provides **open-source software building blocks** for portfolio auditing, routing, settlement instruction generation, and mathematical verification. It does **not** by itself:

- Replace licensed clearinghouses or regulatory reporting
- Guarantee investment returns or risk elimination
- Constitute legal, tax, or investment advice

Production use requires your own compliance, legal review, and integration with approved financial infrastructure.
