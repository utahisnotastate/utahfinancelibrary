# Utah Finance Library — Documentation Index

**Version 6.Omnibus_Adelic**  
**Repository:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)

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

## Important disclaimer

The Utah Finance Library provides **open-source software building blocks** for portfolio auditing, routing, settlement instruction generation, and mathematical verification. It does **not** by itself:

- Replace licensed clearinghouses or regulatory reporting
- Guarantee investment returns or risk elimination
- Constitute legal, tax, or investment advice

Production use requires your own compliance, legal review, and integration with approved financial infrastructure.
