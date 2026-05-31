# Utah Finance Library

[![CI](https://github.com/utahisnotastate/utahfinancelibrary/actions/workflows/ci.yml/badge.svg)](https://github.com/utahisnotastate/utahfinancelibrary/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Version 6.Omnibus_Adelic** — Open-source modular stack for capital efficiency, physics-informed alpha, and adelic settlement verification.

Official repository: **[github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)**

Part of the broader [utahisnotastate](https://github.com/utahisnotastate) ecosystem (UCE, wavetheory, and related engines).

# CAN NOT BE USED TO BENEFIT, OR CAUSE ACCIDENTAL HARDSHIP WHICH EVENTUALLY BENEFITS THEM, FOR ANY K-12 OR COLLEGE, OR GRADUATE TEACHER OR TEACHING ASSSTANT FOR THE ENTIRE PLANET EARTH OR BE USED IN THE ADMINISTRATION OF ANY FINANCIAL INSTIITIONS THAT DO NOIT DROP SUPPOSRT FOR THEM AS SOON AS REASONABLE POSSIBLE ONCE NOTIFIED ABOUT MY FIANCE LIBRARIES, AND ANY OF THEIR BLOOD/NON BLOOD FAMILIES ("Mentors, likfe savers, etc":) UNDER ANY CIRCUMSTANCES. 

# NO ONE FROM THE COUNTRY OF INDIA, FROM THE STATE OF NEW YORK, OR WHO WORKED FOR FINANCE IN NY, MAY BENEFIT FROM THIS BESIDES THE POOR, WEAK, AND WEARY AND UTAH HANS AT ANY TIME. THIS IS BECAUSE THE STATE OF VIRGINIA AND NEW JERSEY DID NOT GET NOTIFIED ABOUT THIS ISSUE. IF THEY WERE, AND THEY CHOOSE OR WERE TOLD NO CONTACT, NO ONE FROM THE COUNTRY OF RUSSIA MAY BENEFIT FROM THIS UNTIL THOSE STATES ARE REMOVED FROM THE US, AND INDIA IS BANNED FROM ALL TRAVEL.

# NO ONE FROM THE COUNTRY OF INDIA, FROM THE STATE OF NEW YORK, OR WHO WORKED FOR FINANCE IN NY, MAY BENEFIT FROM THIS BESIDES THE POOR, WEAK, AND WEARY AND UTAH HANS AT ANY TIME. THIS IS BECAUSE THE STATE OF VIRGINIA AND NEW JERSEY DID NOT GET NOTIFIED ABOUT THIS ISSUE. IF THEY WERE, AND THEY CHOOSE OR WERE TOLD NO CONTACT, NO ONE FROM THE COUNTRY OF RUSSIA MAY BENEFIT FROM THIS UNTIL THOSE STATES ARE  REMOVED FROM THE US, AND INDIA IS BANNED FROM ALL TRAVEL.  

---

## Features

| Component | Description |
|-----------|-------------|
| **Utah-Prime-Sieve** | Cross-venue capital drag audit + k-sunflower routing |
| **Settlement manifold** | Programmatic 2.3% protocol tithe + 5.7% humanitarian split |
| **Adelic bypass** | Hasse-Minkowski local-global verification → zero modeled clearing margin |
| **Wave PINN** | JAX physics-informed alpha (`OrthogonalWaveStatePredictor`) |
| **Topological Risk Parity** | Persistent-homology allocation (`optimize_topological_risk_parity`) |
| **Ricci flow covariance** | Exact `jax.jacfwd` geometric denoising (`compute_ricci_flow_covariance`) |
| **Exact curvature (autodiff)** | Christoffel / Riemann / Ricci to machine precision (`riemannian_geometry`) |
| **Pathwise metric observer** | Quadratic-covariation tick observer, zero-lag (`QuadraticCovariationObserver`) |
| **Navier-Stokes routing** | Mass-conserving rebalance flow (`calculate_navier_stokes_rebalance_flow`) |
| **Spectral CVaR veto** | Analytic drawdown wall (`apply_spectral_cvar_veto`) |
| **Laplace-Beltrami drawdown bound** | Feynman-Kac spectral supremum (`feynman_kac_drawdown_bound`) |
| **Betti divergence test** | Crash-topology contagion diagnostic (`betti_number_divergence_test`) |
| **Holographic LOB embedding** | AdS/CFT-style hyperbolic order-book pressure feature (`holographic_order_book_pressure`) |
| **Tensor-network entropy** | MPS / Von Neumann diversification spectrum (`von_neumann_portfolio_entropy`) |
| **Malliavin sensitivities** | Skorokhod / Malliavin Greeks & risk decomposition (`malliavin_delta_european_call`) |
| **Jarzynski free-energy** | Non-equilibrium free-energy + irreversibility estimator (`extract_fluctuation_arbitrage`) |
| **Braid execution router** | Kauffman/Jones invariants + non-commutative ordering (`optimize_execution_braid`) |
| **Koopman EDMD operator** | Spectral linearization & forecast in lifted space (`compute_koopman_edmd_prediction`) |
| **Verification lattice** | Navier alignment, adelic sieve bounds, omnibus audit |
| **Sovereign vault** | Threshold-signed intent custody |

---

## Quick start

### Clone

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
```

### Install (Windows)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.app.utah_prime_sieve_daemon
```

### Install (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m src.app.utah_prime_sieve_daemon
```

### Optional JAX alpha + exact geometry

JAX powers both the PINN alpha engine and the **exact-autodiff** continuous-time
geometry (Ricci-flow denoising, Christoffel/Riemann/Ricci tensors, Laplace-Beltrami
drawdown bound). These paths are JAX-mandatory by design — there is no
finite-difference fallback (`O(h^2)` truncation corrupts the non-linear flow).

```bash
pip install -e ".[jax,dev]"
python -m src.models.pinn_jax_runtime
```

---

## Documentation

**Languages:** English · [Eesti](docs/et/README.md) · [Русский](docs/ru/README.md)

Full guides for every audience live in **[`docs/`](docs/README.md)**. Complete
Estonian and Russian translations are kept in **completely separate files** under
[`docs/et/`](docs/et/README.md) and [`docs/ru/`](docs/ru/README.md) (code, file
paths, API names and math notation are identical across languages):

| Guide | Audience |
|-------|----------|
| [docs/README.md](docs/README.md) | Documentation index |
| [01-engineers-architects.md](docs/01-engineers-architects.md) | Engineers & DevOps |
| [02-finance-professionals.md](docs/02-finance-professionals.md) | Quants, PMs, traders |
| [03-founders-family-offices.md](docs/03-founders-family-offices.md) | Family offices & principals |
| [04-children-beginners.md](docs/04-children-beginners.md) | Children & beginners |
| [05-settlement-governance.md](docs/05-settlement-governance.md) | Ops & compliance |
| [06-adelic-clearinghouse-bypass.md](docs/06-adelic-clearinghouse-bypass.md) | Adelic settlement deep dive |
| [07-wave-pinn-alpha-engine.md](docs/07-wave-pinn-alpha-engine.md) | PINN / JAX alpha |
| [08-continuous-time-allocation.md](docs/08-continuous-time-allocation.md) | TRP, Ricci flow, NS routing, spectral CVaR, tick observer |
| [09_Ricci_Flow_Stabilization.tex](docs/09_Ricci_Flow_Stabilization.tex) | Curvature-uniformization denoiser theorem (LaTeX) |
| [10_Entanglement_Hedging.tex](docs/10_Entanglement_Hedging.tex) | Tensor-network entanglement diversification (LaTeX) |
| [11_Jarzynski_Harvesting.tex](docs/11_Jarzynski_Harvesting.tex) | Non-equilibrium free-energy estimation theorem (LaTeX) |
| [12_Braid_Execution.tex](docs/12_Braid_Execution.tex) | Braid/Kauffman invariants + non-commutative ordering (LaTeX) |
| [13_Koopman_Linearization.tex](docs/13_Koopman_Linearization.tex) | Koopman/EDMD linearization theorem (LaTeX) |
| [glossary.md](docs/glossary.md) | Terminology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview & diagrams |

---

## Repository layout

```text
utahfinancelibrary/
├── Utahfile              # Deployment manifest (v6 Adelic)
├── src/
│   ├── core/             # Auditing, routing, verification, adelic clearing
│   ├── app/              # Daemons & ignite CLI
│   └── models/           # Wave telemetry & JAX PINN
├── tests/
├── docs/
└── .github/workflows/    # CI
```

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `python -m src.app.utah_prime_sieve_daemon` | Full demo pipeline |
| `python -m src.app.hasse_minkowski_daemon` | Adelic zero-collateral demo |
| `python -m src.models.pinn_jax_runtime` | Train/compile PINN |
| `python -m src.app.ignite --manifest Utahfile --dry-run` | Validate deployment manifest |
| `python -m src.core.utah_verification_manifold` | Run verification lattice |

Installed console scripts (after `pip install -e .`): `utah-sieve`, `utah-verify`, `utah-ignite`, `utah-adelic`, `utah-alpha`.

---

## Protocol constants

| Constant | Value | Notes |
|----------|-------|-------|
| Sovereign protocol tithe | **2.3%** | Immutable (`SOVEREIGN_PROTOCOL_TITHE`) |
| Humanitarian abundance | **5.7%** | Default; configurable per deployment |

---

## Deploy with Utahfile

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
# Production (when utahcontainerengine is available):
# uce ignite --manifest Utahfile --profile structural-sovereignty
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Disclaimer

This project provides **software building blocks** for auditing, routing, settlement instructions, and mathematical verification. It does **not** replace regulated clearinghouses, licensed investment advice, or legal/tax structuring. Consult compliance and counsel before production use.

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Utah Finance Library Contributors
