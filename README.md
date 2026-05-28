# Utah Finance Library

[![CI](https://github.com/utahisnotastate/utahfinancelibrary/actions/workflows/ci.yml/badge.svg)](https://github.com/utahisnotastate/utahfinancelibrary/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Version 6.Omnibus_Adelic** — Open-source modular stack for capital efficiency, physics-informed alpha, and adelic settlement verification.

Official repository: **[github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)**

Part of the broader [utahisnotastate](https://github.com/utahisnotastate) ecosystem (UCE, wavetheory, and related engines).

---

## Features

| Component | Description |
|-----------|-------------|
| **Utah-Prime-Sieve** | Cross-venue capital drag audit + k-sunflower routing |
| **Settlement manifold** | Programmatic 2.3% protocol tithe + 5.7% humanitarian split |
| **Adelic bypass** | Hasse-Minkowski local-global verification → zero modeled clearing margin |
| **Wave PINN** | JAX physics-informed alpha (`OrthogonalWaveStatePredictor`) |
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

### Optional JAX alpha

```bash
pip install -e ".[jax,dev]"
python -m src.models.pinn_jax_runtime
```

---

## Documentation

Full guides for every audience live in **[`docs/`](docs/README.md)**:

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
