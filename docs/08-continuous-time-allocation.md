# Continuous-Time Topological Allocation

**Audience:** Quant researchers, portfolio engineers migrating from `riskfolio-lib`  
**Version:** 6.Omnibus_Adelic

This module set reframes portfolio optimization away from static quadratic
programming on discrete covariance matrices toward **continuous-time geometry**:
persistent homology, Ricci flow, fluid routing, and spectral risk bounds.

> **Honesty note.** These are working, testable NumPy/JAX implementations of the
> *named* mathematical ideas, designed as research-grade building blocks. They
> are not magic and do not "prove" zero drawdown. Treat the geometric metrics as
> complementary signal to — not a guaranteed replacement for — validated convex
> methods. Backtest before trusting capital to them.

## Overview

| API | File | Replaces (riskfolio) |
|-----|------|----------------------|
| `optimize_topological_risk_parity` | `src/core/topological_allocation.py` | HRP / NCO |
| `compute_ricci_flow_covariance` | `src/models/manifold_kernel.py` | Ledoit-Wolf / OAS shrinkage, denoising |
| `calculate_navier_stokes_rebalance_flow` | `src/core/sunflower_router.py` | L1 turnover / transaction-cost constraints |
| `apply_spectral_cvar_veto` | `src/core/risk_supervisor.py` | Mean-CVaR / EVaR / Max-Drawdown |

---

## 1. Topological Risk Parity (TRP)

Builds the Mantegna correlation distance $d_{ij}=\sqrt{2(1-\rho_{ij})}$ and a
Vietoris-Rips filtration over it.

- **H0** persistence is computed *exactly* via single-linkage union-find.
- **H1** (cycles / contagion loops) uses the 1-skeleton cycle rank $E - V + b_0$.
- **H2** uses a filled-triangle cavity surrogate.

Capital is allocated **inversely** to each asset's topological load (bridging
merges + cycle degree), then blended toward $1/N$ by a Wasserstein barcode term.

```python
from src.core.topological_allocation import optimize_topological_risk_parity

weights = optimize_topological_risk_parity(
    tensor_data=returns,            # (T, N) array-like (NumPy or jax.Array)
    max_homology_dimension=2,
    wasserstein_penalty=0.01,
)
```

Rich diagnostics:

```python
from src.core.topological_allocation import topological_risk_parity_report
report = topological_risk_parity_report(returns)
print(report.betti_numbers, report.barcode_wasserstein, report.filtration_radius)
```

---

## 2. Ricci Flow Covariance

Treats covariance as a metric $g_{ij}$ and integrates volume-normalised Ricci flow

$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \tfrac{2}{m} r\, g_{ij}$$

in the eigenbasis, contracting the dispersion of log-eigenvalues (RMT noise)
toward constant curvature while preserving the trace (total variance).

```python
from src.models.manifold_kernel import compute_ricci_flow_covariance

denoised = compute_ricci_flow_covariance(
    empirical_metric_tensor=cov,    # (N, N) SPD
    flow_duration=1.0,
    manifold_dimension=cov.shape[0],
)
```

A dynamic curvature trajectory for `wave_theory_engine`:

```python
from src.models.manifold_kernel import ricci_flow_curvature_field
times, scalar_curvature = ricci_flow_curvature_field(cov, 1.0, cov.shape[0], samples=8)
```

Empirically the flow **lowers the condition number** (denoising) while keeping
SPD structure — see `tests/test_manifold_kernel.py`.

---

## 3. Navier-Stokes Liquidity Routing

Models rebalancing as incompressible Stokes flow: viscosity = market impact,
body force = alpha gradient. Solves a saddle-point (KKT) system so the velocity
field is **mass-conserving** ($\mathbf{1}^\top v = 0$).

```python
from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow

velocity, pressure_gradient = calculate_navier_stokes_rebalance_flow(
    current_weights=w_now,
    target_manifold=w_target,
    market_viscosity_tensor=0.1,        # scalar, (N,), or (N, N)
    kinematic_constraints={"max_velocity": 0.25, "coupling": 1.0},
)
```

Returns the continuous execution **velocity field** (how fast/where capital
flows) and the **pressure gradient** routing the arbitrage — not just static
target weights.

---

## 4. Eigenmanifold (Spectral) CVaR Veto

Discretises a Schrödinger-type operator $-\Delta + V(x)$ with Dirichlet boundary
conditions, where $V$ is sampled from your `loss_operator`. If the spectral
radius breaches the confidence wall, the **Symplectic Veto** fires.

```python
from src.core.risk_supervisor import apply_spectral_cvar_veto

veto = apply_spectral_cvar_veto(
    loss_operator=lambda x: 50.0 * x**2,   # loss density / potential
    confidence_level=0.99,
    dirichlet_boundary_conditions=[0.0, 0.0],
)
if veto:
    raise RuntimeError("Spectral CVaR wall breached — halt execution")
```

Inspect the numbers:

```python
from src.core.risk_supervisor import spectral_cvar_diagnostics
radius, boundary, veto = spectral_cvar_diagnostics(op, 0.99, [0.0, 0.0])
```

The Laplacian uses a scale-stable graph stencil (eigenvalues in $[0,4]$) so the
**loss potential** — not the grid spacing — governs the risk wall.

---

## End-to-end sketch

```python
import numpy as np
from src.core.topological_allocation import optimize_topological_risk_parity
from src.models.manifold_kernel import compute_ricci_flow_covariance
from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow
from src.core.risk_supervisor import apply_spectral_cvar_veto

returns = load_returns()                      # (T, N)
target = optimize_topological_risk_parity(returns)
cov = compute_ricci_flow_covariance(np.cov(returns, rowvar=False), 1.0, returns.shape[1])

if not apply_spectral_cvar_veto(lambda x: 10*x**2, 0.99, [0.0, 0.0]):
    v, p = calculate_navier_stokes_rebalance_flow(current, target, 0.1, {"max_velocity": 0.2})
    execute(v)
```

See also: [07-wave-pinn-alpha-engine.md](07-wave-pinn-alpha-engine.md),
[glossary.md](glossary.md).
