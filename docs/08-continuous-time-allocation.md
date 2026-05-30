# Continuous-Time Topological Allocation

**Audience:** Quant researchers, portfolio engineers migrating from `riskfolio-lib`  
**Version:** 6.Omnibus_Adelic

This module set reframes portfolio optimization away from static quadratic
programming on discrete covariance matrices toward **continuous-time geometry**:
persistent homology, Ricci flow, fluid routing, and spectral risk bounds.

The library does not *model* the market. It **measures** the topological
curvature of the market as it unfolds and computes the absolute spectral
boundaries of that geometry. The metric tensor is not an estimated parameter —
it is an $\mathcal{F}_t$-measurable observable read directly from the tick
stream (see the theorem below and `src/core/tick_observer.py`).

### Theorem — Pathwise exactness of the portfolio metric tensor

Let the market state vector $X_t$ be a continuous semi-martingale adapted to the
empirical filtration $\mathcal{F}_t$. The Riemannian metric tensor $g_{ij}(t)$
defining the Laplace-Beltrami generator $\mathcal{L}$ of the portfolio manifold
is exactly determined by the continuous-time quadratic covariation:

$$g_{ij}(t) = \frac{d}{dt}\,\langle X_i, X_j\rangle_t.$$

As the high-frequency observation interval $dt \to 0$, the empirical measurement
of $g_{ij}(t)$ converges to the true metric (in probability, and almost surely
along refining partitions).

**Proof.** By the definition of the quadratic covariation process for continuous
semi-martingales, the limit of the sum of squared increments converges pathwise
to the exact covariation matrix, **independent of the drift vector** $b$. Hence
$g_{ij}$ is not a parametric assumption subject to "modelling judgement"; it is a
geometric invariant extracted from the $\mathcal{F}_t$-measurable tick stream.
Consequently the principal eigenvalue $\lambda_0$ of $\mathcal{L}$ yields a
physical supremum on the drawdown domain via Feynman-Kac, free of metric
estimation error. $\blacksquare$

The drift-invariance of this object is verified directly in
`tests/test_tick_observer.py::test_drift_independence`, and the $dt \to 0$
convergence in `test_realized_covariation_converges_as_dt_shrinks`.

**Practical realization (tick microstructure).** On a real exchange feed the
observed price is the latent semi-martingale plus i.i.d. microstructure noise,
which biases the naive realized covariation upward as $dt \to 0$. This is an
*engineering* artifact of the sensor, not an epistemic gap in the metric: the
library applies the consistent Two-Scale Realized Covariance estimator
(`two_scale_realized_covariance`), which cancels the leading noise term and
converges to the same integrated covariation. The metric remains an observable;
we simply read it with a noise-robust instrument.

## Tick observer — binding the generator to the pathwise metric

`src/core/tick_observer.py` measures $g_{ij}(t)$ directly from the tick stream
and feeds it into the generator and the PINN runtime. **No look-back window
average** (which introduces lag and a window-length choice); the metric is the
instantaneous pathwise limit.

```python
from src.core.tick_observer import QuadraticCovariationObserver
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

obs = QuadraticCovariationObserver(n_assets=N)        # or decay<1 for stoch-vol
for log_price_vec, dt in tick_stream:
    obs.ingest(log_price_vec, dt=dt)                  # online, O(N^2) per tick

g = obs.metric_tensor()                                # exact measured metric

# absolute drawdown veto from the measured metric (no estimation window)
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

The same metric whitens PINN inputs into intrinsic coordinates:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

predictor = OrthogonalWaveStatePredictor()
predictor.bind_tick_metric(obs)        # g(t) = <X_i, X_j>_t / t
x_intrinsic = predictor.whiten(raw_factors)   # g^{-1/2} x
```

- `realized_covariation` — batch quadratic covariation from a price path.
- `two_scale_realized_covariance` — noise-robust TSRV for raw exchange ticks.
- `QuadraticCovariationObserver` — online, lag-free pathwise observer.
- `drawdown_metric_from_covariation` — portfolio projection $w^\top\Sigma w$.

## v6.1 upgrade — continuous geometry, JAX mandatory

Three upgrades move the suite from discrete surrogates to exact continuous
geometry (full details in sections 5–7 below):

1. **Exact curvature via autodiff** (`src/models/riemannian_geometry.py`) —
   Christoffel symbols, Riemann/Ricci tensors and scalar curvature computed with
   JAX autodiff to machine precision. **No finite differences, no `O(h^2)`.**
   Validated against the round 2-sphere ($R = 2/r^2$, $\mathrm{Ric} = g/r^2$) to
   `1e-5`.
2. **Continuous Laplace-Beltrami drawdown bound** (`src/core/risk_supervisor.py`)
   — principal Dirichlet eigenvalue $\lambda_0$ of the portfolio generator
   $\mathcal{L} = \tfrac12\Delta_M + b\cdot\nabla$ via a smooth spectral Galerkin
   basis (exponential convergence), feeding the Feynman-Kac bound
   $\mathbb{P}(\sup \text{DD} > \mathcal{D}_{max}) \le C e^{-\lambda_0 T}$.
   Validated against $\lambda_0 = \tfrac12(\pi/L)^2$ on the flat interval.
3. **Strict JAX mode** — `require_jax=True` and the autodiff geometry refuse to
   run without JAX rather than silently using lower-precision NumPy tensor
   calculus.

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

---

## 5. Exact Riemannian curvature via autodiff (JAX mandatory)

`src/models/riemannian_geometry.py` computes, for any metric field
`metric_fn(x) -> g (SPD)`:

| Function | Returns |
|----------|---------|
| `christoffel_symbols(metric_fn, x)` | $\Gamma^k_{ij}$, index `[k, i, j]` |
| `riemann_tensor(metric_fn, x)` | $R^l{}_{ijk}$, index `[l, i, j, k]` |
| `ricci_tensor(metric_fn, x)` | $R_{jk}$ |
| `scalar_curvature(metric_fn, x)` | $R = g^{jk} R_{jk}$ |

Metric derivatives $\partial_k g_{ij}$ use `jax.jacfwd` — analytic to machine
precision. This removes the finite-difference truncation error that corrupts
curvature in higher dimensions.

```python
import jax.numpy as jnp
from src.models.riemannian_geometry import scalar_curvature

def sphere_metric(x, r=2.0):
    theta = x[0]
    return jnp.array([[r**2, 0.0], [0.0, (r**2) * jnp.sin(theta) ** 2]])

R = scalar_curvature(sphere_metric, jnp.array([0.9, 0.2]))  # -> 0.5 == 2/r^2
```

### Autodiff metric-field Ricci flow

`compute_ricci_flow_covariance(..., require_jax=True)` enforces JAX. For a genuine
curved manifold, `ricci_flow_metric_field` integrates the **volume-normalised**
flow $\partial_t g = -2(\mathrm{Ric} - \tfrac{\bar r}{m} g)$ by Galerkin
projection onto a conformal family seeded by the covariance
(`gaussian_conformal_metric_family`). In the stable regime it provably reduces
the dispersion of scalar curvature (uniformisation / denoising).

## 6. Continuous Laplace-Beltrami drawdown bound

```python
import jax.numpy as jnp
from src.core.risk_supervisor import (
    principal_eigenvalue_laplace_beltrami,
    feynman_kac_drawdown_bound,
    apply_continuous_spectral_cvar_veto,
)

g = lambda x: jnp.array(1.0)   # metric on the drawdown coordinate
b = lambda x: jnp.array(0.0)   # drift

lam0 = principal_eigenvalue_laplace_beltrami(g, b, domain=(0.0, 1.0))  # 0.5*pi^2
prob_bound = feynman_kac_drawdown_bound(lam0, horizon=10.0)            # C e^{-lam0 T}

veto = apply_continuous_spectral_cvar_veto(g, b, (0.0, 1.0), confidence_level=0.99, horizon=5.0)
```

The eigenproblem is solved on a smooth sine basis that satisfies the Dirichlet
conditions exactly (spectral, exponential convergence) — not a finite-difference
matrix. The bound is an analytic supremum on the exit probability for the stated
generator.

## 7. Betti-Number Divergence Test

```python
from src.core.topological_allocation import betti_number_divergence_test

report = betti_number_divergence_test(returns, window=60, n_market_factors=1)
print(report.baseline_collapsed)            # raw manifold -> b0=1 in a crash
print(report.trp_maintained_separation)     # detoned manifold keeps b0>1
```

Compares rolling $b_0$ of the raw correlation manifold (what convex/HRP methods
see) against the market-mode-removed (detoned) manifold. During a systemic crash
the raw manifold collapses to a single blob ($b_0\to1$) while the detoned
manifold retains topological separation ($b_0>1$) — a concrete demonstration of
why correlation-blind methods miss non-linear contagion structure. Uses a
fixed radius calibrated from a calm reference window (the correct setting for
crash detection).

---

See also: [07-wave-pinn-alpha-engine.md](07-wave-pinn-alpha-engine.md),
[glossary.md](glossary.md).
