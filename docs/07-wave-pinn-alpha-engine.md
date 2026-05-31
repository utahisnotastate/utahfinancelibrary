# Wave-State PINN Alpha Engine

**Audience:** ML engineers, quant researchers  
**Version:** 6.Omnibus_Adelic

> **Languages:** English · [Eesti](et/07-wave-pinn-alpha-engine.md) · [Русский](ru/07-wave-pinn-alpha-engine.md)

## Why PINNs instead of black-box ML?

| Issue with legacy ML | PINN approach in this library |
|----------------------|-------------------------------|
| Overfits historical noise | Physics-informed Jacobian penalty |
| Unbounded activations | Orthogonal phase-lock: $\sin(x)e^{-x^2/2}$ |
| Opaque deployment | JAX → XLA JIT via `compile_bare_metal_graph()` |

## Architecture

**Class:** `OrthogonalWaveStatePredictor` (`src/models/pinn_jax_runtime.py`)

### Forward pass

```python
hidden = sin(x @ W1 + b1) * exp(-0.5 * hidden**2)
y_hat = hidden @ W2 + b2
```

### Loss

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum(\hat{y}-y)^2}_{\text{data loss}} + \lambda \underbrace{\mathbb{E}[|\text{trace}(J)|]}_{\text{vorticity penalty}}$$

where $J$ is the batch Jacobian of the scalarized forward map.

## Training

```python
from src.models.pinn_jax_runtime import (
    OrthogonalWaveStatePredictor,
    PINNTrainConfig,
)

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()

x_train = ...  # (N, features) market factors
y_train = ...  # (N, 1) realized alpha

params, predict_fn = predictor.train(
    x_train,
    y_train,
    PINNTrainConfig(
        hidden_dim=64,
        learning_rate=1e-3,
        steps=200,
        vorticity_weight=0.01,
    ),
)
```

Uses **Optax Adam** when installed; falls back to manual gradient descent.

## Integration with wave telemetry

```python
from src.models.wave_theory_engine import WaveTelemetryEngine, TickSample

wave = WaveTelemetryEngine(capacity=4096)
wave.ingest(TickSample(timestamp_ns=..., price=..., volume=...))
z = wave.z_score(latest_price)  # feed as feature to PINN
```

`utah_prime_sieve_daemon.py` runs both in the unified pipeline.

## Dependencies

```bash
pip install -e ".[jax]"
# jax, jaxlib, optax
```

## Hardware

Runs on standard CPU/GPU with JAX wheels. No custom ASIC required.

## Sovereign routing constant

```python
SOVEREIGN_TITHE_ROUTING_DEFAULT = "0xUtahHansSovereignVault"
```

Carried on predictor instance for compliance metadata; wire to settlement layer for production.

## Evaluation checklist

1. Walk-forward validation on held-out regimes  
2. Compare Sharpe vs baseline after transaction costs  
3. Monitor Jacobian penalty magnitude during crashes  
4. Bound `orthogonal_bound` buffer size for HFT feeds

## CLI

```bash
python -m src.models.pinn_jax_runtime
```

## Cross-reference

- Engineers guide: [01-engineers-architects.md](01-engineers-architects.md)  
- Finance overview: [02-finance-professionals.md](02-finance-professionals.md)
