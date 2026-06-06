# Laine-oleku PINN alfamootor

**Sihtrühm:** ML-insenerid, kvantuurijad  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../07-wave-pinn-alpha-engine.md) · Eesti · [Русский](../ru/07-wave-pinn-alpha-engine.md) · [日本語](../ja/07-wave-pinn-alpha-engine.md)

## Miks PINN-id musta kasti ML asemel?

| Pärand-ML probleem | PINN lähenemine selles raamatukogus |
|--------------------|--------------------------------------|
| Ülesobitab ajaloolise müra | Füüsikateadlik Jacobiani trahv |
| Piiramatud aktiveerimised | Ortogonaalne faasilukk: $\sin(x)e^{-x^2/2}$ |
| Läbipaistmatu juurutus | JAX → XLA JIT `compile_bare_metal_graph()` kaudu |

## Arhitektuur

**Klass:** `OrthogonalWaveStatePredictor` (`src/models/pinn_jax_runtime.py`)

### Edasisuunaline läbimine

```python
hidden = sin(x @ W1 + b1) * exp(-0.5 * hidden**2)
y_hat = hidden @ W2 + b2
```

### Kadu

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum(\hat{y}-y)^2}_{\text{andmekadu}} + \lambda \underbrace{\mathbb{E}[|\text{trace}(J)|]}_{\text{vortilisuse trahv}}$$

kus $J$ on skalariseeritud edasisuunalise kaardistuse partii Jacobian.

## Treenimine

```python
from src.models.pinn_jax_runtime import (
    OrthogonalWaveStatePredictor,
    PINNTrainConfig,
)

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()

x_train = ...  # (N, features) turufaktorid
y_train = ...  # (N, 1) realiseeritud alfa

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

Kasutab **Optax Adam**-i, kui paigaldatud; muidu langeb tagasi käsitsi gradientlaskumisele.

## Integratsioon laine-telemeetriaga

```python
from src.models.wave_theory_engine import WaveTelemetryEngine, TickSample

wave = WaveTelemetryEngine(capacity=4096)
wave.ingest(TickSample(timestamp_ns=..., price=..., volume=...))
z = wave.z_score(latest_price)  # sööda PINN-i tunnusena
```

`utah_prime_sieve_daemon.py` käivitab mõlemad ühtses konveieris.

## Sõltuvused

```bash
pip install -e ".[jax]"
# jax, jaxlib, optax
```

## Riistvara

Töötab tava-CPU/GPU-l JAX-i wheel’idega. Kohandatud ASIC pole vajalik.

## Suveräänne marsruutimiskonstant

```python
SOVEREIGN_TITHE_ROUTING_DEFAULT = "0xUtahHansSovereignVault"
```

Kantakse ennustaja eksemplaril vastavuse metaandmetena; ühenda settlement-kihiga tootmiseks.

## Hindamise kontrollnimekiri

1. Edasiastuv (walk-forward) valideerimine kõrvalejäetud režiimidel
2. Võrdle Sharpe’i lähtejoonega pärast tehingukulusid
3. Jälgi Jacobiani trahvi suurust krahhide ajal
4. Piira `orthogonal_bound` puhvri suurust HFT-voogude jaoks

## CLI

```bash
python -m src.models.pinn_jax_runtime
```

## Ristviited

- Inseneride juhend: [01-engineers-architects.md](01-engineers-architects.md)
- Finantsülevaade: [02-finance-professionals.md](02-finance-professionals.md)
