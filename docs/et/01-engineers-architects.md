# Piiritletud transfiniitse taristu juurutamine

**Sihtrühm:** tarkvarainsenerid, DevOps, süsteemiarhitektid  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../01-engineers-architects.md) · Eesti · [Русский](../ru/01-engineers-architects.md)

## Ülevaade

Utah Finance Library on **mitmemooduliline Pythoni monorepo**, mis asendab
monoliitsed fondihalduse virnad determineeritud, testitavate komponentidega:

| Moodul | Tee | Vastutus |
|--------|-----|----------|
| Kapitalisõel | `src/core/capital_sieve.py` | Platvormideülene tulususe hõõrdumine → tasakaalustusplaanid |
| Päevalille-marsruuter | `src/core/sunflower_router.py` | Lahusolevad likviidsuse kroonlehed ($k$-piiriga partitsioonid) |
| Verifitseerimisvõre | `src/core/utah_verification_manifold.py` | Navier’ joondus, adeelne sõela piir, omnibus-audit |
| Adeelne kliiring | `src/core/adelic_clearing.py` | Hasse-Minkowski lokaal-globaalne settlement-verifitseerimine |
| Settlement | `src/app/settlement.py` | Saagi jaotused (kümnis + humanitaar + reinvesteering) |
| PINN alfa | `src/models/pinn_jax_runtime.py` | JAX füüsikateadlik laineennustaja |
| Laine-telemeetria | `src/models/wave_theory_engine.py` | Piiritletud ringpuhvri tiksustatistika |
| Tiksuvaatleja | `src/core/tick_observer.py` | Rajapõhine ruutkovariatsiooni meetrika `g_ij(t)` |
| Muutkonna tuum | `src/models/manifold_kernel.py` | Täpne autodiff Ricci-voo kovariatsiooni mürafilter |
| Riemanni geomeetria | `src/models/riemannian_geometry.py` | Christoffel / Riemann / Ricci `jax.jacfwd` abil |
| Topoloogiline allokatsioon | `src/core/topological_allocation.py` | Püsivhomoloogia riskipariteet + Betti hajumine |
| Riskijärelevaataja | `src/core/risk_supervisor.py` | Spektraalne CVaR + Laplace-Beltrami väljavõtu piir |
| Hoidla | `src/app/vault_daemon.py` | Lävega allkirjastatud kavatsuste levitamine |
| Orkestraator | `src/app/alpha_orchestrator.py` | Riskipiirded + kaitselüliti |

## Eeltingimused

- Python 3.10+
- Windows: kasuta `py -3`, kui `python` pole PATH-is
- Valikuline JAX-virn: `pip install -e ".[jax,dev]"`

## Paigaldus

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Repositooriumi struktuur

```text
/
├── Utahfile                 # Juurutusmanifest (v6)
├── .cursorrules             # Cursor AI inseneri mandaat
├── src/
│   ├── core/                # Matemaatilised + finantskernelid
│   ├── app/                 # Deemonid ja CLI sisenemispunktid
│   └── models/              # JAX PINN + laine-telemeetria
├── tests/                   # pytest komplekt
└── docs/                    # See dokumentatsioonikomplekt
```

## Utahfile v6

`Utahfile` on ainus juurutusmanifest. Peamised v6 lisandused:

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    collateral_requirement: 0.00
```

`src/app/ignite.py` valideerib:

- settlement-eelne kümnis `0.023`
- automatiseeritud humanitaarmäär `0.057`
- null tagatis adeelsel teenusel
- adeelse settlement-protokolli string

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
```

## OrthogonalWaveStatePredictor (JAX / XLA)

Alfamootor asub failis `src/models/pinn_jax_runtime.py`.

**Edasisuunaline läbimine** kasutab ortogonaalset faasilukustuse aktiveerimist:

$$\text{hidden} = \sin(z) \cdot e^{-z^2/2}, \quad z = x W_1 + b_1$$

**Kadu** ühendab MSE ja partii Jacobianist tuletatud vortilisuse trahvi:

$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda \cdot \mathbb{E}[|\text{trace}(J)|]$$

Kompileeri juurutamiseks:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor, PINNTrainConfig

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()
params, predict = predictor.train(market_x, alpha_y, PINNTrainConfig(steps=200))
```

## Pidevaja geomeetria virn (JAX kohustuslik)

v6 geomeetriamoodulid asendavad staatilise kovariatsioonihinnangu täpse pidevaja
diferentsiaalgeomeetriaga. Ricci-voo mürafilter, Christoffeli/Riemanni/Ricci
tensorid ja Laplace-Beltrami väljavõtu piir on **ainult JAX-iga** —
NumPy lõplike vahede tagavara puudub, sest `O(h^2)` katkestusviga rikub
mittelineaarse voo PDE.

```python
import numpy as np
from src.core.tick_observer import QuadraticCovariationObserver
from src.models.manifold_kernel import compute_ricci_flow_covariance
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

# 1. Mõõda rajapõhine meetrikatensor tiksudest (null mahajäämus)
obs = QuadraticCovariationObserver(n_assets=N)
obs.ingest_prices(price_path, dt=1 / len(price_path))
g = obs.metric_tensor()                       # g_ij(t) = d/dt <X_i, X_j>_t

# 2. Mürafiltreeri täpse autodiff normaliseeritud Ricci voo abil (rajal pole puhverlahendust)
denoised = compute_ricci_flow_covariance(g, flow_duration=1.0, manifold_dimension=N)

# 3. Absoluutne väljavõtu veto mõõdetud meetrikast
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

Valideeri täpsust analüütiliste etalonide vastu:

```bash
pytest tests/test_riemannian_geometry.py   # 2-sfäär R = 2/r^2 kuni 1e-5
pytest tests/test_ricci_flow_autodiff.py   # proxy on rajalt välistatud; range mürafiltreerimine
pytest tests/test_tick_observer.py         # dt->0 koondumine, triivisõltumatus
```

Täielik teooria: [08-continuous-time-allocation.md](08-continuous-time-allocation.md)
ja [Ricci-voo stabiliseerimise teoreem](../09_Ricci_Flow_Stabilization.tex).

## Adeelse kliiringu integratsioon

```python
from src.core.adelic_clearing import AdelicClearinghouseEngine, AtomicTrade, VaultSnapshot

engine = AdelicClearinghouseEngine()
engine.register_vault(VaultSnapshot("buyer", {"USD": 1e9}))
engine.register_vault(VaultSnapshot("seller", {"WETH": 500}))
result = engine.attempt_atomic_settlement(AtomicTrade(...))
assert result.zero_collateral  # kui lokaal-globaalsed kontrollid läbivad
```

## Verifitseerimisvõre

Käivita eraldiseisvalt:

```bash
python -m src.core.utah_verification_manifold
```

Kontrollid hõlmavad:

1. **Navier’ geomeetriline ammendumine** — vortilisus vs venitustensor’i vahepealne omavektor
2. **Adeelne sõela intervall** — $Y(x) = \Theta(x \log x \log \log x)$ vs $O(x^2)$ pärandlagi
3. **Omnibus-audit** — kümnise määr, spektraalne jäikus, lahusolev marsruutimine

## Testimine

```bash
pytest -q
pytest tests/test_pinn.py                  # PINN alfa (nõuab JAX lisa)
pytest tests/test_riemannian_geometry.py   # täpne kõverus (nõuab JAX lisa)
pytest tests/test_manifold_kernel.py tests/test_ricci_flow_autodiff.py
pytest tests/test_tick_observer.py tests/test_laplace_beltrami.py
```

## Väline ökosüsteem

Mõeldud integreerimiseks [github.com/utahisnotastate](https://github.com/utahisnotastate):

- `utahcontainerengine` — tootmise unikerneli käituskeskkond (kohalik asendus: `ignite.py`)
- `wavetheory_5` — telemeetria mustrid, mida peegeldab `wave_theory_engine.py`

## Tootmise kontrollnimekiri

1. Asenda demo-hoidlad/positsioonid API-voogudega
2. Säilita settlement-räsid ja auditijälg muutumatusse hoidlasse
3. Ühenda `SovereignVault` reaalse TSS/HSM allkirjastamisega
4. Tee vastavusülevaade enne null-kliiringutagatise väitmist
5. Fikseeri sõltuvuste versioonid tootmis-image’ites
