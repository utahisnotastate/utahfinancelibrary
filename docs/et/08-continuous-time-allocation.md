# Pidevaja topoloogiline allokatsioon

**Sihtrühm:** kvantuurijad, `riskfolio-lib`-ist migreeruvad portfelliinsenerid  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../08-continuous-time-allocation.md) · Eesti · [Русский](../ru/08-continuous-time-allocation.md)

See mooduli komplekt raamib portfelli optimeerimise ümber staatiliselt
ruutprogrammeerimiselt diskreetsetel kovariatsioonimaatriksitel **pidevaja
geomeetriale**: püsivhomoloogia, Ricci voog, vedeliku marsruutimine ja
spektraalsed riskipiirid.

Raamatukogu ei *modelleeri* turgu. Ta **mõõdab** turu topoloogilist kõverust
selle avanedes ja arvutab selle geomeetria absoluutsed spektraalpiirid.
Meetrikatensor pole hinnatud parameeter — see on $\mathcal{F}_t$-mõõdetav
vaadeldav suurus, mis loetakse otse tiksuvoost (vt teoreem allpool ja
`src/core/tick_observer.py`).

### Teoreem — portfelli meetrikatensori rajapõhine täpsus

Olgu turu olekuvektor $X_t$ pidev poolmartingaal, mis on adapteeritud empiirilisse
filtratsiooni $\mathcal{F}_t$. Riemanni meetrikatensor $g_{ij}(t)$, mis määratleb
portfelli muutkonna Laplace-Beltrami generaatori $\mathcal{L}$, on täpselt
määratud pidevaja ruutkovariatsiooniga:

$$g_{ij}(t) = \frac{d}{dt}\,\langle X_i, X_j\rangle_t.$$

Kui kõrgsageduslik vaatlusintervall $dt \to 0$, koondub $g_{ij}(t)$ empiiriline
mõõtmine tegelikule meetrikale (tõenäosuses ja peaaegu kindlasti piki täpsustuvaid
partitsioone).

**Tõestus.** Pidevate poolmartingaalide ruutkovariatsiooni protsessi definitsiooni
järgi koondub ruutsuurenemiste summa piir rajaliselt täpsesse
kovariatsioonimaatriksisse, **sõltumatult triivivektorist** $b$. Seega pole
$g_{ij}$ parameetriline eeldus, mis allub „modelleerimisotsustusele”; see on
geomeetriline invariant, mis on ekstraheeritud $\mathcal{F}_t$-mõõdetavast
tiksuvoost. Järelikult annab $\mathcal{L}$ peamine omaväärtus $\lambda_0$
Feynman-Kaci kaudu füüsilise ülempiiri väljavõtu valdkonnale, vaba meetrika
hindamisveast. $\blacksquare$

Selle objekti triivisõltumatust kontrollitakse otse failis
`tests/test_tick_observer.py::test_drift_independence` ning $dt \to 0$ koondumist
failis `test_realized_covariation_converges_as_dt_shrinks`.

**Praktiline teostus (tiksu mikrostruktuur).** Reaalsel börsivool on vaadeldav
hind latentne poolmartingaal pluss i.i.d. mikrostruktuuri müra, mis kallutab
naiivset realiseeritud kovariatsiooni ülespoole, kui $dt \to 0$. See on sensori
*inseneriartefakt*, mitte epistemoloogiline lünk meetrikas: raamatukogu rakendab
järjepidevat kaheskaalalist realiseeritud kovariatsiooni hinnangut
(`two_scale_realized_covariance`), mis tühistab juhtiva müraliikme ja koondub
samasse integreeritud kovariatsiooni. Meetrika jääb vaadeldavaks suuruseks; me
lihtsalt loeme seda mürakindla instrumendiga.

## Tiksuvaatleja — generaatori sidumine rajapõhise meetrikaga

`src/core/tick_observer.py` mõõdab $g_{ij}(t)$ otse tiksuvoost ja söödab selle
generaatorisse ja PINN-i käituskeskkonda. **Pole tagasivaateakna keskmist** (mis
toob mahajäämuse ja akna pikkuse valiku); meetrika on hetkeline rajapõhine piir.

```python
from src.core.tick_observer import QuadraticCovariationObserver
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

obs = QuadraticCovariationObserver(n_assets=N)        # või decay<1 stoh-vol jaoks
for log_price_vec, dt in tick_stream:
    obs.ingest(log_price_vec, dt=dt)                  # online, O(N^2) tiksu kohta

g = obs.metric_tensor()                                # täpne mõõdetud meetrika

# absoluutne väljavõtu veto mõõdetud meetrikast (ilma hindamisaknata)
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

Sama meetrika valgendab PINN-i sisendid sisemistesse koordinaatidesse:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

predictor = OrthogonalWaveStatePredictor()
predictor.bind_tick_metric(obs)        # g(t) = <X_i, X_j>_t / t
x_intrinsic = predictor.whiten(raw_factors)   # g^{-1/2} x
```

- `realized_covariation` — partii ruutkovariatsioon hinnarajast.
- `two_scale_realized_covariance` — mürakindel TSRV toore börsitiksu jaoks.
- `QuadraticCovariationObserver` — online, mahajäämuseta rajapõhine vaatleja.
- `drawdown_metric_from_covariation` — portfelli projektsioon $w^\top\Sigma w$.

## v6.1 uuendus — pidev geomeetria, JAX kohustuslik

Kolm uuendust viivad komplekti diskreetsetelt asendustelt täpsele pidevale
geomeetriale (täielikud üksikasjad jaotistes 5–7 allpool):

1. **Täpne kõverus autodiff’i abil** (`src/models/riemannian_geometry.py`) —
   Christoffeli sümbolid, Riemanni/Ricci tensorid ja skalaarkõverus, mis on
   arvutatud JAX autodiff’iga masintäpsuseni. **Pole lõplikke vahesid, pole
   `O(h^2)`.** Valideeritud ümara 2-sfääri vastu ($R = 2/r^2$,
   $\mathrm{Ric} = g/r^2$) kuni `1e-5`.
2. **Pidev Laplace-Beltrami väljavõtu piir** (`src/core/risk_supervisor.py`) —
   portfelli generaatori $\mathcal{L} = \tfrac12\Delta_M + b\cdot\nabla$ peamine
   Dirichlet’ omaväärtus $\lambda_0$ sileda spektraalse Galerkini baasi kaudu
   (eksponentsiaalne koondumine), mis söödab Feynman-Kaci piiri
   $\mathbb{P}(\sup \text{DD} > \mathcal{D}_{max}) \le C e^{-\lambda_0 T}$.
   Valideeritud $\lambda_0 = \tfrac12(\pi/L)^2$ vastu lamedal intervallil.
3. **Range JAX-režiim** — `require_jax=True` ja autodiff-geomeetria keelduvad
   töötamast ilma JAX-ita, selle asemel et vaikselt kasutada madalama täpsusega
   NumPy tensorarvutust.

## Ülevaade

| API | Fail | Asendab (riskfolio) |
|-----|------|---------------------|
| `optimize_topological_risk_parity` | `src/core/topological_allocation.py` | HRP / NCO |
| `compute_ricci_flow_covariance` | `src/models/manifold_kernel.py` | Ledoit-Wolf / OAS kahandus, mürafiltreerimine |
| `calculate_navier_stokes_rebalance_flow` | `src/core/sunflower_router.py` | L1 käibe / tehingukulu piirangud |
| `apply_spectral_cvar_veto` | `src/core/risk_supervisor.py` | Mean-CVaR / EVaR / Max-Drawdown |

---

## 1. Topoloogiline riskipariteet (TRP)

Ehitab Mantegna korrelatsioonikauguse $d_{ij}=\sqrt{2(1-\rho_{ij})}$ ja selle peal
Vietoris-Ripsi filtratsiooni.

- **H0** püsivus arvutatakse *täpselt* üksiklingi union-find’iga.
- **H1** (tsüklid / nakkuslingid) kasutab 1-skeleti tsüklite astakut $E - V + b_0$.
- **H2** kasutab täidetud kolmnurga õõnsuse asendust.

Kapital jaotatakse **pöördvõrdeliselt** iga vara topoloogilise koormusega
(sildühendused + tsükli aste), seejärel segatakse $1/N$ poole Wassersteini
vöötkoodi liikme abil.

```python
from src.core.topological_allocation import optimize_topological_risk_parity

weights = optimize_topological_risk_parity(
    tensor_data=returns,            # (T, N) array-like (NumPy või jax.Array)
    max_homology_dimension=2,
    wasserstein_penalty=0.01,
)
```

Rikkalik diagnostika:

```python
from src.core.topological_allocation import topological_risk_parity_report
report = topological_risk_parity_report(returns)
print(report.betti_numbers, report.barcode_wasserstein, report.filtration_radius)
```

---

## 2. Ricci voo kovariatsioon

Integreerib mahu-normaliseeritud Ricci voo

$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \tfrac{2}{m} r\, g_{ij}$$

kui **otsesilla täpse autodiff meetrikavälja voole** — rajal pole NumPy lõplike
vahede asendust. Konstantne kovariatsioonimaatriks on *lame* meetrika (null
Ricci), seega ehitab API esmalt kõverdunud Riemanni meetrika*välja*, mille kõverus
on indutseeritud kovariatsioonispektrist, ning integreerib siis voo Christoffeli
sümbolite ja Ricci tensoriga, mis on saadud `jax.jacfwd` abil masintäpsuseni.
Spekter kahandatakse oma keskmise poole **mõõdetud kõveruse-ühtlustumise suhtega**

$$\gamma = \frac{\operatorname{std}_x R(T)}{\operatorname{std}_x R(0)} \in [0,1],
\qquad \log\lambda_i^{\text{den}} = \overline{\log\lambda} + \gamma\,(\log\lambda_i - \overline{\log\lambda}),$$

nii et kahanemist juhib voo täpne geomeetria ($\gamma \to 0$, kui skalaarkõverus
muutub konstantseks), mitte diskreetne asendus. Jälg (kogudispersioon) säilib.
`ricci_curvature_proxy` jääb eraldiseisvaks spektraalseks diagnostikaks, kuid seda
**ei kutsuta kunagi** mürafiltri poolt (jõustatud failis
`tests/test_ricci_flow_autodiff.py::test_api_does_not_call_numpy_proxy`).

> Kõveruse-ühtlustumise teoreem ja selle täpne vastavus teostatud kahanemissuhtega
> $\gamma$ on vormistatud failis
> [`docs/09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex).

```python
from src.models.manifold_kernel import compute_ricci_flow_covariance

denoised = compute_ricci_flow_covariance(
    empirical_metric_tensor=cov,    # (N, N) SPD
    flow_duration=1.0,
    manifold_dimension=cov.shape[0],
)
```

Täpne (autodiff’iga mõõdetud) skalaarkõveruse-dispersiooni trajektoor
`wave_theory_engine` jaoks:

```python
from src.models.manifold_kernel import ricci_flow_curvature_field
times, curv_dispersion = ricci_flow_curvature_field(cov, 1.0, cov.shape[0], samples=8)
```

Empiiriliselt **kahandab voog omaväärtuste hajumist** (mürafiltreerimine),
säilitades SPD-struktuuri ja jälje — vt `tests/test_manifold_kernel.py` ja
`tests/test_ricci_flow_autodiff.py`.

---

## 3. Navier-Stokesi likviidsuse marsruutimine

Modelleerib tasakaalustamist kokkusurumatu Stokesi vooluna: viskoossus =
turumõju, kehajõud = alfa gradient. Lahendab sadulpunkti (KKT) süsteemi, nii et
kiirusväli on **massisäilitav** ($\mathbf{1}^\top v = 0$).

```python
from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow

velocity, pressure_gradient = calculate_navier_stokes_rebalance_flow(
    current_weights=w_now,
    target_manifold=w_target,
    market_viscosity_tensor=0.1,        # skalaar, (N,) või (N, N)
    kinematic_constraints={"max_velocity": 0.25, "coupling": 1.0},
)
```

Tagastab pideva täitmise **kiirusvälja** (kui kiiresti/kuhu kapital voolab) ja
**rõhugradiendi**, mis marsruudib arbitraaži — mitte ainult staatilised
sihtkaalud.

---

## 4. Omavektor-muutkonna (spektraalne) CVaR veto

Diskretiseerib Schrödingeri-tüüpi operaatori $-\Delta + V(x)$ Dirichlet’
ääretingimustega, kus $V$ on valimitud sinu `loss_operator`-ist. Kui spektraalne
raadius murrab usaldusseina, käivitub **sümplektiline veto**.

```python
from src.core.risk_supervisor import apply_spectral_cvar_veto

veto = apply_spectral_cvar_veto(
    loss_operator=lambda x: 50.0 * x**2,   # kadu-tihedus / potentsiaal
    confidence_level=0.99,
    dirichlet_boundary_conditions=[0.0, 0.0],
)
if veto:
    raise RuntimeError("Spectral CVaR wall breached — halt execution")
```

Uuri numbreid:

```python
from src.core.risk_supervisor import spectral_cvar_diagnostics
radius, boundary, veto = spectral_cvar_diagnostics(op, 0.99, [0.0, 0.0])
```

Laplacian kasutab skaala-stabiilset graafi šablooni (omaväärtused $[0,4]$), nii et
**kadu-potentsiaal** — mitte võrgu samm — juhib riskiseina.

---

## Otsast-otsani visand

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

## 5. Täpne Riemanni kõverus autodiff’i abil (JAX kohustuslik)

`src/models/riemannian_geometry.py` arvutab iga meetrikavälja
`metric_fn(x) -> g (SPD)` jaoks:

| Funktsioon | Tagastab |
|------------|----------|
| `christoffel_symbols(metric_fn, x)` | $\Gamma^k_{ij}$, indeks `[k, i, j]` |
| `riemann_tensor(metric_fn, x)` | $R^l{}_{ijk}$, indeks `[l, i, j, k]` |
| `ricci_tensor(metric_fn, x)` | $R_{jk}$ |
| `scalar_curvature(metric_fn, x)` | $R = g^{jk} R_{jk}$ |

Meetrika tuletised $\partial_k g_{ij}$ kasutavad `jax.jacfwd` — analüütiline
masintäpsuseni. See eemaldab lõplike vahede katkestusvea, mis rikub kõverust
kõrgemates dimensioonides.

```python
import jax.numpy as jnp
from src.models.riemannian_geometry import scalar_curvature

def sphere_metric(x, r=2.0):
    theta = x[0]
    return jnp.array([[r**2, 0.0], [0.0, (r**2) * jnp.sin(theta) ** 2]])

R = scalar_curvature(sphere_metric, jnp.array([0.9, 0.2]))  # -> 0.5 == 2/r^2
```

### Autodiff meetrikavälja Ricci voog

`compute_ricci_flow_covariance` **on** see voog — peamine mürafiltri API on otsene
sild `ricci_flow_metric_field`-ile, ilma NumPy asenduseta rajal. See integreerib
**mahu-normaliseeritud** voo $\partial_t g = -2(\mathrm{Ric} - \tfrac{\bar r}{m} g)$
Galerkini projektsiooni teel kõverdunud meetrikaperekonnale, mis on seemnestatud
kovariatsioonispektriga (`_anisotropic_conformal_family`; saadaval on ka 2-D
konformne `gaussian_conformal_metric_family`). Ricci tensor igas valimipunktis on
`jax.jacfwd`-täpne, voog vähendab skalaarkõveruse dispersiooni (ühtlustumine) ja
see mõõdetud suhe juhib spektri kahanemist (mürafiltreerimine) — jälg säilib, SPD
garanteeritud.

## 6. Pidev Laplace-Beltrami väljavõtu piir

```python
import jax.numpy as jnp
from src.core.risk_supervisor import (
    principal_eigenvalue_laplace_beltrami,
    feynman_kac_drawdown_bound,
    apply_continuous_spectral_cvar_veto,
)

g = lambda x: jnp.array(1.0)   # meetrika väljavõtu koordinaadil
b = lambda x: jnp.array(0.0)   # triiv

lam0 = principal_eigenvalue_laplace_beltrami(g, b, domain=(0.0, 1.0))  # 0.5*pi^2
prob_bound = feynman_kac_drawdown_bound(lam0, horizon=10.0)            # C e^{-lam0 T}

veto = apply_continuous_spectral_cvar_veto(g, b, (0.0, 1.0), confidence_level=0.99, horizon=5.0)
```

Omaväärtusprobleem lahendatakse sileda siinusbaasi peal, mis rahuldab Dirichlet’
tingimusi täpselt (spektraalne, eksponentsiaalne koondumine) — mitte lõplike
vahede maatriks. Piir on analüütiline ülempiir väljumistõenäosusele antud
generaatori jaoks.

## 7. Betti-arvu hajumise test

```python
from src.core.topological_allocation import betti_number_divergence_test

report = betti_number_divergence_test(returns, window=60, n_market_factors=1)
print(report.baseline_collapsed)            # toore muutkond -> b0=1 krahhis
print(report.trp_maintained_separation)     # detoneeritud muutkond hoiab b0>1
```

Võrdleb toore korrelatsioonimuutkonna libisevat $b_0$ (mida näevad kumerus/HRP
meetodid) turumoodi-eemaldatud (detoneeritud) muutkonnaga. Süsteemse krahhi ajal
variseb toore muutkond üheks pundiks ($b_0\to1$), samal ajal kui detoneeritud
muutkond säilitab topoloogilise eralduse ($b_0>1$) — konkreetne demonstratsioon,
miks korrelatsioonipimedad meetodid jätavad mittelineaarse nakkusstruktuuri
märkamata. Kasutab fikseeritud raadiust, mis on kalibreeritud rahuliku
referentsakna järgi (õige seade krahhi tuvastamiseks).

---

Vt ka: [07-wave-pinn-alpha-engine.md](07-wave-pinn-alpha-engine.md),
[glossary.md](glossary.md).
