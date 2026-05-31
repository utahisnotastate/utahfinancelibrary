# Operatiivse hõõrdumise ja patenteeritud väljapressimise lõpp

**Sihtrühm:** kvantanalüütikud, portfellihaldurid, täitmiskauplejad, riskiohvitserid  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../02-finance-professionals.md) · Eesti · [Русский](../ru/02-finance-professionals.md)

## Kokkuvõte juhtkonnale

Kui sinu fond maksab endiselt seitsmekohalisi aastalitsentse Bloomberg AIM-i,
Enfusioni või Aladdini eest **ja** lukustab sadu miljoneid kliiringutagatisse,
siis veritsed alfat kahel rindel: **tarkvara väljatõmbamine** ja **tagatise
hõõrdumine**.

Utah Finance Library on avatud lähtekoodiga alternatiiv, mis automatiseerib:

1. **Kapitali hõõrdumise tuvastamine** — leiab primaarmaaklerite ja platvormide vahel kaotatud tulu
2. **Globaalne tasaarveldus** — marsruudib likviidsust lahusolevates partiides, et vähendada kattuvust
3. **Programmaatiline settlement** — jagab iga saagi täitmise ajal
4. **Adeelne möödaviik** — verifitseerib tehinguid aatomilise settlement’i jaoks ilma modelleeritud CCP-tagatiseta (kui maksevõime tõestused läbivad)
5. **Füüsikateadlik alfa** — JAX PINN kiht läbipaistmatu musta kasti ML asemel
6. **Pidevaja geomeetria** — mõõdab turu meetrikatensorit tiksudest ja piirab väljavõttu spektraalselt, aegunud kovariatsiooniakende hindamise asemel

## Probleem 1: platvormideülene kapitalileke

**Pärandtöövoog:** tabelarvutused + päevalõpu primaarmaakleri väljavõtted. Hõõrdumine avastatakse päevi hiljem.

**Utah’ töövoog:** `AutonomousAuditor` rühmitab positsioonid tikkeri järgi, võrdleb riskiga korrigeeritud tulusust ja väljastab `AllocationIntent`-i, kui hõõrdumine ületab läve.

Näidislogi:

```text
[AUDIT ALERT] Capital leakage in USD at prime_custody_01. Drag: 90000.0000
```

**Tegevus:** suuna jõude USD madala tulususega platvormilt optimaalsele automaatselt.

## Probleem 2: kliiringukojas lukus tagatis

**Pärandtöövoog:** tehing täitub börsil → kliiringuliige postitab tagatise DTCC-s → T+1 settlement → kapital seisab öö läbi jõude.

**Utah’ töövoog:** `HasseMinkowskiVerifier` kontrollib:

- **Reaalväli:** ostjal on noteeritav valuuta, müüjal baasvara
- **Lokaalsed väljad:** maksevõime kongruentsid väikeste algarvude mooduli järgi ($p$-aadiline ühilduvuse asendus)
- Kui kõik läbivad → `required_collateral = 0` → aatomiline pearaamatuvahetus

Demo-tehing vabastab **2,5 mln $ × 1000 WETH nominaali** modelleeritud pärandtagatise lukust, kui verifitseerimine läbib.

> **Riski avalikustus:** reaalne kliiring hõlmab seaduslikke tasaarvestusõigusi, CCP-reegleid ja regulatiivset kapitali. See raamatukogu rakendab **verifitseerimis- ja pearaamatumudelit**, mille pead enne tootmist kaardistama heakskiidetud taristule.

## Probleem 3: stohhastilise ML-alfa lagunemine

**Pärandtöövoog:** Transformer/LSTM ajaloolistel ribadel → ülesobitab müra → ebaõnnestub režiimimuutusel.

**Utah’ töövoog:** `OrthogonalWaveStatePredictor` jõustab:

- piiritletud aktiveerimismuutkonna (sin × Gaussi ümbris)
- Navier-Stokesist inspireeritud Jacobiani trahvi treeningu ajal
- XLA kompileerimise `compile_bare_metal_graph()` kaudu

## Probleem 4: käsitsi vastavusse viimine

**Pärandtöövoog:** operatsioonimeeskond viib EMS-i, PB ja fondiadministraatori vastavusse kuu lõpus.

**Utah’ töövoog:** iga settlement toodab determineeritud `SettlementInstruction` ridu ja SHA-256 settlement-räsisid adeelsetel tehingutel.

## Tasude konveieri läbipaistvus

Iga saak läbib `AutonomousSettlementEngine`-i:

| Saaja | Määr | Eesmärk |
|-------|------|---------|
| Protokolli kümnis (Utah Hans) | **2,3%** (muutumatu) | Raamatukogu ülalpidamine |
| Humanitaarse külluse maatriks | **5,7%** (vaikimisi, seadistatav) | Mõjuallokatsioon |
| Sisemine hoidla | Ülejäänu | Reinvesteering |

## Võrdlustabel

| Võimekus | Pärandvirn | Utah Finance Library |
|----------|------------|----------------------|
| Platvormideülene hõõrdumise audit | Käsitsi / viivitusega | Pidev (`capital_sieve`) |
| Tasaarveldus | Päevalõpu partii | Päevalille kroonlehed (lahusolev marsruutimine) |
| Settlement-jaotused | Kuulõpu raamatupidamine | Saagipõhine programmaatiline |
| Kliiringutagatis | CCP-reeglid + lõikused | Adeelne mudel → 0 kui verifitseeritud |
| Alfamudel | Läbipaistmatu tarnija ML | Avatud PINN + laine-telemeetria |
| Auditijälg | Andmebaasi muudatused | Räsiga kinnitatud sündmused |

## Alustamine (kauplemislaud)

```bash
python -m src.app.utah_prime_sieve_daemon --json > daily_audit.json
python -m src.app.hasse_minkowski_daemon
```

Vaata JSON-väljundis üle `migration_intents`, `adelic_clearing` ja `alpha_signal`.

## Probleem 5: aegunud akendel hinnatud staatiline kovariatsioon

**Pärandtöövoog:** tagasivaateaken hindab kovariatsioonimaatriksi, mida seejärel
kahandatakse (Ledoit-Wolf/OAS) ja söödetakse kumeruslahendajasse. Hinnang jääb
turust maha ning sisaldab akna pikkuse otsustust ja hindamisviga.

**Utah’ töövoog:** [Pidevaja topoloogiline allokatsioon](08-continuous-time-allocation.md)
**mõõdab** turu geomeetriat, mitte ei modelleeri seda:

- `QuadraticCovariationObserver` loeb rajapõhise meetrikatensori
  `g_ij(t) = d/dt ⟨X_i, X_j⟩_t` otse tiksuvoost — **ilma tagasivaateaknata, ilma
  mahajäämuseta** — koos kaheskaalalise (TSRV) hinnanguga mikrostruktuuri müra jaoks.
- `compute_ricci_flow_covariance` mürafiltreerib selle meetrika **täpse autodiff**
  normaliseeritud Ricci vooga (`jax.jacfwd`, ilma lõplike vahede puhverlahenduseta).
- `feynman_kac_drawdown_bound` muudab Laplace-Beltrami peamise omaväärtuse
  analüütiliseks väljavõtu ülempiiriks `P(sup DD > D_max) ≤ C e^{-λ₀T}`.
- `betti_number_divergence_test` märgistab nakkustopoloogiat: sünteetilise krahhi
  ajal variseb toore korrelatsioonimuutkond kokku (`b₀ → 1`), samal ajal kui
  detoneeritud topoloogilise riskipariteedi muutkond hoiab oma klastrid lahus.

## Migreerumine `riskfolio-lib`-ist

Pidevaja komplekt pakub geomeetrilisi vasteid tavalistele `riskfolio-lib`
töövoogudele:

| riskfolio-lib | Utah’ vaste |
|---------------|-------------|
| `HCPortfolio` (HRP/NCO) | `optimize_topological_risk_parity` |
| Ledoit-Wolf / OAS kahandus | `compute_ricci_flow_covariance` (täpne autodiff) |
| Valim/EWMA kovariatsioon | `QuadraticCovariationObserver` (rajapõhine, null mahajäämus) |
| L1 käibe piirang | `calculate_navier_stokes_rebalance_flow` |
| Mean-CVaR / EVaR / Max-DD | `apply_spectral_cvar_veto`, `feynman_kac_drawdown_bound` |

Geomeetria on täpne (valideeritud 2-sfääril kuni `1e-5`) ja rajapõhine meetrika on
$\mathcal{F}_t$-mõõdetav vaadeldav suurus, mitte hinnatud parameeter. Kui reaalsed
börsitiksud sisaldavad mikrostruktuuri müra, eemaldab kaheskaalaline hinnang
juhtiva nihke. Nagu iga uue meetodi puhul, valideeri olemasolevate
kumeruslähtejoonte vastu enne kapitali ümberjaotamist.

## Migratsiooniteekond Bloombergist / Enfusionist

1. **1. nädal:** ekspordi positsioonid platvormide kaupa → sööda `AssetPosition` loendisse
2. **2. nädal:** varjurežiimi kavatsused (ilma täitmiseta)
3. **3. nädal:** ühenda hoidla hetktõmmised adeelseks verifitseerimiseks
4. **4. nädal:** piloot aatomiline settlement sisemisel pearaamatul
5. **Jooksvalt:** järkjärguline EMS/PB voogude integreerimine
