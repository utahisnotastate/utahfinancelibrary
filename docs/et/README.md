# Utah Finance Library — dokumentatsiooni register

**Versioon 6.Omnibus_Adelic**  
**Repositoorium:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)

> **Keeled:** [English](../README.md) · Eesti · [Русский](../ru/README.md) · [日本語](../ja/README.md)

Tere tulemast Utah Finance Library ametliku dokumentatsiooni juurde. Need
juhendid on kirjutatud erinevatele rollidele ja kogemustasemetele. Alusta sellest
juhendist, mis sobib sinuga kõige paremini.

| Sihtrühm | Juhend | Mida õpid |
|----------|--------|-----------|
| Tarkvarainsenerid ja arhitektid | [Piiritletud taristu juurutamine](01-engineers-architects.md) | Repo struktuur, Utahfile, JAX PINN, verifitseerimisvõre, juurutamine |
| Kvandid, portfellihaldurid, kauplejad | [Operatiivse hõõrdumise lõpp](02-finance-professionals.md) | Alfa marsruutimine, tasaarveldus, settlement, adeelne möödaviik vs pärandsüsteemid |
| Asutajad ja perekontorid | [Absoluutne finantssuveräänsus](03-founders-family-offices.md) | Kulud, turvalisus, jaotusreeglid, migratsiooniteekond |
| Lapsed ja algajad | [Ülitark rikkumatu rahakassa](04-children-beginners.md) | Sõbralik selgitus raha marsruutimisest ja jagamisreeglitest |
| Vastavus ja operatsioonid | [Settlement’i ja kümnise protokoll](05-settlement-governance.md) | 2,3% kümnis, 5,7% humanitaarmäär, auditikonksud |
| Edasijõudnud / SOTA | [Adeelne kliiringukoja möödaviik](06-adelic-clearinghouse-bypass.md) | Hasse-Minkowski null-tagatise mudel |
| Edasijõudnud / SOTA | [Laine-oleku PINN alfamootor](07-wave-pinn-alpha-engine.md) | Füüsikateadlikud närvivõrgud JAX-is |
| Edasijõudnud / SOTA | [Pidevaja topoloogiline allokatsioon](08-continuous-time-allocation.md) | TRP, Ricci voog, tiksuvaatleja, NS marsruutimine, Laplace-Beltrami väljavõtu piir |
| Kõigile | [Sõnastik](glossary.md) | Raamatukogus kasutatavad mõisted |

> **Teooria-lisad (LaTeX):** matemaatilised teoreemid `09`–`13`
> ([`09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex),
> [`10_Entanglement_Hedging.tex`](../10_Entanglement_Hedging.tex),
> [`11_Jarzynski_Harvesting.tex`](../11_Jarzynski_Harvesting.tex),
> [`12_Braid_Execution.tex`](../12_Braid_Execution.tex),
> [`13_Koopman_Linearization.tex`](../13_Koopman_Linearization.tex)) on hoitud
> keeleneutraalsetena (universaalne matemaatiline tähistus), seega neid ei
> dubleerita iga keele jaoks.

## Uurimismoodulid / SOTA füüsika

Lisaks ülaltoodud juhenditele sisaldab raamatukogu uurimistaseme mooduleid, mis
laenavad matemaatilise füüsika tööriistu. Iga moodul on **päris, testitud
hindaja/diagnostika** koos ausa raamistusega (ei mingit imevõitu, oraaklit ega
nulli-libisemise lubadust): `holographic_projection.py` (AdS/CFT LOB-rõhk),
`tensor_network_hedge.py` (MPS / Von Neumanni entroopia), `chrono_drift.py`
(Malliavin/Skorokhod), `jarzynski_harvester.py` (Jarzynski vaba energia),
`braid_router.py` (Kauffman/Jones + täitmisjärjestus), `koopman_oracle.py`
(Koopman/EDMD prognoos). Kogu positiivne tulu marsruuditakse läbipaistva
`enforce_universal_tithe` kaudu.

## Utah’le maksmine

2,3% protokolli kümnis ja humanitaarjaotus on raamatukogu sisesed
raamatupidamismarsruudid. **Tegeliku makse tegemiseks Utah’le** — sponsorlus,
kümnise ülekanne või toetus — võta otse ühendust:

> **Kontakt:** [utah@utahcreates.com](mailto:utah@utahcreates.com)

Praegu on see käsitsi inimsamm (e-kiri meetodi, viite ja summa kokkuleppimiseks).
**Planeeritud on eraldi GUI-rakendus**, mis haldab Utah’le maksmist algusest
lõpuni — koostab ülekande andmed, jälgib kümnise/humanitaarjaotust ja salvestab
kviitungid — nii et sellest saab paari kliki asi. Kuni see ilmub, on ülaltoodud
e-post ametlik viis Utah’le maksta.

## Kiirkäsud

```bash
# Täielik konveier (audit → marsruutimine → settlement → adeelne → alfa)
python -m src.app.utah_prime_sieve_daemon

# Adeelne null-tagatise demo
python -m src.app.hasse_minkowski_daemon

# JAX alfamootor (nõuab pip install -e ".[jax]")
python -m src.models.pinn_jax_runtime

# Valideeri Utahfile v6 konksud
python -m src.app.ignite --manifest Utahfile --dry-run
```

## Oluline lahtiütlus

Utah Finance Library pakub **avatud lähtekoodiga tarkvara komponente** portfelli
auditeerimiseks, marsruutimiseks, settlement-juhiste loomiseks ja matemaatiliseks
verifitseerimiseks. See **ei**:

- asenda litsentseeritud kliiringukodasid ega regulatiivset aruandlust;
- garanteeri investeeringutulu ega riski kõrvaldamist;
- ole õigus-, maksu- ega investeerimisnõustamine.

Tootmiskasutus eeldab sinu enda vastavuskontrolli, õiguslikku ülevaadet ja
integreerimist heakskiidetud finantstaristuga.
