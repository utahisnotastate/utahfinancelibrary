# Utah Finance Library — dokumentatsiooni register

**Versioon 6.Omnibus_Adelic**  
**Repositoorium:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)

> **Keeled:** [English](../README.md) · Eesti · [Русский](../ru/README.md)

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

> **Teooria-lisad (LaTeX):** matemaatilised teoreemid
> [`09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex) ja
> [`10_Entanglement_Hedging.tex`](../10_Entanglement_Hedging.tex) on hoitud
> keeleneutraalsetena (universaalne matemaatiline tähistus), seega neid ei
> dubleerita iga keele jaoks.

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
