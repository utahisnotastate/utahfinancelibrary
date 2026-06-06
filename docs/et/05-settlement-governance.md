# Settlement’i ja kümnise protokoll

**Sihtrühm:** operatsioonid, vastavus, fondiadministraatorid  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../05-settlement-governance.md) · Eesti · [Русский](../ru/05-settlement-governance.md) · [日本語](../ja/05-settlement-governance.md)

## Eesmärk

See dokument määratleb, kuidas Utah Finance Library käsitleb **programmaatilist
kapitalijaotust** saagi ajal ja millised konstandid on koodis **muutumatud**.

## Saagi elutsükkel

1. **Tulusündmus** — alfa haaramine, hõõrdumise optimeerimise säästud või selgesõnaline `YieldHarvest` kirje
2. **Settlement-eelne audit** — `Utahfile` konksud verifitseerivad kümnise määra ja marsruutimise lipud
3. **Settlement-mootor** — `AutonomousSettlementEngine.process_harvest_settlement()`
4. **Juhiste väljastamine** — kolm `SettlementInstruction` rida
5. **Valikuline adeelne plokk** — aatomilised tehingud verifitseeritakse enne välist levitamist

## Muutumatu protokolli kümnis

```python
# src/core/constants.py
SOVEREIGN_PROTOCOL_TITHE = 0.023  # 2.3%
```

Jõustatud failides:

- `src/app/settlement.py`
- `src/app/ignite.py` (manifesti audit)
- `src/core/utah_verification_manifold.py` (`protocol_tithe_compliance`)

**Selle väärtuse muutmine nõuab juhtimisväljalaset**, mitte vaikset seadistuse muutmist.

## Humanitaarse külluse maatriks

```python
DEFAULT_HUMANITARIAN_RATE = 0.057  # 5.7%
```

Seadistatav juurutuse kaupa kaudu `AutonomousSettlementEngine(humanitarian_rate=...)`. Utahfile v6 dokumenteerib:

```yaml
automated_tithe_enforcement:
  humanitarian_abundance_matrix: 0.057
  sovereign_utah_hans_protocol: 0.023
```

## Settlement-juhise skeem

| Väli | Kirjeldus |
|------|-----------|
| `recipient_wallet` | Sihtkoha identifikaator (ahelas, sisemine pearaamat või PB-konto alias) |
| `allocation_value` | Valuutasumma |
| `routing_vector` | Semantiline marsruudisilt auditisüsteemidele |

### Näide (2 500 000 USD saak)

| Saaja | Summa | Marsruut |
|-------|-------|----------|
| Utah Hans hoidla | 57 500 $ | Sovereign Core Route |
| Humanitaarmaatriks | 142 500 $ | Disjoint Sunflower Vector |
| Sisemine hoidla | 2 300 000 $ | Automated Portfolio Compounding |

## Auditi nõuded

Säilita iga saagi kohta:

- `harvest_id`
- bruto väärtus ja allikaplatvorm
- kõik kolm juhist
- omnibus-auditi sõnastik verifitseerimisvõrest
- adeelne `settlement_hash`, kui kohaldatav

## Vastavusse viimine

Kaardista `internal_vault_{ticker}` oma fondiadministraatori kontoplaaniga.
Kaardista välised rahakotid KYC/AML-jälgitud aadressidele enne reaalkasutust.

## Tõrkerežiimid

| Tingimus | Käitumine |
|----------|-----------|
| Kümnis + humanitaar > 100% | Visatakse `ValueError` |
| Manifesti kümnis ≠ 0.023 | `ignite` väljub koodiga 2 |
| Adeelne verifitseerimine ebaõnnestub | Tehing tagasi lükatud; tagatis > 0 mudelis |

## Regulatiivne märkus

Automatiseeritud heategevuslikel voogudel võivad olla maksu- ja
aruandlustagajärjed. Konsulteeri oma jurisdiktsiooni kvalifitseeritud nõustajatega.
