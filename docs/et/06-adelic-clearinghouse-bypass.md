# Adeelne kliiringukoja möödaviik (Hasse-Minkowski settlement)

**Sihtrühm:** kvantinsenerid, settlement-arhitektid, edasijõudnud lugejad  
**Versioon:** 6.Omnibus_Adelic

> **Keeled:** [English](../06-adelic-clearinghouse-bypass.md) · Eesti · [Русский](../ru/06-adelic-clearinghouse-bypass.md) · [日本語](../ja/06-adelic-clearinghouse-bypass.md)

## Probleem

Kesksed vastaspooled (CCP-d) ja primaarmaaklerid nõuavad **tagatist**, et
garanteerida T+1 settlement. Institutsionaalses mastaabis lukustab see **sadu
miljoneid kuni miljardeid** madala tulususega tagatisse.

## Mudel (Hasse printsiibist inspireeritud)

**Hasse printsiip** (lokaal-globaalne printsiip) arvuteoorias: sobivatel
tingimustel, kui Diofantilisel võrrandil on lahendid $\mathbb{R}$-s ja kõigis
$\mathbb{Q}_p$-des, siis on tal ratsionaalne lahend.

Me kohandame seda **metafoorselt** settlement’i jaoks:

| Väli | Teostus koodis |
|------|----------------|
| Reaal | `VaultSnapshot` saldod katavad nominaalkohustused |
| Lokaalne ($\mathbb{Q}_p$ asendus) | Skaleeritud täisarvu kongruentsi kontrollid algarvude $p \in \{2,3,5,7,11,13\}$ mooduli järgi |
| Globaalne läbimine | Reaal JA kõik lokaalsed kontrollid õnnestuvad |

Kui `global_passed=True`, on `required_collateral=0.0` ja
`AdelicClearinghouseEngine` teeb **aatomilise vahetuse** — täitmine ja settlement
on üks sündmus.

## API viide

### Tüübid

- `VaultSnapshot(vault_id, balances)`
- `AtomicTrade(trade_id, buyer_vault_id, seller_vault_id, base_asset, quote_asset, quantity, price)`
- `HasseVerificationResult` — auditikirje koos `settlement_hash`-iga

### Põhiklassid

```python
verifier = HasseMinkowskiVerifier(primes=(2, 3, 5, 7, 11))
result = verifier.verify(trade, buyer_snapshot, seller_snapshot)

engine = AdelicClearinghouseEngine()
engine.register_vault(...)
engine.attempt_atomic_settlement(trade)
```

## Settlement-räsi

SHA-256 üle kanoonilise JSON-i:

```json
{
  "trade_id": "...",
  "buyer": "...",
  "seller": "...",
  "base": "WETH",
  "quote": "USD",
  "qty": 1000.0,
  "price": 2500.0,
  "global_passed": true
}
```

Kasuta muutumatute auditijälgede jaoks.

## Utahfile integratsioon

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    execution_command: "python -m src.app.hasse_minkowski_daemon"
    collateral_requirement: 0.00
```

## Demo-stsenaarium

`hasse_minkowski_daemon.py` simuleerib:

- **Fond Alpha:** 600 mln $ USD
- **Fond Beta:** 10 000 WETH
- Tehing: 1000 WETH @ 2500 $

Välditud pärand-modelleeritud tagatis: **2,5 mld $ nominaal** (demo-meetrika = `quantity × price` arveldatud tehingutel).

## Piirangud (loe hoolikalt)

1. **Lihtsustatud $p$-aadiline kontroll** — mitte täielik adeelirengu matemaatika
2. **Mälusisene pearaamat** — pole CCP asendus ilma õigusliku raamistikuta
3. **Pole DvP päris ahelates** — integreeri hoidla ja õigusliku ISDA/GMRA-ga
4. **Algarvude komplekt on seadistatav** — soovituslik on tundlikkusanalüüs

## Laienduse teekaart

- Konksu reaalsed hoidlasaldod `Omnibus-Cryptographic-Custody`-st
- Säilita auditijälg ainult-lisamise hoidlasse
- Mitme vara ristmarginaali graafid
- Tasaarvelduspartii päevalille-marsruuteri kaudu enne adeelset verifitseerimist

## CLI

```bash
python -m src.app.hasse_minkowski_daemon
python -m src.app.hasse_minkowski_daemon --json
```
