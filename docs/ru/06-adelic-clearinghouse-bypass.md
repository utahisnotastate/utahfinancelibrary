# Адельный обход клиринговой палаты (расчёт Хассе-Минковского)

**Аудитория:** квант-инженеры, архитекторы расчётов, продвинутые читатели  
**Версия:** 6.Omnibus_Adelic

> **Языки:** [English](../06-adelic-clearinghouse-bypass.md) · [Eesti](../et/06-adelic-clearinghouse-bypass.md) · Русский

## Проблема

Центральные контрагенты (CCP) и прайм-брокеры требуют **залог**, чтобы
гарантировать расчёт T+1. В институциональном масштабе это блокирует **сотни
миллионов до миллиардов** в низкодоходном залоге.

## Модель (вдохновение принципом Хассе)

**Принцип Хассе** (локально-глобальный принцип) в теории чисел: при подходящих
условиях, если диофантово уравнение имеет решения в $\mathbb{R}$ и во всех
$\mathbb{Q}_p$, оно имеет рациональное решение.

Мы адаптируем это **метафорически** для расчётов:

| Поле | Реализация в коде |
|------|-------------------|
| Вещественное | Балансы `VaultSnapshot` покрывают номинальные обязательства |
| Локальное ($\mathbb{Q}_p$ суррогат) | Проверки целочисленного сравнения по модулю простых $p \in \{2,3,5,7,11,13\}$ |
| Глобальный проход | Вещественная И все локальные проверки успешны |

Когда `global_passed=True`, `required_collateral=0.0` и
`AdelicClearinghouseEngine` выполняет **атомарный обмен** — исполнение и расчёт
являются одним событием.

## Справочник API

### Типы

- `VaultSnapshot(vault_id, balances)`
- `AtomicTrade(trade_id, buyer_vault_id, seller_vault_id, base_asset, quote_asset, quantity, price)`
- `HasseVerificationResult` — аудиторская запись с `settlement_hash`

### Основные классы

```python
verifier = HasseMinkowskiVerifier(primes=(2, 3, 5, 7, 11))
result = verifier.verify(trade, buyer_snapshot, seller_snapshot)

engine = AdelicClearinghouseEngine()
engine.register_vault(...)
engine.attempt_atomic_settlement(trade)
```

## Хеш расчёта

SHA-256 по каноническому JSON:

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

Используйте для неизменяемых аудиторских следов.

## Интеграция Utahfile

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    execution_command: "python -m src.app.hasse_minkowski_daemon"
    collateral_requirement: 0.00
```

## Демо-сценарий

`hasse_minkowski_daemon.py` симулирует:

- **Фонд Alpha:** 600 млн $ USD
- **Фонд Beta:** 10 000 WETH
- Сделка: 1000 WETH @ 2500 $

Избегнутый легаси-моделируемый залог: **2,5 млрд $ номинала** (демо-метрика = `quantity × price` на рассчитанных сделках).

## Ограничения (читайте внимательно)

1. **Упрощённая $p$-адическая проверка** — не полная математика кольца аделей
2. **Реестр в памяти** — не замена CCP без правовой базы
3. **Нет DvP в реальных цепях** — интегрируйте с кастоди и юридическим ISDA/GMRA
4. **Набор простых настраивается** — рекомендуется анализ чувствительности

## Дорожная карта расширения

- Подключить реальные балансы хранилища из `Omnibus-Cryptographic-Custody`
- Сохранять аудиторский след в хранилище только для добавления
- Графы кросс-маржи по нескольким активам
- Партия неттинга через подсолнечный маршрутизатор перед адельной верификацией

## CLI

```bash
python -m src.app.hasse_minkowski_daemon
python -m src.app.hasse_minkowski_daemon --json
```
