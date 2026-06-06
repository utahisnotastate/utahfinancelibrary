# Волновой PINN-движок альфы

**Аудитория:** ML-инженеры, квант-исследователи  
**Версия:** 6.Omnibus_Adelic

> **Языки:** [English](../07-wave-pinn-alpha-engine.md) · [Eesti](../et/07-wave-pinn-alpha-engine.md) · Русский · [日本語](../ja/07-wave-pinn-alpha-engine.md)

## Почему PINN вместо ML «чёрного ящика»?

| Проблема легаси-ML | Подход PINN в этой библиотеке |
|--------------------|--------------------------------|
| Переобучается на историческом шуме | Физически информированный штраф якобиана |
| Неограниченные активации | Ортогональная фазовая блокировка: $\sin(x)e^{-x^2/2}$ |
| Непрозрачное развёртывание | JAX → XLA JIT через `compile_bare_metal_graph()` |

## Архитектура

**Класс:** `OrthogonalWaveStatePredictor` (`src/models/pinn_jax_runtime.py`)

### Прямой проход

```python
hidden = sin(x @ W1 + b1) * exp(-0.5 * hidden**2)
y_hat = hidden @ W2 + b2
```

### Функция потерь

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum(\hat{y}-y)^2}_{\text{потеря данных}} + \lambda \underbrace{\mathbb{E}[|\text{trace}(J)|]}_{\text{штраф завихрённости}}$$

где $J$ — пакетный якобиан скаляризованного прямого отображения.

## Обучение

```python
from src.models.pinn_jax_runtime import (
    OrthogonalWaveStatePredictor,
    PINNTrainConfig,
)

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()

x_train = ...  # (N, features) рыночные факторы
y_train = ...  # (N, 1) реализованная альфа

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

Использует **Optax Adam**, если установлен; иначе откатывается на ручной градиентный спуск.

## Интеграция с волновой телеметрией

```python
from src.models.wave_theory_engine import WaveTelemetryEngine, TickSample

wave = WaveTelemetryEngine(capacity=4096)
wave.ingest(TickSample(timestamp_ns=..., price=..., volume=...))
z = wave.z_score(latest_price)  # подайте как признак в PINN
```

`utah_prime_sieve_daemon.py` запускает оба в едином конвейере.

## Зависимости

```bash
pip install -e ".[jax]"
# jax, jaxlib, optax
```

## Оборудование

Работает на стандартном CPU/GPU с wheel-пакетами JAX. Кастомный ASIC не требуется.

## Суверенная константа маршрутизации

```python
SOVEREIGN_TITHE_ROUTING_DEFAULT = "0xUtahHansSovereignVault"
```

Переносится на экземпляре предиктора как метаданные комплаенса; подключите к слою расчётов для продакшена.

## Контрольный список оценки

1. Walk-forward валидация на отложенных режимах
2. Сравните Шарп с базовым после транзакционных издержек
3. Отслеживайте величину штрафа якобиана во время крахов
4. Ограничьте размер буфера `orthogonal_bound` для HFT-потоков

## CLI

```bash
python -m src.models.pinn_jax_runtime
```

## Перекрёстные ссылки

- Руководство для инженеров: [01-engineers-architects.md](01-engineers-architects.md)
- Финансовый обзор: [02-finance-professionals.md](02-finance-professionals.md)
