# Развёртывание ограниченной трансфинитной инфраструктуры

**Аудитория:** инженеры-программисты, DevOps, системные архитекторы  
**Версия:** 6.Omnibus_Adelic

> **Языки:** [English](../01-engineers-architects.md) · [Eesti](../et/01-engineers-architects.md) · Русский · [日本語](../ja/01-engineers-architects.md)

## Обзор

Utah Finance Library — это **многомодульный Python-монорепозиторий**, который
заменяет монолитные стеки управления фондами детерминированными, тестируемыми
компонентами:

| Модуль | Путь | Ответственность |
|--------|------|-----------------|
| Решето капитала | `src/core/capital_sieve.py` | Межплощадочное трение доходности → планы ребалансировки |
| Подсолнечный маршрутизатор | `src/core/sunflower_router.py` | Непересекающиеся лепестки ликвидности (разбиения с ограничением $k$) |
| Решётка верификации | `src/core/utah_verification_manifold.py` | Выравнивание Навье, граница адельного решета, омнибус-аудит |
| Адельный клиринг | `src/core/adelic_clearing.py` | Локально-глобальная верификация расчёта Хассе-Минковского |
| Расчёты | `src/app/settlement.py` | Распределение урожая (десятина + гуманитарная + реинвест) |
| PINN альфа | `src/models/pinn_jax_runtime.py` | JAX физически информированный волновой предиктор |
| Волновая телеметрия | `src/models/wave_theory_engine.py` | Статистика тиков в ограниченном кольцевом буфере |
| Наблюдатель тиков | `src/core/tick_observer.py` | Путевая метрика квадратической ковариации `g_ij(t)` |
| Ядро многообразия | `src/models/manifold_kernel.py` | Точный autodiff поток Риччи для шумоподавления ковариации |
| Геометрия Римана | `src/models/riemannian_geometry.py` | Кристоффель / Риман / Риччи через `jax.jacfwd` |
| Топологическое распределение | `src/core/topological_allocation.py` | Топологический риск-паритет на персистентной гомологии + дивергенция Бетти |
| Супервизор риска | `src/core/risk_supervisor.py` | Спектральный CVaR + граница просадки Лапласа-Бельтрами |
| Хранилище | `src/app/vault_daemon.py` | Распространение намерений с пороговой подписью |
| Оркестратор | `src/app/alpha_orchestrator.py` | Защитные ограждения риска + автоматический выключатель |

## Требования

- Python 3.10+
- Windows: используйте `py -3`, если `python` нет в PATH
- Опциональный стек JAX: `pip install -e ".[jax,dev]"`

## Установка

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Структура репозитория

```text
/
├── Utahfile                 # Манифест развёртывания (v6)
├── .cursorrules             # Инженерный мандат Cursor AI
├── src/
│   ├── core/                # Математические + финансовые ядра
│   ├── app/                 # Демоны и точки входа CLI
│   └── models/              # JAX PINN + волновая телеметрия
├── tests/                   # Набор pytest
└── docs/                    # Этот набор документации
```

## Utahfile v6

`Utahfile` — единственный манифест развёртывания. Ключевые дополнения v6:

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    collateral_requirement: 0.00
```

`src/app/ignite.py` проверяет:

- десятину перед расчётом `0.023`
- автоматизированную гуманитарную ставку `0.057`
- нулевой залог на адельном сервисе
- строку адельного протокола расчёта

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
```

## OrthogonalWaveStatePredictor (JAX / XLA)

Движок альфы находится в `src/models/pinn_jax_runtime.py`.

**Прямой проход** использует ортогональную активацию фазовой блокировки:

$$\text{hidden} = \sin(z) \cdot e^{-z^2/2}, \quad z = x W_1 + b_1$$

**Функция потерь** объединяет MSE и штраф завихрённости, выведенный из пакетного якобиана:

$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda \cdot \mathbb{E}[|\text{trace}(J)|]$$

Скомпилируйте для развёртывания:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor, PINNTrainConfig

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()
params, predict = predictor.train(market_x, alpha_y, PINNTrainConfig(steps=200))
```

## Стек геометрии непрерывного времени (JAX обязателен)

Геометрические модули v6 заменяют статическую оценку ковариации точной
дифференциальной геометрией непрерывного времени. Шумоподавитель потока Риччи,
тензоры Кристоффеля/Римана/Риччи и граница просадки Лапласа-Бельтрами работают
**только на JAX** — резервного варианта на конечных разностях NumPy нет,
поскольку ошибка усечения `O(h^2)` искажает нелинейное PDE потока.

```python
import numpy as np
from src.core.tick_observer import QuadraticCovariationObserver
from src.models.manifold_kernel import compute_ricci_flow_covariance
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

# 1. Измерить путевой метрический тензор из тиков (нулевое запаздывание)
obs = QuadraticCovariationObserver(n_assets=N)
obs.ingest_prices(price_path, dt=1 / len(price_path))
g = obs.metric_tensor()                       # g_ij(t) = d/dt <X_i, X_j>_t

# 2. Шумоподавление через точный autodiff нормированный поток Риччи (без прокси на пути)
denoised = compute_ricci_flow_covariance(g, flow_duration=1.0, manifold_dimension=N)

# 3. Абсолютное вето просадки из измеренной метрики
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

Проверьте точность относительно аналитических эталонов:

```bash
pytest tests/test_riemannian_geometry.py   # 2-сфера R = 2/r^2 с точностью 1e-5
pytest tests/test_ricci_flow_autodiff.py   # прокси исключён из пути; шумоподавление спектра
pytest tests/test_tick_observer.py         # сходимость dt->0, независимость от сноса
```

Полная теория: [08-continuous-time-allocation.md](08-continuous-time-allocation.md)
и [теорема стабилизации потока Риччи](../09_Ricci_Flow_Stabilization.tex).

## Интеграция адельного клиринга

```python
from src.core.adelic_clearing import AdelicClearinghouseEngine, AtomicTrade, VaultSnapshot

engine = AdelicClearinghouseEngine()
engine.register_vault(VaultSnapshot("buyer", {"USD": 1e9}))
engine.register_vault(VaultSnapshot("seller", {"WETH": 500}))
result = engine.attempt_atomic_settlement(AtomicTrade(...))
assert result.zero_collateral  # если локально-глобальные проверки пройдены
```

## Решётка верификации

Запустить автономно:

```bash
python -m src.core.utah_verification_manifold
```

Проверки включают:

1. **Геометрическое исчерпание Навье** — завихрённость против промежуточного собственного вектора тензора скорости деформации
2. **Интервал адельного решета** — $Y(x) = \Theta(x \log x \log \log x)$ против легаси-потолка $O(x^2)$
3. **Омнибус-аудит** — ставка десятины, спектральная жёсткость, непересекающаяся маршрутизация

## Тестирование

```bash
pytest -q
pytest tests/test_pinn.py                  # PINN альфа (требует доп. JAX)
pytest tests/test_riemannian_geometry.py   # точная кривизна (требует доп. JAX)
pytest tests/test_manifold_kernel.py tests/test_ricci_flow_autodiff.py
pytest tests/test_tick_observer.py tests/test_laplace_beltrami.py
```

## Внешняя экосистема

Предназначена для интеграции с [github.com/utahisnotastate](https://github.com/utahisnotastate):

- `utahcontainerengine` — продакшен-рантайм юникернела (локальная замена: `ignite.py`)
- `wavetheory_5` — паттерны телеметрии, отражённые в `wave_theory_engine.py`

## Контрольный список для продакшена

1. Замените демо-хранилища/позиции на API-потоки
2. Сохраняйте хеши расчётов и аудиторский след в неизменяемое хранилище
3. Подключите `SovereignVault` к реальной подписи TSS/HSM
4. Проведите юридическую проверку комплаенса перед заявлением о нулевом клиринговом залоге
5. Зафиксируйте версии зависимостей в продакшен-образах
