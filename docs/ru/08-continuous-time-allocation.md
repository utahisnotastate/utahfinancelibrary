# Топологическое распределение в непрерывном времени

**Аудитория:** квант-исследователи, портфельные инженеры, мигрирующие с `riskfolio-lib`  
**Версия:** 6.Omnibus_Adelic

> **Языки:** [English](../08-continuous-time-allocation.md) · [Eesti](../et/08-continuous-time-allocation.md) · Русский · [日本語](../ja/08-continuous-time-allocation.md)

Этот набор модулей переосмысливает оптимизацию портфеля, уходя от статического
квадратичного программирования на дискретных матрицах ковариации к **геометрии
непрерывного времени**: персистентная гомология, поток Риччи, маршрутизация
жидкости и спектральные границы риска.

Библиотека не *моделирует* рынок. Она **измеряет** топологическую кривизну рынка
по мере его развёртывания и вычисляет абсолютные спектральные границы этой
геометрии. Метрический тензор — не оценённый параметр, а
$\mathcal{F}_t$-измеримая наблюдаемая величина, считываемая прямо из потока тиков
(см. теорему ниже и `src/core/tick_observer.py`).

### Теорема — путевая точность метрического тензора портфеля

Пусть вектор состояния рынка $X_t$ — непрерывный семимартингал, адаптированный к
эмпирической фильтрации $\mathcal{F}_t$. Риманов метрический тензор $g_{ij}(t)$,
определяющий генератор Лапласа-Бельтрами $\mathcal{L}$ многообразия портфеля,
точно определяется квадратической ковариацией непрерывного времени:

$$g_{ij}(t) = \frac{d}{dt}\,\langle X_i, X_j\rangle_t.$$

Когда высокочастотный интервал наблюдения $dt \to 0$, эмпирическое измерение
$g_{ij}(t)$ сходится к истинной метрике (по вероятности и почти наверное вдоль
измельчающихся разбиений).

**Доказательство.** По определению процесса квадратической ковариации для
непрерывных семимартингалов предел суммы квадратов приращений сходится путевым
образом к точной матрице ковариации, **независимо от вектора сноса** $b$. Поэтому
$g_{ij}$ — не параметрическое предположение, подверженное «суждению
моделирования»; это геометрический инвариант, извлечённый из
$\mathcal{F}_t$-измеримого потока тиков. Следовательно, главное собственное
значение $\lambda_0$ оператора $\mathcal{L}$ даёт физический супремум на область
просадки через Фейнмана-Каца, свободный от ошибки оценки метрики. $\blacksquare$

Независимость от сноса этого объекта проверяется напрямую в
`tests/test_tick_observer.py::test_drift_independence`, а сходимость $dt \to 0$ — в
`test_realized_covariation_converges_as_dt_shrinks`.

**Практическая реализация (микроструктура тиков).** На реальной биржевой ленте
наблюдаемая цена — это латентный семимартингал плюс i.i.d. шум микроструктуры,
который смещает наивную реализованную ковариацию вверх при $dt \to 0$. Это
*инженерный* артефакт датчика, а не эпистемический пробел в метрике: библиотека
применяет состоятельную двухмасштабную оценку реализованной ковариации
(`two_scale_realized_covariance`), которая сокращает ведущий шумовой член и
сходится к той же интегрированной ковариации. Метрика остаётся наблюдаемой
величиной; мы просто считываем её шумоустойчивым прибором.

## Наблюдатель тиков — привязка генератора к путевой метрике

`src/core/tick_observer.py` измеряет $g_{ij}(t)$ прямо из потока тиков и подаёт её
в генератор и рантайм PINN. **Без среднего по окну ретроспективы** (которое вносит
запаздывание и выбор длины окна); метрика — это мгновенный путевой предел.

```python
from src.core.tick_observer import QuadraticCovariationObserver
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

obs = QuadraticCovariationObserver(n_assets=N)        # или decay<1 для стох-вол
for log_price_vec, dt in tick_stream:
    obs.ingest(log_price_vec, dt=dt)                  # online, O(N^2) на тик

g = obs.metric_tensor()                                # точная измеренная метрика

# абсолютное вето просадки из измеренной метрики (без окна оценки)
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

Та же метрика отбеливает входы PINN во внутренние координаты:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

predictor = OrthogonalWaveStatePredictor()
predictor.bind_tick_metric(obs)        # g(t) = <X_i, X_j>_t / t
x_intrinsic = predictor.whiten(raw_factors)   # g^{-1/2} x
```

- `realized_covariation` — пакетная квадратическая ковариация из ценового пути.
- `two_scale_realized_covariance` — шумоустойчивый TSRV для сырых биржевых тиков.
- `QuadraticCovariationObserver` — онлайн, путевой наблюдатель без запаздывания.
- `drawdown_metric_from_covariation` — проекция портфеля $w^\top\Sigma w$.

## Обновление v6.1 — непрерывная геометрия, JAX обязателен

Три обновления переводят набор с дискретных суррогатов на точную непрерывную
геометрию (полные детали в разделах 5–7 ниже):

1. **Точная кривизна через autodiff** (`src/models/riemannian_geometry.py`) —
   символы Кристоффеля, тензоры Римана/Риччи и скалярная кривизна, вычисленные
   через autodiff JAX до машинной точности. **Без конечных разностей, без
   `O(h^2)`.** Проверено относительно круглой 2-сферы ($R = 2/r^2$,
   $\mathrm{Ric} = g/r^2$) с точностью `1e-5`.
2. **Непрерывная граница просадки Лапласа-Бельтрами** (`src/core/risk_supervisor.py`)
   — главное собственное значение Дирихле $\lambda_0$ генератора портфеля
   $\mathcal{L} = \tfrac12\Delta_M + b\cdot\nabla$ через гладкий спектральный
   базис Галёркина (экспоненциальная сходимость), питающее границу Фейнмана-Каца
   $\mathbb{P}(\sup \text{DD} > \mathcal{D}_{max}) \le C e^{-\lambda_0 T}$.
   Проверено относительно $\lambda_0 = \tfrac12(\pi/L)^2$ на плоском интервале.
3. **Строгий режим JAX** — `require_jax=True` и autodiff-геометрия отказываются
   работать без JAX, вместо того чтобы тихо использовать тензорное исчисление
   NumPy с меньшей точностью.

## Обзор

| API | Файл | Заменяет (riskfolio) |
|-----|------|----------------------|
| `optimize_topological_risk_parity` | `src/core/topological_allocation.py` | HRP / NCO |
| `compute_ricci_flow_covariance` | `src/models/manifold_kernel.py` | Сжатие Ledoit-Wolf / OAS, шумоподавление |
| `calculate_navier_stokes_rebalance_flow` | `src/core/sunflower_router.py` | Ограничения оборота L1 / транзакционных издержек |
| `apply_spectral_cvar_veto` | `src/core/risk_supervisor.py` | Mean-CVaR / EVaR / Max-Drawdown |

---

## 1. Топологический риск-паритет (TRP)

Строит корреляционное расстояние Мантеньи $d_{ij}=\sqrt{2(1-\rho_{ij})}$ и
фильтрацию Вьеториса-Рипса над ним.

- Персистентность **H0** вычисляется *точно* через union-find одиночной связи.
- **H1** (циклы / петли заражения) использует ранг циклов 1-скелета $E - V + b_0$.
- **H2** использует суррогат полости заполненного треугольника.

Капитал распределяется **обратно пропорционально** топологической нагрузке каждого
актива (мосты слияний + степень цикла), затем смешивается к $1/N$ через член
вассерштейновского штрихкода.

```python
from src.core.topological_allocation import optimize_topological_risk_parity

weights = optimize_topological_risk_parity(
    tensor_data=returns,            # (T, N) array-like (NumPy или jax.Array)
    max_homology_dimension=2,
    wasserstein_penalty=0.01,
)
```

Богатая диагностика:

```python
from src.core.topological_allocation import topological_risk_parity_report
report = topological_risk_parity_report(returns)
print(report.betti_numbers, report.barcode_wasserstein, report.filtration_radius)
```

---

## 2. Ковариация потока Риччи

Интегрирует нормированный по объёму поток Риччи

$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \tfrac{2}{m} r\, g_{ij}$$

как **прямой мост к точному autodiff потоку метрического поля** — на пути нет
прокси на конечных разностях NumPy. Постоянная матрица ковариации — это *плоская*
метрика (нулевой Риччи), поэтому API сначала строит искривлённое риманово
метрическое *поле*, кривизна которого индуцирована спектром ковариации, затем
интегрирует поток с символами Кристоффеля и тензором Риччи, полученными через
`jax.jacfwd` до машинной точности. Спектр стягивается к своему среднему по
**измеренному коэффициенту унификации кривизны**

$$\gamma = \frac{\operatorname{std}_x R(T)}{\operatorname{std}_x R(0)} \in [0,1],
\qquad \log\lambda_i^{\text{den}} = \overline{\log\lambda} + \gamma\,(\log\lambda_i - \overline{\log\lambda}),$$

так что стягивание управляется точной геометрией потока ($\gamma \to 0$ по мере
того как скалярная кривизна становится постоянной), а не дискретным суррогатом.
След (общая дисперсия) сохраняется. `ricci_curvature_proxy` остаётся отдельной
спектральной диагностикой, но **никогда** не вызывается шумоподавителем
(обеспечивается в
`tests/test_ricci_flow_autodiff.py::test_api_does_not_call_numpy_proxy`).

> Теорема унификации кривизны и её точное соответствие реализованному
> коэффициенту стягивания $\gamma$ формализованы в
> [`docs/09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex).

```python
from src.models.manifold_kernel import compute_ricci_flow_covariance

denoised = compute_ricci_flow_covariance(
    empirical_metric_tensor=cov,    # (N, N) SPD
    flow_duration=1.0,
    manifold_dimension=cov.shape[0],
)
```

Точная (измеренная autodiff) траектория дисперсии скалярной кривизны для
`wave_theory_engine`:

```python
from src.models.manifold_kernel import ricci_flow_curvature_field
times, curv_dispersion = ricci_flow_curvature_field(cov, 1.0, cov.shape[0], samples=8)
```

Эмпирически поток **стягивает разброс собственных значений** (шумоподавление),
сохраняя SPD-структуру и след — см. `tests/test_manifold_kernel.py` и
`tests/test_ricci_flow_autodiff.py`.

---

## 3. Маршрутизация ликвидности Навье-Стокса

Моделирует ребалансировку как несжимаемый поток Стокса: вязкость = рыночное
воздействие, объёмная сила = градиент альфы. Решает седловую (KKT) систему, так что
поле скорости **сохраняет массу** ($\mathbf{1}^\top v = 0$).

```python
from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow

velocity, pressure_gradient = calculate_navier_stokes_rebalance_flow(
    current_weights=w_now,
    target_manifold=w_target,
    market_viscosity_tensor=0.1,        # скаляр, (N,) или (N, N)
    kinematic_constraints={"max_velocity": 0.25, "coupling": 1.0},
)
```

Возвращает непрерывное **поле скорости** исполнения (как быстро/куда течёт капитал)
и **градиент давления**, маршрутизирующий арбитраж — а не просто статические
целевые веса.

---

## 4. Вето CVaR на собственном многообразии (спектральное)

Дискретизирует оператор шрёдингеровского типа $-\Delta + V(x)$ с граничными
условиями Дирихле, где $V$ сэмплируется из вашего `loss_operator`. Если
спектральный радиус пробивает стену доверия, срабатывает **симплектическое вето**.

```python
from src.core.risk_supervisor import apply_spectral_cvar_veto

veto = apply_spectral_cvar_veto(
    loss_operator=lambda x: 50.0 * x**2,   # плотность потерь / потенциал
    confidence_level=0.99,
    dirichlet_boundary_conditions=[0.0, 0.0],
)
if veto:
    raise RuntimeError("Spectral CVaR wall breached — halt execution")
```

Изучите числа:

```python
from src.core.risk_supervisor import spectral_cvar_diagnostics
radius, boundary, veto = spectral_cvar_diagnostics(op, 0.99, [0.0, 0.0])
```

Лапласиан использует масштабно-стабильный граф-шаблон (собственные значения в
$[0,4]$), так что **потенциал потерь** — а не шаг сетки — управляет стеной риска.

---

## Сквозной набросок

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

## 5. Точная риманова кривизна через autodiff (JAX обязателен)

`src/models/riemannian_geometry.py` вычисляет для любого метрического поля
`metric_fn(x) -> g (SPD)`:

| Функция | Возвращает |
|---------|------------|
| `christoffel_symbols(metric_fn, x)` | $\Gamma^k_{ij}$, индекс `[k, i, j]` |
| `riemann_tensor(metric_fn, x)` | $R^l{}_{ijk}$, индекс `[l, i, j, k]` |
| `ricci_tensor(metric_fn, x)` | $R_{jk}$ |
| `scalar_curvature(metric_fn, x)` | $R = g^{jk} R_{jk}$ |

Производные метрики $\partial_k g_{ij}$ используют `jax.jacfwd` — аналитические до
машинной точности. Это устраняет ошибку усечения конечных разностей, которая
портит кривизну в высоких размерностях.

```python
import jax.numpy as jnp
from src.models.riemannian_geometry import scalar_curvature

def sphere_metric(x, r=2.0):
    theta = x[0]
    return jnp.array([[r**2, 0.0], [0.0, (r**2) * jnp.sin(theta) ** 2]])

R = scalar_curvature(sphere_metric, jnp.array([0.9, 0.2]))  # -> 0.5 == 2/r^2
```

### Autodiff поток Риччи метрического поля

`compute_ricci_flow_covariance` **является** этим потоком — главный API
шумоподавления это прямой мост к `ricci_flow_metric_field`, без прокси NumPy на
пути. Он интегрирует **нормированный по объёму** поток
$\partial_t g = -2(\mathrm{Ric} - \tfrac{\bar r}{m} g)$ проекцией Галёркина на
искривлённое семейство метрик, засеянное спектром ковариации
(`_anisotropic_conformal_family`; также доступно 2-мерное конформное
`gaussian_conformal_metric_family`). Тензор Риччи в каждой точке сэмпла
`jax.jacfwd`-точен, поток уменьшает дисперсию скалярной кривизны (унификация), и
этот измеренный коэффициент управляет спектральным стягиванием (шумоподавление) —
след сохранён, SPD гарантировано.

## 6. Непрерывная граница просадки Лапласа-Бельтрами

```python
import jax.numpy as jnp
from src.core.risk_supervisor import (
    principal_eigenvalue_laplace_beltrami,
    feynman_kac_drawdown_bound,
    apply_continuous_spectral_cvar_veto,
)

g = lambda x: jnp.array(1.0)   # метрика на координате просадки
b = lambda x: jnp.array(0.0)   # снос

lam0 = principal_eigenvalue_laplace_beltrami(g, b, domain=(0.0, 1.0))  # 0.5*pi^2
prob_bound = feynman_kac_drawdown_bound(lam0, horizon=10.0)            # C e^{-lam0 T}

veto = apply_continuous_spectral_cvar_veto(g, b, (0.0, 1.0), confidence_level=0.99, horizon=5.0)
```

Задача на собственные значения решается на гладком синусном базисе, который точно
удовлетворяет условиям Дирихле (спектральная, экспоненциальная сходимость) — а не
на матрице конечных разностей. Граница — аналитический супремум на вероятность
выхода для указанного генератора.

## 7. Тест дивергенции чисел Бетти

```python
from src.core.topological_allocation import betti_number_divergence_test

report = betti_number_divergence_test(returns, window=60, n_market_factors=1)
print(report.baseline_collapsed)            # сырое многообразие -> b0=1 в крахе
print(report.trp_maintained_separation)     # детонированное многообразие держит b0>1
```

Сравнивает скользящее $b_0$ сырого многообразия корреляций (что видят
выпуклые/HRP методы) с многообразием с удалённой рыночной модой (детонированным).
Во время системного краха сырое многообразие схлопывается в один комок
($b_0\to1$), тогда как детонированное многообразие сохраняет топологическое
разделение ($b_0>1$) — конкретная демонстрация того, почему методы, слепые к
корреляции, упускают нелинейную структуру заражения. Использует фиксированный
радиус, откалиброванный по спокойному референсному окну (правильная настройка для
обнаружения краха).

---

См. также: [07-wave-pinn-alpha-engine.md](07-wave-pinn-alpha-engine.md),
[glossary.md](glossary.md).
