# 連続時間トポロジカル配分

**対象読者:** クオンツ研究者、`riskfolio-lib` から移行するポートフォリオエンジニア  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../08-continuous-time-allocation.md) · [Eesti](../et/08-continuous-time-allocation.md) · [Русский](../ru/08-continuous-time-allocation.md) · 日本語

このモジュール群は、離散共分散行列上の静的二次計画法から **連続時間幾何** へと、
ポートフォリオ最適化を再構成します。持続的ホモロジー、リッチフロー、流体ルーティ
ング、スペクトルリスク限界です。

ライブラリは市場を *モデル化* しません。市場が展開するにつれてその位相的曲率を
**測定** し、その幾何の絶対的なスペクトル境界を計算します。メトリックテンソルは
推定パラメータではなく、ティックストリームから直接読み取られる $\mathcal{F}_t$
可測な観測量です（下の定理および `src/core/tick_observer.py` を参照）。

### 定理 — ポートフォリオ・メトリックテンソルのパス単位の厳密性

市場状態ベクトル $X_t$ を、経験的フィルトレーション $\mathcal{F}_t$ に適合した連続
セミマルチンゲールとします。ポートフォリオ多様体の Laplace-Beltrami 生成作用素
$\mathcal{L}$ を定義するリーマン・メトリックテンソル $g_{ij}(t)$ は、連続時間の
二次共変動によって厳密に定まります。

$$g_{ij}(t) = \frac{d}{dt}\,\langle X_i, X_j\rangle_t.$$

高頻度の観測間隔 $dt \to 0$ のとき、$g_{ij}(t)$ の経験的測定は真のメトリックに収束
します（確率収束、かつ細分化分割に沿ってほとんど確実に）。

**証明.** 連続セミマルチンゲールの二次共変動過程の定義により、二乗増分の和の極限は
**ドリフトベクトル** $b$ に依存せず、厳密な共変動行列にパス単位で収束します。したが
って $g_{ij}$ は「モデル化の判断」に従うパラメトリックな仮定ではなく、$\mathcal{F}_t$
可測なティックストリームから抽出された幾何的不変量です。結果として、$\mathcal{L}$
の主固有値 $\lambda_0$ は、Feynman-Kac を通じて、メトリック推定誤差のないドローダウ
ン領域の物理的上限を与えます。$\blacksquare$

この対象のドリフト非依存性は `tests/test_tick_observer.py::test_drift_independence`
で、$dt \to 0$ 収束は `test_realized_covariation_converges_as_dt_shrinks` で直接検証
されます。

**実践的な実現（ティックのマイクロストラクチャ）。** 実際の取引所フィードでは、観測
価格は潜在セミマルチンゲールに i.i.d. のマイクロストラクチャノイズが加わったもの
であり、$dt \to 0$ につれて素朴な実現共変動を上方にバイアスします。これはメトリッ
クの認識論的ギャップではなく、センサーの *工学的* アーティファクトです。ライブラリ
は一貫した二スケール実現共変動推定（`two_scale_realized_covariance`）を適用し、主要
なノイズ項を打ち消して同じ積分共変動に収束します。メトリックは観測量のままであり、
私たちはそれをノイズに頑健な計器で読み取るだけです。

## ティック観測器 — 生成作用素をパス単位メトリックに結びつける

`src/core/tick_observer.py` は $g_{ij}(t)$ をティックストリームから直接測定し、生成
作用素と PINN ランタイムに供給します。**振り返り窓の平均なし**（ラグと窓長の選択を
招く）であり、メトリックは瞬時のパス単位極限です。

```python
from src.core.tick_observer import QuadraticCovariationObserver
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

obs = QuadraticCovariationObserver(n_assets=N)        # または確率ボラ用に decay<1
for log_price_vec, dt in tick_stream:
    obs.ingest(log_price_vec, dt=dt)                  # オンライン、ティックあたり O(N^2)

g = obs.metric_tensor()                                # 厳密な測定メトリック

# 測定メトリックからの絶対ドローダウン拒否（推定窓なし）
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

同じメトリックが PINN 入力を内在座標に白色化します。

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

predictor = OrthogonalWaveStatePredictor()
predictor.bind_tick_metric(obs)        # g(t) = <X_i, X_j>_t / t
x_intrinsic = predictor.whiten(raw_factors)   # g^{-1/2} x
```

- `realized_covariation` — 価格パスからのバッチ二次共変動。
- `two_scale_realized_covariance` — 生の取引所ティック向けのノイズ頑健 TSRV。
- `QuadraticCovariationObserver` — オンライン、ラグなしのパス単位観測器。
- `drawdown_metric_from_covariation` — ポートフォリオ射影 $w^\top\Sigma w$。

## v6.1 アップグレード — 連続幾何、JAX 必須

3 つのアップグレードが、本スイートを離散代替物から厳密な連続幾何へ移行させます
（詳細は下の 5〜7 節）。

1. **autodiff による厳密な曲率** (`src/models/riemannian_geometry.py`) —
   Christoffel 記号、Riemann/Ricci テンソル、スカラー曲率を JAX autodiff で機械精度
   まで計算。**有限差分なし、`O(h^2)` なし。** 丸い 2 次元球面（$R = 2/r^2$、
   $\mathrm{Ric} = g/r^2$）に対して `1e-5` まで検証済み。
2. **連続 Laplace-Beltrami ドローダウン限界** (`src/core/risk_supervisor.py`) —
   ポートフォリオ生成作用素 $\mathcal{L} = \tfrac12\Delta_M + b\cdot\nabla$ の主
   Dirichlet 固有値 $\lambda_0$ を、滑らかなスペクトル Galerkin 基底（指数収束）で
   求め、Feynman-Kac 限界
   $\mathbb{P}(\sup \text{DD} > \mathcal{D}_{max}) \le C e^{-\lambda_0 T}$ に供給。
   平坦区間で $\lambda_0 = \tfrac12(\pi/L)^2$ に対して検証済み。
3. **厳格 JAX モード** — `require_jax=True` と autodiff 幾何は、低精度の NumPy テンソ
   ル計算へ静かに切り替える代わりに、JAX なしでの実行を拒否します。

## 概要

| API | ファイル | 置き換え対象 (riskfolio) |
|-----|----------|--------------------------|
| `optimize_topological_risk_parity` | `src/core/topological_allocation.py` | HRP / NCO |
| `compute_ricci_flow_covariance` | `src/models/manifold_kernel.py` | Ledoit-Wolf / OAS 縮小、ノイズ除去 |
| `calculate_navier_stokes_rebalance_flow` | `src/core/sunflower_router.py` | L1 回転率 / 取引コスト制約 |
| `apply_spectral_cvar_veto` | `src/core/risk_supervisor.py` | Mean-CVaR / EVaR / Max-Drawdown |

---

## 1. トポロジカルリスクパリティ (TRP)

Mantegna 相関距離 $d_{ij}=\sqrt{2(1-\rho_{ij})}$ と、その上の Vietoris-Rips フィルト
レーションを構築します。

- **H0** 持続性は、単連結 union-find で *厳密に* 計算されます。
- **H1**（サイクル / 伝染ループ）は 1-スケルトンのサイクルランク $E - V + b_0$ を使用。
- **H2** は充填三角形の空洞代替を使用。

資本は各資産の位相的負荷（橋渡しのマージ + サイクル次数）に **反比例** して配分され、
その後 Wasserstein バーコード項によって $1/N$ へとブレンドされます。

```python
from src.core.topological_allocation import optimize_topological_risk_parity

weights = optimize_topological_risk_parity(
    tensor_data=returns,            # (T, N) array-like (NumPy または jax.Array)
    max_homology_dimension=2,
    wasserstein_penalty=0.01,
)
```

豊富な診断:

```python
from src.core.topological_allocation import topological_risk_parity_report
report = topological_risk_parity_report(returns)
print(report.betti_numbers, report.barcode_wasserstein, report.filtration_radius)
```

---

## 2. リッチフロー共分散

体積正規化リッチフロー

$$\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \tfrac{2}{m} r\, g_{ij}$$

を **厳密な autodiff メトリック場フローへの直接の橋渡し** として積分します — パス上に
NumPy 有限差分の代替はありません。定数共分散行列は *平坦* なメトリック（Ricci ゼロ）
なので、API はまず共分散スペクトルによって曲率が誘導される曲がったリーマン・メトリ
ック *場* を構築し、その後 `jax.jacfwd` で機械精度まで得た Christoffel 記号と Ricci
テンソルでフローを積分します。スペクトルは **測定された曲率一様化比** によってバルク
平均へ縮約されます。

$$\gamma = \frac{\operatorname{std}_x R(T)}{\operatorname{std}_x R(0)} \in [0,1],
\qquad \log\lambda_i^{\text{den}} = \overline{\log\lambda} + \gamma\,(\log\lambda_i - \overline{\log\lambda}),$$

したがって縮約はフローの厳密な幾何（スカラー曲率が定数になるにつれ $\gamma \to 0$）
によって支配され、離散代替物によるものではありません。トレース（総分散）は保存され
ます。`ricci_curvature_proxy` は独立したスペクトル診断として残りますが、ノイズ除去器
からは **決して** 呼び出されません
（`tests/test_ricci_flow_autodiff.py::test_api_does_not_call_numpy_proxy` で強制）。

> 曲率一様化定理と、実装された縮約比 $\gamma$ への正確な対応は、
> [`docs/09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex)
> で形式化されています。

```python
from src.models.manifold_kernel import compute_ricci_flow_covariance

denoised = compute_ricci_flow_covariance(
    empirical_metric_tensor=cov,    # (N, N) SPD
    flow_duration=1.0,
    manifold_dimension=cov.shape[0],
)
```

`wave_theory_engine` 向けの、厳密な（autodiff 測定の）スカラー曲率分散の軌跡:

```python
from src.models.manifold_kernel import ricci_flow_curvature_field
times, curv_dispersion = ricci_flow_curvature_field(cov, 1.0, cov.shape[0], samples=8)
```

経験的に、フローは SPD 構造とトレースを保ちつつ **固有値の広がりを縮約** （ノイズ
除去）します — `tests/test_manifold_kernel.py` および
`tests/test_ricci_flow_autodiff.py` を参照。

---

## 3. Navier-Stokes 流動性ルーティング

リバランスを非圧縮 Stokes 流としてモデル化します。粘性 = マーケットインパクト、体積
力 = アルファ勾配。鞍点 (KKT) 系を解き、速度場が **質量保存** ($\mathbf{1}^\top v = 0$)
となるようにします。

```python
from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow

velocity, pressure_gradient = calculate_navier_stokes_rebalance_flow(
    current_weights=w_now,
    target_manifold=w_target,
    market_viscosity_tensor=0.1,        # スカラー、(N,)、または (N, N)
    kinematic_constraints={"max_velocity": 0.25, "coupling": 1.0},
)
```

連続的な執行 **速度場**（資本がどれだけ速く／どこへ流れるか）と、裁定をルーティング
する **圧力勾配** を返します — 静的な目標ウェイトだけではありません。

---

## 4. 固有多様体（スペクトル）CVaR 拒否

Schrödinger 型作用素 $-\Delta + V(x)$ を Dirichlet 境界条件で離散化します。ここで
$V$ はあなたの `loss_operator` からサンプリングされます。スペクトル半径が信頼の壁を
破ると、**シンプレクティック拒否** が発火します。

```python
from src.core.risk_supervisor import apply_spectral_cvar_veto

veto = apply_spectral_cvar_veto(
    loss_operator=lambda x: 50.0 * x**2,   # 損失密度 / ポテンシャル
    confidence_level=0.99,
    dirichlet_boundary_conditions=[0.0, 0.0],
)
if veto:
    raise RuntimeError("Spectral CVaR wall breached — halt execution")
```

数値を調べる:

```python
from src.core.risk_supervisor import spectral_cvar_diagnostics
radius, boundary, veto = spectral_cvar_diagnostics(op, 0.99, [0.0, 0.0])
```

Laplacian はスケール安定なグラフステンシル（固有値は $[0,4]$）を使うため、格子間隔
ではなく **損失ポテンシャル** がリスクの壁を支配します。

---

## エンドツーエンドのスケッチ

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

## 5. autodiff による厳密なリーマン曲率（JAX 必須）

`src/models/riemannian_geometry.py` は、任意のメトリック場 `metric_fn(x) -> g (SPD)`
に対して次を計算します。

| 関数 | 戻り値 |
|------|--------|
| `christoffel_symbols(metric_fn, x)` | $\Gamma^k_{ij}$、インデックス `[k, i, j]` |
| `riemann_tensor(metric_fn, x)` | $R^l{}_{ijk}$、インデックス `[l, i, j, k]` |
| `ricci_tensor(metric_fn, x)` | $R_{jk}$ |
| `scalar_curvature(metric_fn, x)` | $R = g^{jk} R_{jk}$ |

メトリック微分 $\partial_k g_{ij}$ は `jax.jacfwd` を使用 — 機械精度まで解析的です。
これにより、高次元で曲率を損なう有限差分の打ち切り誤差を取り除きます。

```python
import jax.numpy as jnp
from src.models.riemannian_geometry import scalar_curvature

def sphere_metric(x, r=2.0):
    theta = x[0]
    return jnp.array([[r**2, 0.0], [0.0, (r**2) * jnp.sin(theta) ** 2]])

R = scalar_curvature(sphere_metric, jnp.array([0.9, 0.2]))  # -> 0.5 == 2/r^2
```

### autodiff メトリック場リッチフロー

`compute_ricci_flow_covariance` は **この** フローです — 主要なノイズ除去 API は
`ricci_flow_metric_field` への直接の橋渡しで、パス上に NumPy プロキシはありません。
共分散スペクトルで種づけされた曲がったメトリック族（`_anisotropic_conformal_family`、
2 次元共形 `gaussian_conformal_metric_family` も利用可能）への Galerkin 射影によって、
**体積正規化** フロー $\partial_t g = -2(\mathrm{Ric} - \tfrac{\bar r}{m} g)$ を積分
します。各サンプル点での Ricci テンソルは `jax.jacfwd` で厳密、フローはスカラー曲率
分散を低減（一様化）し、その測定比がスペクトル縮約（ノイズ除去）を駆動します —
トレースは保存、SPD は保証されます。

## 6. 連続 Laplace-Beltrami ドローダウン限界

```python
import jax.numpy as jnp
from src.core.risk_supervisor import (
    principal_eigenvalue_laplace_beltrami,
    feynman_kac_drawdown_bound,
    apply_continuous_spectral_cvar_veto,
)

g = lambda x: jnp.array(1.0)   # ドローダウン座標上のメトリック
b = lambda x: jnp.array(0.0)   # ドリフト

lam0 = principal_eigenvalue_laplace_beltrami(g, b, domain=(0.0, 1.0))  # 0.5*pi^2
prob_bound = feynman_kac_drawdown_bound(lam0, horizon=10.0)            # C e^{-lam0 T}

veto = apply_continuous_spectral_cvar_veto(g, b, (0.0, 1.0), confidence_level=0.99, horizon=5.0)
```

固有値問題は、Dirichlet 条件を厳密に満たす滑らかな正弦基底上で解かれます（スペクト
ル的、指数収束）— 有限差分行列ではありません。この限界は、指定された生成作用素に
対する退出確率の解析的上限です。

## 7. Betti 数発散テスト

```python
from src.core.topological_allocation import betti_number_divergence_test

report = betti_number_divergence_test(returns, window=60, n_market_factors=1)
print(report.baseline_collapsed)            # 生の多様体 -> クラッシュで b0=1
print(report.trp_maintained_separation)     # デトーン多様体は b0>1 を維持
```

生の相関多様体の移動 $b_0$（凸/HRP 手法が見るもの）と、市場モードを除去した（デトー
ン）多様体を比較します。システミッククラッシュの間、生の多様体は単一の塊へ崩壊
（$b_0\to1$）する一方、デトーン多様体は位相的分離（$b_0>1$）を保ちます — 相関に盲目
な手法が非線形の伝染構造を見逃す理由の具体的な実証です。穏やかな参照窓から較正した
固定半径を使用します（クラッシュ検出のための正しい設定）。

---

参照: [07-wave-pinn-alpha-engine.md](07-wave-pinn-alpha-engine.md)、
[glossary.md](glossary.md)。
