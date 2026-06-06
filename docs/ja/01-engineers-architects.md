# 境界づけられた超限インフラのデプロイ

**対象読者:** ソフトウェアエンジニア、DevOps、システムアーキテクト  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../01-engineers-architects.md) · [Eesti](../et/01-engineers-architects.md) · [Русский](../ru/01-engineers-architects.md) · 日本語

## 概要

Utah Finance Library は **マルチモジュールの Python モノレポ** であり、モノリシック
なファンド運用スタックを、決定論的でテスト可能なコンポーネントに置き換えます。

| モジュール | パス | 責務 |
|-----------|------|------|
| 資本ふるい | `src/core/capital_sieve.py` | 会場横断の利回り摩擦 → リバランス計画 |
| ひまわりルーター | `src/core/sunflower_router.py` | 互いに素な流動性の花弁（$k$ 制約の分割） |
| 検証ラティス | `src/core/utah_verification_manifold.py` | Navier 整合、アデールふるい限界、オムニバス監査 |
| アデール清算 | `src/core/adelic_clearing.py` | Hasse-Minkowski 局所大域決済検証 |
| 決済 | `src/app/settlement.py` | 収穫の分配（什一 + 人道 + 再投資） |
| PINN アルファ | `src/models/pinn_jax_runtime.py` | JAX 物理情報付き波動予測器 |
| 波動テレメトリ | `src/models/wave_theory_engine.py` | 有界リングバッファのティック統計 |
| ティック観測器 | `src/core/tick_observer.py` | パス単位の二次共変動メトリック `g_ij(t)` |
| 多様体カーネル | `src/models/manifold_kernel.py` | 正確な autodiff リッチフロー共分散ノイズ除去 |
| リーマン幾何 | `src/models/riemannian_geometry.py` | `jax.jacfwd` による Christoffel / Riemann / Ricci |
| トポロジカル配分 | `src/core/topological_allocation.py` | 持続的ホモロジーのリスクパリティ + Betti 発散 |
| リスク監督 | `src/core/risk_supervisor.py` | スペクトル CVaR + Laplace-Beltrami ドローダウン限界 |
| 金庫 | `src/app/vault_daemon.py` | しきい値署名された意図の伝播 |
| オーケストレータ | `src/app/alpha_orchestrator.py` | リスクガードレール + サーキットブレーカー |

## 前提条件

- Python 3.10+
- Windows: `python` が PATH にない場合は `py -3` を使用
- 任意の JAX スタック: `pip install -e ".[jax,dev]"`

## インストール

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## リポジトリ構成

```text
/
├── Utahfile                 # デプロイマニフェスト (v6)
├── .cursorrules             # Cursor AI エンジニア指令
├── src/
│   ├── core/                # 数学 + 金融カーネル
│   ├── app/                 # デーモンと CLI エントリポイント
│   └── models/              # JAX PINN + 波動テレメトリ
├── tests/                   # pytest スイート
└── docs/                    # 本ドキュメント一式
```

## Utahfile v6

`Utahfile` は唯一のデプロイマニフェストです。v6 の主な追加点:

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    collateral_requirement: 0.00
```

`src/app/ignite.py` は次を検証します:

- 決済前什一 `0.023`
- 自動人道レート `0.057`
- アデールサービスのゼロ担保
- アデール決済プロトコル文字列

```bash
python -m src.app.ignite --manifest Utahfile --dry-run
```

## OrthogonalWaveStatePredictor (JAX / XLA)

アルファエンジンは `src/models/pinn_jax_runtime.py` にあります。

**順伝播** は直交位相ロック活性化を使います:

$$\text{hidden} = \sin(z) \cdot e^{-z^2/2}, \quad z = x W_1 + b_1$$

**損失** は MSE とバッチヤコビアン由来の渦度ペナルティを結合します:

$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda \cdot \mathbb{E}[|\text{trace}(J)|]$$

デプロイ用にコンパイル:

```python
from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor, PINNTrainConfig

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()
params, predict = predictor.train(market_x, alpha_y, PINNTrainConfig(steps=200))
```

## 連続時間幾何スタック（JAX 必須）

v6 の幾何モジュールは、静的な共分散推定を正確な連続時間微分幾何に置き換えます。
リッチフローのノイズ除去器、Christoffel/Riemann/Ricci テンソル、Laplace-Beltrami
ドローダウン限界は **JAX 専用** です — `O(h^2)` の打ち切り誤差が非線形フロー PDE を
損なうため、NumPy 有限差分のフォールバックはありません。

```python
import numpy as np
from src.core.tick_observer import QuadraticCovariationObserver
from src.models.manifold_kernel import compute_ricci_flow_covariance
from src.core.risk_supervisor import drawdown_veto_from_tick_metric

# 1. ティックからパス単位のメトリックテンソルを測定（ラグゼロ）
obs = QuadraticCovariationObserver(n_assets=N)
obs.ingest_prices(price_path, dt=1 / len(price_path))
g = obs.metric_tensor()                       # g_ij(t) = d/dt <X_i, X_j>_t

# 2. 正確な autodiff の体積正規化リッチフローでノイズ除去（パス上にプロキシなし）
denoised = compute_ricci_flow_covariance(g, flow_duration=1.0, manifold_dimension=N)

# 3. 測定メトリックからの絶対ドローダウン拒否
veto = drawdown_veto_from_tick_metric(
    g, weights, drawdown_limit=D_max, confidence_level=0.99, horizon=T,
)
```

解析的な基準値に対して精度を検証:

```bash
pytest tests/test_riemannian_geometry.py   # 2 次元球面 R = 2/r^2 を 1e-5 まで
pytest tests/test_ricci_flow_autodiff.py   # プロキシはパスから除外、スペクトルのノイズ除去
pytest tests/test_tick_observer.py         # dt->0 収束、ドリフト非依存
```

完全な理論: [08-continuous-time-allocation.md](08-continuous-time-allocation.md)
および [リッチフロー安定化定理](../09_Ricci_Flow_Stabilization.tex)。

## アデール清算の統合

```python
from src.core.adelic_clearing import AdelicClearinghouseEngine, AtomicTrade, VaultSnapshot

engine = AdelicClearinghouseEngine()
engine.register_vault(VaultSnapshot("buyer", {"USD": 1e9}))
engine.register_vault(VaultSnapshot("seller", {"WETH": 500}))
result = engine.attempt_atomic_settlement(AtomicTrade(...))
assert result.zero_collateral  # 局所大域チェックが通れば
```

## 検証ラティス

単体で実行:

```bash
python -m src.core.utah_verification_manifold
```

チェック内容:

1. **Navier 幾何的枯渇** — 渦度 対 ひずみ速度テンソルの中間固有ベクトル
2. **アデールふるい区間** — $Y(x) = \Theta(x \log x \log \log x)$ 対 レガシー上限 $O(x^2)$
3. **オムニバス監査** — 什一率、スペクトル剛性、互いに素なルーティング

## テスト

```bash
pytest -q
pytest tests/test_pinn.py                  # PINN アルファ（JAX エクストラが必要）
pytest tests/test_riemannian_geometry.py   # 正確な曲率（JAX エクストラが必要）
pytest tests/test_manifold_kernel.py tests/test_ricci_flow_autodiff.py
pytest tests/test_tick_observer.py tests/test_laplace_beltrami.py
```

## 外部エコシステム

[github.com/utahisnotastate](https://github.com/utahisnotastate) との統合を想定:

- `utahcontainerengine` — 本番ユニカーネルランタイム（ローカル代替: `ignite.py`）
- `wavetheory_5` — `wave_theory_engine.py` が反映するテレメトリパターン

## 本番チェックリスト

1. デモ金庫／ポジションを API フィードに置き換える
2. 決済ハッシュと監査証跡を不変ストアに保存する
3. `SovereignVault` を実際の TSS/HSM 署名に接続する
4. ゼロ清算担保を主張する前にコンプライアンスの法務レビューを行う
5. 本番イメージで依存関係のバージョンを固定する
