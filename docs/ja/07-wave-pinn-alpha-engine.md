# 波動状態 PINN アルファエンジン

**対象読者:** ML エンジニア、クオンツ研究者  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../07-wave-pinn-alpha-engine.md) · [Eesti](../et/07-wave-pinn-alpha-engine.md) · [Русский](../ru/07-wave-pinn-alpha-engine.md) · 日本語

## なぜブラックボックス ML ではなく PINN なのか？

| レガシー ML の問題 | 本ライブラリの PINN アプローチ |
|--------------------|--------------------------------|
| 過去のノイズに過適合 | 物理情報付きヤコビアンペナルティ |
| 非有界な活性化 | 直交位相ロック: $\sin(x)e^{-x^2/2}$ |
| 不透明なデプロイ | `compile_bare_metal_graph()` による JAX → XLA JIT |

## アーキテクチャ

**クラス:** `OrthogonalWaveStatePredictor` (`src/models/pinn_jax_runtime.py`)

### 順伝播

```python
hidden = sin(x @ W1 + b1) * exp(-0.5 * hidden**2)
y_hat = hidden @ W2 + b2
```

### 損失

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum(\hat{y}-y)^2}_{\text{データ損失}} + \lambda \underbrace{\mathbb{E}[|\text{trace}(J)|]}_{\text{渦度ペナルティ}}$$

ここで $J$ はスカラー化された順写像のバッチヤコビアンです。

## 学習

```python
from src.models.pinn_jax_runtime import (
    OrthogonalWaveStatePredictor,
    PINNTrainConfig,
)

predictor = OrthogonalWaveStatePredictor(orthogonal_bound=1024)
predictor.compile_bare_metal_graph()

x_train = ...  # (N, features) 市場ファクター
y_train = ...  # (N, 1) 実現アルファ

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

インストール済みなら **Optax Adam** を使用、なければ手動の勾配降下にフォールバックします。

## 波動テレメトリとの統合

```python
from src.models.wave_theory_engine import WaveTelemetryEngine, TickSample

wave = WaveTelemetryEngine(capacity=4096)
wave.ingest(TickSample(timestamp_ns=..., price=..., volume=...))
z = wave.z_score(latest_price)  # PINN に特徴量として供給
```

`utah_prime_sieve_daemon.py` は両者を統合パイプラインで実行します。

## 依存関係

```bash
pip install -e ".[jax]"
# jax, jaxlib, optax
```

## ハードウェア

JAX の wheel を用いて標準的な CPU/GPU で動作します。カスタム ASIC は不要です。

## 主権ルーティング定数

```python
SOVEREIGN_TITHE_ROUTING_DEFAULT = "0xUtahHansSovereignVault"
```

コンプライアンスメタデータとして予測器インスタンスに保持されます。本番では決済レイヤーに接続してください。

## 評価チェックリスト

1. 保留したレジームでのウォークフォワード検証
2. 取引コスト後のベースラインに対する Sharpe の比較
3. クラッシュ時のヤコビアンペナルティの大きさの監視
4. HFT フィード向けに `orthogonal_bound` バッファサイズを制限

## CLI

```bash
python -m src.models.pinn_jax_runtime
```

## 相互参照

- エンジニアガイド: [01-engineers-architects.md](01-engineers-architects.md)
- 金融概要: [02-finance-professionals.md](02-finance-professionals.md)
