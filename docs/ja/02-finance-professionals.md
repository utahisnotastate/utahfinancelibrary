# 運用上の摩擦と専有的な搾取の終焉

**対象読者:** クオンツアナリスト、ポートフォリオマネージャー、執行トレーダー、リスク責任者  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../02-finance-professionals.md) · [Eesti](../et/02-finance-professionals.md) · [Русский](../ru/02-finance-professionals.md) · 日本語

## 経営層向け要約

あなたのファンドが今も Bloomberg AIM、Enfusion、Aladdin に七桁の年間ライセンスを
支払い、**かつ** 数億ドルを清算担保に固定しているなら、二つの戦線でアルファを失血
しています。**ソフトウェアの搾取** と **担保の摩擦** です。

Utah Finance Library はオープンソースの代替で、次を自動化します。

1. **資本摩擦の検出** — プライムブローカーと会場の間で失われた利回りを発見
2. **グローバルネッティング** — 重複を減らすため流動性を互いに素なバッチでルーティング
3. **プログラム的決済** — 各収穫を執行時に分配
4. **アデール的バイパス** — モデル化された CCP 担保なしでアトミック決済を検証（支払能力証明が通れば）
5. **物理情報付きアルファ** — 不透明なブラックボックス ML の代わりに JAX PINN レイヤー
6. **連続時間幾何** — 古びた共分散推定窓の代わりに、ティックから市場のメトリックテンソルを測定しドローダウンをスペクトル的に制限

## 問題 1: 会場横断の資本流出

**レガシー手順:** スプレッドシート + 日次のプライムブローカー明細。摩擦の発見は数日後。

**Utah 手順:** `AutonomousAuditor` がティッカー別にポジションをまとめ、リスク調整後リターンを比較し、摩擦がしきい値を超えると `AllocationIntent` を発行します。

サンプルログ:

```text
[AUDIT ALERT] Capital leakage in USD at prime_custody_01. Drag: 90000.0000
```

**アクション:** 低利回りの会場から最適な会場へ、遊休 USD を自動的に再ルーティング。

## 問題 2: 清算機関に固定された担保

**レガシー手順:** 取引が取引所で約定 → 清算会員が DTCC に担保を差し入れ → T+1 決済 → 資本が一晩遊休。

**Utah 手順:** `HasseMinkowskiVerifier` が次を検証します。

- **実体:** 買い手は建て通貨を、売り手は原資産を保有
- **局所体:** 小さな素数を法とする支払能力の合同（$p$ 進的な互換性の代替）
- すべて通れば → `required_collateral = 0` → アトミックな台帳交換

デモ取引は、検証が通れば **250 万ドル × 1000 WETH 想定元本** のモデル化レガシー担保を解放します。

> **リスク開示:** 実際の清算には、法的ネッティング権、CCP 規則、規制資本が関わります。本ライブラリは **検証と台帳のモデル** を実装しており、本番前に承認済みインフラへ対応付ける必要があります。

## 問題 3: 確率的 ML アルファの劣化

**レガシー手順:** 過去バー上の Transformer/LSTM → ノイズに過適合 → レジーム転換で破綻。

**Utah 手順:** `OrthogonalWaveStatePredictor` は次を強制します。

- 有界な活性化多様体（sin × ガウス包絡）
- 学習時の Navier-Stokes 由来ヤコビアンペナルティ
- `compile_bare_metal_graph()` による XLA コンパイル

## 問題 4: 手作業の照合

**レガシー手順:** 運用チームが月末に EMS、PB、ファンド管理者を照合。

**Utah 手順:** 各決済が決定論的な `SettlementInstruction` 行と、アデール取引上の SHA-256 決済ハッシュを生成。

## 手数料パイプラインの透明性

各収穫は `AutonomousSettlementEngine` を通ります。

| 受領者 | レート | 目的 |
|--------|--------|------|
| 議定書什一 (Utah Hans) | **2.3%**（不変） | ライブラリの維持 |
| 人道的豊穣マトリクス | **5.7%**（既定、設定可能） | インパクト配分 |
| 内部金庫 | 残り | 再投資 |

## 比較表

| 機能 | レガシースタック | Utah Finance Library |
|------|------------------|----------------------|
| 会場横断の摩擦監査 | 手動 / 遅延 | 連続（`capital_sieve`） |
| ネッティング | 日次バッチ | ひまわり花弁（互いに素なルーティング） |
| 決済分配 | 月末会計 | 収穫ごとのプログラム的処理 |
| 清算担保 | CCP 規則 + ヘアカット | アデールモデル → 検証されれば 0 |
| アルファモデル | 不透明なベンダー ML | オープン PINN + 波動テレメトリ |
| 監査証跡 | DB の変更 | ハッシュ固定されたイベント |

## はじめ方（トレーディングデスク）

```bash
python -m src.app.utah_prime_sieve_daemon --json > daily_audit.json
python -m src.app.hasse_minkowski_daemon
```

JSON 出力の `migration_intents`、`adelic_clearing`、`alpha_signal` を確認してください。

## 問題 5: 古びた窓で推定された静的共分散

**レガシー手順:** 振り返り窓で共分散行列を推定し、それを縮小（Ledoit-Wolf/OAS）して
凸ソルバーに供給します。推定は市場から遅れ、窓長の判断と推定誤差を含みます。

**Utah 手順:** [連続時間トポロジカル配分](08-continuous-time-allocation.md) は市場の
幾何を **測定** し、モデル化しません。

- `QuadraticCovariationObserver` は、パス単位のメトリックテンソル
  `g_ij(t) = d/dt ⟨X_i, X_j⟩_t` をティックストリームから直接読み取ります —
  **振り返り窓なし、ラグなし** — 生のティックには二スケール（TSRV）推定を併用します。
- `compute_ricci_flow_covariance` は、このメトリックを **正確な autodiff** の
  体積正規化リッチフロー（`jax.jacfwd`、有限差分プロキシなし）でノイズ除去します。
- `feynman_kac_drawdown_bound` は、Laplace-Beltrami の主固有値をドローダウンの
  解析的上限 `P(sup DD > D_max) ≤ C e^{-λ₀T}` に変換します。
- `betti_number_divergence_test` は伝染トポロジーを検出します。合成クラッシュ時、
  生の相関多様体は崩壊（`b₀ → 1`）する一方、デトーンされたトポロジカルリスクパリ
  ティ多様体はクラスタ分離を維持します。

## `riskfolio-lib` からの移行

連続時間スイートは、一般的な `riskfolio-lib` の手順に対する幾何的対応物を提供します。

| riskfolio-lib | Utah の対応物 |
|---------------|---------------|
| `HCPortfolio` (HRP/NCO) | `optimize_topological_risk_parity` |
| Ledoit-Wolf / OAS 縮小 | `compute_ricci_flow_covariance`（正確な autodiff） |
| 標本/EWMA 共分散 | `QuadraticCovariationObserver`（パス単位、ラグゼロ） |
| L1 回転率制約 | `calculate_navier_stokes_rebalance_flow` |
| Mean-CVaR / EVaR / Max-DD | `apply_spectral_cvar_veto`, `feynman_kac_drawdown_bound` |

幾何は正確（2 次元球面で `1e-5` まで検証済み）であり、パス単位のメトリックは推定
パラメータではなく $\mathcal{F}_t$ 可測な観測量です。実際の取引所ティックがマイクロ
ストラクチャノイズを含む場合、二スケール推定が主要なバイアスを除去します。どの新
手法でも同様に、資本を再配分する前に既存の凸ベースラインに対して検証してください。

## Bloomberg / Enfusion からの移行ロードマップ

1. **第 1 週:** 会場別にポジションをエクスポート → `AssetPosition` リストへ供給
2. **第 2 週:** 意図のシャドウモード（執行なし）
3. **第 3 週:** アデール検証のために金庫スナップショットを接続
4. **第 4 週:** 内部台帳でアトミック決済をパイロット
5. **継続的に:** EMS/PB フィードの段階的統合
