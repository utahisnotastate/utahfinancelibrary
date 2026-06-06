# Utah Finance Library — ドキュメント目次

**バージョン 6.Omnibus_Adelic**  
**リポジトリ:** [github.com/utahisnotastate/utahfinancelibrary](https://github.com/utahisnotastate/utahfinancelibrary)

> **言語:** [English](../README.md) · [Eesti](../et/README.md) · [Русский](../ru/README.md) · 日本語

Utah Finance Library の公式ドキュメントへようこそ。これらのガイドは、役割や経験
レベルごとに書かれています。あなたに最も合うガイドから読み始めてください。

| 対象読者 | ガイド | 学べること |
|----------|--------|------------|
| ソフトウェアエンジニア・アーキテクト | [境界づけられたインフラのデプロイ](01-engineers-architects.md) | リポジトリ構成、Utahfile、JAX PINN、検証ラティス、デプロイ |
| クオンツ・PM・トレーダー | [運用上の摩擦の終焉](02-finance-professionals.md) | アルファのルーティング、ネッティング、決済、アデール的バイパス対レガシー |
| 創業者・ファミリーオフィス | [絶対的な金融主権](03-founders-family-offices.md) | コスト、セキュリティ、分配ルール、移行の道筋 |
| 子ども・初心者 | [超かしこい壊れない貯金箱](04-children-beginners.md) | お金のルーティングと分け合いルールのやさしい説明 |
| コンプライアンス・オペレーション | [決済と什一の議定書](05-settlement-governance.md) | 2.3% の什一、5.7% の人道的レート、監査フック |
| 上級 / SOTA | [アデール的清算機関バイパス](06-adelic-clearinghouse-bypass.md) | Hasse-Minkowski ゼロ担保モデル |
| 上級 / SOTA | [波動状態 PINN アルファエンジン](07-wave-pinn-alpha-engine.md) | JAX による物理情報付きニューラルネット |
| 上級 / SOTA | [連続時間トポロジカル配分](08-continuous-time-allocation.md) | TRP、リッチフロー、ティック観測器、NS ルーティング、Laplace-Beltrami ドローダウン |
| 全員 | [用語集](glossary.md) | ライブラリ全体で使われる用語 |

> **理論付録（LaTeX）:** 数学定理
> [`09_Ricci_Flow_Stabilization.tex`](../09_Ricci_Flow_Stabilization.tex)、
> [`10_Entanglement_Hedging.tex`](../10_Entanglement_Hedging.tex)、
> [`11_Jarzynski_Harvesting.tex`](../11_Jarzynski_Harvesting.tex)、
> [`12_Braid_Execution.tex`](../12_Braid_Execution.tex)、
> [`13_Koopman_Linearization.tex`](../13_Koopman_Linearization.tex)
> は言語中立（普遍的な数学記法）に保たれているため、各言語向けに複製していません。

## 研究 / SOTA 物理モジュール

対象読者向けガイドに加えて、本ライブラリには数理物理の道具立てを借用した研究グレ
ードのモジュール群があります。いずれも、誇大な主張（無リスク裁定・予言オラクル・
スリッページゼロ）を伴わない、正直な但し書き付きの本物のテスト済み推定器／診断ツ
ールです。

| モジュール | 目的 | 理論ノート |
|-----------|------|-----------|
| `holographic_projection.py` | AdS/CFT 風の有界な板情報（LOB）圧力特徴量 | — |
| `tensor_network_hedge.py` | MPS / フォン・ノイマンエントロピーによる分散診断 | [10_Entanglement_Hedging.tex](../10_Entanglement_Hedging.tex) |
| `chrono_drift.py` | 完成パス上の Malliavin / Skorokhod グリークス（先読みなし） | — |
| `jarzynski_harvester.py` | Jarzynski 自由エネルギー + Crooks 不可逆性の推定 | [11_Jarzynski_Harvesting.tex](../11_Jarzynski_Harvesting.tex) |
| `braid_router.py` | Kauffman/Jones 不変量 + 非可換な執行順序付け | [12_Braid_Execution.tex](../12_Braid_Execution.tex) |
| `koopman_oracle.py` | Koopman / EDMD によるリフト空間での線形予測 | [13_Koopman_Linearization.tex](../13_Koopman_Linearization.tex) |

正の利回り／節約の出力はすべて、透明で設定可能かつ除去可能な
`enforce_universal_tithe` の会計ヘルパーを経由します。

## Utah への支払い

2.3% の議定書什一と人道的分配は、ライブラリ内部の会計ルート
（[`src/core/protocol_economics.py`](../../src/core/protocol_economics.py) を参照）
です。**実際に Utah へ支払いを行う** には — スポンサーシップ、什一の送金、または
支援 — Utah に直接連絡してください。

> **連絡先:** [utah@utahcreates.com](mailto:utah@utahcreates.com)

現時点では、これは手動の人手による手順です（方法・参照番号・金額をメールで調整し
ます）。**専用の GUI アプリを計画中** で、送金詳細の生成、什一／人道的分配の追跡、
受領記録までを一貫して管理し、メールではなく数クリックで完了できるようになります。
それが公開されるまでは、上記のメールが Utah への支払いの正式な手段です。

## クイックコマンド

```bash
# フルパイプライン（監査 → ルーティング → 決済 → アデール → アルファ）
python -m src.app.utah_prime_sieve_daemon

# アデール的ゼロ担保デモ
python -m src.app.hasse_minkowski_daemon

# JAX アルファエンジン（pip install -e ".[jax]" が必要）
python -m src.models.pinn_jax_runtime

# Utahfile v6 フックの検証
python -m src.app.ignite --manifest Utahfile --dry-run
```

## 重要な免責事項

Utah Finance Library は、ポートフォリオ監査、ルーティング、決済指示の生成、数学的
検証のための **オープンソースのソフトウェア構成要素** を提供します。これ単体では
**次を行いません**。

- ライセンスを受けた清算機関や規制報告の代替とはなりません。
- 投資収益やリスクの排除を保証しません。
- 法務・税務・投資に関する助言ではありません。

本番利用には、独自のコンプライアンス確認、法務レビュー、承認済みの金融インフラと
の統合が必要です。
