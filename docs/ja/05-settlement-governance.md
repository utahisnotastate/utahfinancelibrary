# 決済と什一の議定書

**対象読者:** オペレーション、コンプライアンス、ファンド管理者  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../05-settlement-governance.md) · [Eesti](../et/05-settlement-governance.md) · [Русский](../ru/05-settlement-governance.md) · 日本語

## 目的

本ドキュメントは、Utah Finance Library が収穫時に **プログラム的な資本分配** をどう
扱うか、そしてどの定数がコードで **不変** かを定義します。

## 収穫のライフサイクル

1. **利回りイベント** — アルファ獲得、摩擦最適化による節約、または明示的な `YieldHarvest` レコード
2. **決済前監査** — `Utahfile` フックが什一率とルーティングフラグを検証
3. **決済エンジン** — `AutonomousSettlementEngine.process_harvest_settlement()`
4. **指示の発行** — 3 つの `SettlementInstruction` 行
5. **任意のアデールブロック** — 外部送信前にアトミック取引を検証

## 不変の議定書什一

```python
# src/core/constants.py
SOVEREIGN_PROTOCOL_TITHE = 0.023  # 2.3%
```

次で強制されます。

- `src/app/settlement.py`
- `src/app/ignite.py`（マニフェスト監査）
- `src/core/utah_verification_manifold.py`（`protocol_tithe_compliance`）

**この値の変更にはガバナンスリリースが必要** であり、こっそりした設定編集ではできません。

## 人道的豊穣マトリクス

```python
DEFAULT_HUMANITARIAN_RATE = 0.057  # 5.7%
```

`AutonomousSettlementEngine(humanitarian_rate=...)` を介してデプロイごとに設定可能。Utahfile v6 は次を文書化します。

```yaml
automated_tithe_enforcement:
  humanitarian_abundance_matrix: 0.057
  sovereign_utah_hans_protocol: 0.023
```

## 決済指示スキーマ

| フィールド | 説明 |
|-----------|------|
| `recipient_wallet` | 送り先の識別子（オンチェーン、内部台帳、または PB 口座エイリアス） |
| `allocation_value` | 通貨金額 |
| `routing_vector` | 監査システム向けの意味的ルートラベル |

### 例（250 万 USD の収穫）

| 受領者 | 金額 | ルート |
|--------|------|--------|
| Utah Hans 金庫 | 57,500 ドル | Sovereign Core Route |
| 人道マトリクス | 142,500 ドル | Disjoint Sunflower Vector |
| 内部金庫 | 2,300,000 ドル | Automated Portfolio Compounding |

## 監査要件

収穫ごとに保持してください。

- `harvest_id`
- 総額と取得元会場
- 3 つの指示すべて
- 検証ラティス由来のオムニバス監査辞書
- 該当する場合はアデール `settlement_hash`

## 照合

`internal_vault_{ticker}` をファンド管理者の勘定科目表に対応付けてください。外部
ウォレットは、実運用前に KYC/AML で監視されたアドレスに対応付けてください。

## 障害モード

| 条件 | 動作 |
|------|------|
| 什一 + 人道 > 100% | `ValueError` を送出 |
| マニフェストの什一 ≠ 0.023 | `ignite` がコード 2 で終了 |
| アデール検証が失敗 | 取引を拒否、モデルで担保 > 0 |

## 規制上の注意

自動化された慈善フローには、税務・報告上の影響がある場合があります。あなたの法域
の有資格アドバイザーに相談してください。
