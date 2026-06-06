# アデール的清算機関バイパス（Hasse-Minkowski 決済）

**対象読者:** クオンツエンジニア、決済アーキテクト、上級読者  
**バージョン:** 6.Omnibus_Adelic

> **言語:** [English](../06-adelic-clearinghouse-bypass.md) · [Eesti](../et/06-adelic-clearinghouse-bypass.md) · [Русский](../ru/06-adelic-clearinghouse-bypass.md) · 日本語

## 問題

中央清算機関（CCP）とプライムブローカーは、T+1 決済を保証するため **担保** を要求
します。機関規模では、これが **数億から数十億ドル** を低利回りの担保に固定します。

## モデル（Hasse 原理に着想）

数論における **Hasse 原理**（局所大域原理）: 適切な条件の下で、ディオファントス方程
式が $\mathbb{R}$ と全ての $\mathbb{Q}_p$ で解を持つなら、有理数解を持ちます。

これを決済のために **比喩的に** 適応します。

| 体 | コードでの実装 |
|----|----------------|
| 実 | `VaultSnapshot` の残高が想定債務をカバー |
| 局所（$\mathbb{Q}_p$ の代替） | 素数 $p \in \{2,3,5,7,11,13\}$ を法とするスケール整数合同チェック |
| 大域パス | 実 かつ 全ての局所チェックが成功 |

`global_passed=True` のとき `required_collateral=0.0` となり、
`AdelicClearinghouseEngine` が **アトミックスワップ** を実行します — 執行と決済が
一つのイベントです。

## API リファレンス

### 型

- `VaultSnapshot(vault_id, balances)`
- `AtomicTrade(trade_id, buyer_vault_id, seller_vault_id, base_asset, quote_asset, quantity, price)`
- `HasseVerificationResult` — `settlement_hash` を持つ監査レコード

### コアクラス

```python
verifier = HasseMinkowskiVerifier(primes=(2, 3, 5, 7, 11))
result = verifier.verify(trade, buyer_snapshot, seller_snapshot)

engine = AdelicClearinghouseEngine()
engine.register_vault(...)
engine.attempt_atomic_settlement(trade)
```

## 決済ハッシュ

正規化 JSON に対する SHA-256:

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

不変の監査証跡に使用してください。

## Utahfile 統合

```yaml
engine:
  settlement_protocol: "local_global_padic_verification"

services:
  - name: "adelic-clearing-bypass"
    execution_command: "python -m src.app.hasse_minkowski_daemon"
    collateral_requirement: 0.00
```

## デモシナリオ

`hasse_minkowski_daemon.py` は次をシミュレートします。

- **ファンド Alpha:** 6 億ドル USD
- **ファンド Beta:** 10,000 WETH
- 取引: 1,000 WETH @ 2,500 ドル

回避されたレガシーのモデル化担保: **25 億ドルの想定元本**（デモ指標 = 決済済み取引の `quantity × price`）。

## 制限事項（よく読んでください）

1. **簡略化した $p$ 進チェック** — 完全なアデール環の数学ではありません
2. **インメモリ台帳** — 法的枠組みなしでは CCP の代替にはなりません
3. **実チェーンでの DvP なし** — カストディと法的 ISDA/GMRA と統合してください
4. **素数集合は設定可能** — 感度分析を推奨します

## 拡張ロードマップ

- `Omnibus-Cryptographic-Custody` から実際の金庫残高をフック
- 監査証跡を追記専用ストアに永続化
- 複数資産のクロスマージングラフ
- アデール検証前にひまわりルーター経由でネッティングバッチ

## CLI

```bash
python -m src.app.hasse_minkowski_daemon
python -m src.app.hasse_minkowski_daemon --json
```
