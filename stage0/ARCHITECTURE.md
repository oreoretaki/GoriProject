# Stage 0 データアーキテクチャ

## 🏗️ システム概要

Stage 0は金融時系列データ（USD/JPY）を機械学習用に前処理する基盤システムです。データ品質・整合性・再現性を最優先に設計されています。

## 📊 データフロー図

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SQLite DB     │    │   ETL Process   │    │  Parquet Files  │
│ oanda_historical│ -> │scripts/make_tf_ │ -> │ simple_gap_aware│
│     .db         │    │   data.py       │    │     _*.parquet  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Data Quality   │              │
         │              │   Validation    │              │
         │              │scripts/run_vali-│              │
         │              │     date.py     │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Hash & CI      │              │
         │              │   Integrity     │              │
         │              │.github/workflows│              │
         │              └─────────────────┘              │
         │                                               │
         └─────────────── Archive / Backup ──────────────┘
```

## 🔄 処理フロー詳細

### 1. データ取得層 (Source Layer)
```
oanda_historical.db
├── candles_m1      ← 3.1M records (2017-2025)
├── candles_m5      ← ❌ 使用禁止（整合性破綻）
├── candles_h1      ← ❌ 使用禁止（整合性破綻）
└── metadata        ← 参考情報のみ
```

**重要制約**:
- **Volume列**: 絶対使用禁止（ティック数≠取引量）
- **上位TF**: 0.3-0.6%整合性のため使用禁止
- **M1のみ**: 唯一の信頼できるソース

### 2. ETL処理層 (Processing Layer)
```python
# scripts/make_tf_data.py の処理ステップ

Step 1: M1データ読み込み
├── SQLite → pandas DataFrame
├── Volume列完全除外  
├── タイムゾーン統一 (UTC)
└── gap_flag付与 (将来拡張用)

Step 2: Clean M1保存
├── simple_gap_aware_m1.parquet
├── 「唯一のソース真実」として保存
└── Snappy圧縮 (44.8MB)

Step 3: TF生成 (M1 → 上位TF)
├── resample(rule, label='left', closed='left')  
├── OHLC集約 (first/max/min/last)
├── 不完全バー除去 (.dropna())
└── OHLC論理チェック
```

### 3. データ出力層 (Output Layer)  
```
data/derived/
├── simple_gap_aware_m1.parquet    ← 3.1M records (44.8MB)
├── simple_gap_aware_m5.parquet    ← 630K records (10.7MB)  
├── simple_gap_aware_m15.parquet   ← 210K records (4.3MB)
├── simple_gap_aware_m30.parquet   ← 105K records (2.4MB)
├── simple_gap_aware_h1.parquet    ← 53K records (1.3MB)
├── simple_gap_aware_h4.parquet    ← 14K records (0.4MB)
├── simple_gap_aware_d.parquet     ← 3K records (0.1MB)
└── full_consistency_report.json   ← 100%整合性証明
```

## 🎯 品質保証アーキテクチャ

### 4. 検証層 (Validation Layer)
```
scripts/run_validate.py (0.7秒実行)
├── M1→全TF整合性チェック (100%必達)
├── インデックス整合性チェック  
├── タイムゾーン検証 (UTC固定)
└── exit code 0/1 でCI連携

tests/test_boundary.py
├── 境界条件テスト (08:00 M5 = 08:00-08:04 M1)
├── タイムゾーンテスト (datetime64[ns, UTC])
├── リサンプリングロジックテスト
└── OHLC論理制約テスト
```

### 5. 継続的品質管理 (CI/CD Layer)
```
.github/workflows/data_hash.yml
├── Push時自動実行
├── scripts/run_validate.py → 整合性100%確認
├── reports/hash_manifest.json → MD5一致確認  
├── tests/ → 境界条件確認
└── 失敗時赤色フィードバック
```

## 🔐 データ整合性管理

### ハッシュ管理システム
```json
reports/hash_manifest.json
{
  "simple_gap_aware_m1.parquet": {
    "md5": "becc1aa3...",
    "size_bytes": 47009413
  },
  "simple_gap_aware_m5.parquet": {
    "md5": "995a5d23...", 
    "size_bytes": 10663083
  }
}
```

**運用ルール**:
- データ変更時は必ずハッシュ更新
- CI時に実ファイルとmanifestを照合
- 不一致時は即座にビルド失敗

## ⚡ 性能アーキテクチャ

### パフォーマンス指標
| 処理 | 時間 | スループット |
|------|------|--------------|
| M1読み込み | 0.264秒 | 11.8M records/sec |
| M1→M5生成 | 0.162秒 | 19.2M records/sec |
| M5読み込み | 0.047秒 | 215.1 MB/sec |
| 整合性検証 | 0.7秒 | 全TF100%チェック |

### ストレージ最適化
- **Parquet形式**: カラム型圧縮で高効率
- **Snappy圧縮**: CPUオーバーヘッド最小
- **インデックス保持**: タイムスタンプソート済み
- **スキーマ最適化**: datetime64[ns, UTC]固定

## 🚫 データ制約・禁止事項

### Volume使用絶対禁止
```python
# ❌ 絶対禁止
df['volume_sma'] = df['volume'].rolling(20).mean()
df['vwap'] = (df['close'] * df['volume']).sum() / df['volume'].sum()

# ✅ 正しい使用
df = pd.read_parquet('simple_gap_aware_h1.parquet')  
# Volume列は存在しない
```

### 元SQLiteテーブル使用禁止
```python  
# ❌ 整合性破綻データ
df = pd.read_sql('SELECT * FROM candles_m5', conn)  # 0.6%整合性

# ✅ 整合性保証データ
df = pd.read_parquet('simple_gap_aware_m5.parquet')  # 100%整合性
```

## 🔄 依存関係・環境構築

### 必要ライブラリ
```bash
pip install pandas>=2.0.0
pip install pyarrow>=12.0.0  
pip install matplotlib>=3.7.0
pip install psutil>=5.9.0
```

### 実行環境要件
- **Python**: 3.11+
- **メモリ**: 8GB以上推奨
- **ストレージ**: 100MB (全Parquetファイル)
- **OS**: Linux/Windows/macOS対応

## 📋 メンテナンス・運用

### 定期実行推奨
```bash
# 日次データ品質チェック
python3 scripts/run_validate.py

# 週次ハッシュ整合性確認  
python3 scripts/verify_hash_manifest.py

# 月次パフォーマンステスト
python3 scripts/generate_benchmark.py
```

### トラブルシューティング
1. **整合性失敗**: `scripts/run_validate.py`の出力を確認
2. **ハッシュ不一致**: データ破損の可能性、バックアップから復元
3. **性能劣化**: `reports/benchmark.md`で履歴比較

## 🎯 Stage 0成功基準

### 必達条件
- ✅ M1→全TF整合性: **100.0%**
- ✅ Volume除外: **完全実施**  
- ✅ 検証時間: **<1分**
- ✅ OHLC論理制約: **違反0件**
- ✅ CI自動化: **実装済み**

**Stage 0 Ready**: 前処理・学習パイプライン実行可能状態