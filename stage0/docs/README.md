# OANDA USD/JPY データアクセス ガイド

このフォルダには、OANDA取引所から取得したUSD/JPY通貨ペアの過去価格データに関するドキュメントと、Stage 0準備完了済みの安全なデータが格納されています。

## 🎯 Stage 0 Ready - 使用可能データ

### 推奨: Gap-aware安全データ（Volume除外・100%整合性保証済み）

**データ保存場所**: `../data/derived/`

| 時間足 | ファイル名 | レコード数 | 整合性 | 推奨用途 |
|--------|------------|------------|--------|----------|
| 5分足 | simple_gap_aware_m5.parquet | 630,474 | **100.0%** | 短期分析 |
| 15分足 | simple_gap_aware_m15.parquet | 210,477 | **100.0%** | 中期分析 |
| 30分足 | simple_gap_aware_m30.parquet | 105,252 | **100.0%** | 中期分析 |
| 1時間足 | simple_gap_aware_h1.parquet | 52,632 | **100.0%** | 長期分析 |
| 4時間足 | simple_gap_aware_h4.parquet | 13,601 | **100.0%** | 長期分析 |
| 日足 | simple_gap_aware_d.parquet | 2,640 | **100.0%** | 戦略分析 |

**使用例**:
```python
import pandas as pd

# ✅ 推奨: Gap-aware安全データ読み込み
df = pd.read_parquet('../data/derived/simple_gap_aware_h1.parquet')
# カラム: timestamp, open, high, low, close (volumeは存在しない)
```

### ⚠️ 非推奨: 元のSQLiteデータベース

**データベース名**: `../data/oanda_historical.db`  
**⚠️ 問題**: TF整合性破綻（0-3%）、Volume汚染、使用非推奨

## 📋 主要ドキュメント

### 1. 安全なデータアクセス方法.md 🔥
- **内容**: 推奨データアクセス方法とコード例
- **重要度**: ⭐⭐⭐⭐⭐
- **用途**: 実装時の必読ガイド、安全なコーディングパターン

### 2. Stage0_問題完全解決レポート.md
- **内容**: Stage 0準備完了の解決プロセス全体
- **重要度**: ⭐⭐⭐⭐⭐
- **用途**: プロジェクト状況の把握、問題解決の経緯

### 3. VOLUME使用禁止の注意事項.md
- **内容**: Volume使用禁止の詳細な理由と対策
- **重要度**: ⭐⭐⭐⭐⭐
- **用途**: 開発チーム必読、Volume誤使用防止

### 4. OANDA_USD_JPY_ヒストリカルデータベース仕様書.md
- **内容**: 元データベースの技術仕様（参考用）
- **重要度**: ⭐⭐⭐
- **用途**: 元データ理解、トラブルシューティング

## 🚫 重要な使用制限

### Volume使用絶対禁止
- **理由**: volumeは実際の取引量ではなく、ティック数（価格更新回数）
- **影響**: 分析結果の完全な歪み
- **対策**: **safe_ohlc_*.parquet**では既に完全除外済み

### 元SQLiteテーブルの使用禁止
- **理由**: TF整合性が0-3%レベルで破綻
- **影響**: マルチTF分析の完全な失敗
- **対策**: **simple_gap_aware_*.parquet**で100%整合性保証済み

## 📊 データ品質保証

✅ **完全解決済み**:
- TF整合性: **100.0%**（以前は0.3-0.6%）
- Volume除外: 完全実施
- OHLC検証: 論理違反0件
- Gap対応: リサンプリング最適化済み
- I/O最適化: Parquet高速アクセス

## 🚀 Stage 0 実行準備完了

**前処理・学習が安全に実行可能な状態です**

```python
# Stage 0 推奨データアクセスパターン
def load_safe_data(timeframe='h1'):
    """Gap-aware安全データ読み込み（100%整合性）"""
    return pd.read_parquet(f'../data/derived/simple_gap_aware_{timeframe}.parquet')

# マルチTF分析も100%整合性保証済みで安全
df_m15 = load_safe_data('m15')
df_h1 = load_safe_data('h1') 
df_d = load_safe_data('d')
```

※ データ期間: 2017年1月3日 ～ 2025年6月17日（約8.5年）  
※ すべての時刻はUTC（協定世界時）