# ⚠️ VOLUME フィールド使用禁止の重要な注意事項

## 概要
OANDAヒストリカルデータベース内の**volumeカラムは絶対に使用してはいけません**。

## 理由

### volumeは実際の取引量ではない
- **volumeの正体**: ティック数（価格更新回数）
- **実際の取引量**: 通貨の売買された実際の数量
- **混同の危険性**: volumeを取引量と誤解して使用すると、完全に間違った分析結果になる

### FXにおける真の取引量データ
- FX市場は分散型市場のため、真の取引量データは存在しない
- 各ブローカーは自社の顧客取引のみを把握
- 市場全体の取引量は測定不可能

## 使用してはいけない理由

### 1. データの意味が異なる
```python
# ❌ 間違った理解
volume = 1000  # 1000通貨分の取引があった（誤解）

# ✅ 正しい理解  
volume = 1000  # 1000回価格が更新された（ティック数）
```

### 2. 分析結果の歪み
- **出来高分析**: 完全に無意味
- **流動性指標**: ティック数≠流動性
- **ボリューム加重価格**: 意味のない計算結果

### 3. 機械学習への悪影響
- **特徴量として使用**: ノイズとして機能し、モデル性能を悪化
- **重み付け**: 不適切な重み付けによる学習の歪み
- **正規化**: 意味のないスケーリング

## 使用すべきデータ

### ✅ 使用可能なカラム
- **timestamp**: 時刻情報
- **open**: 始値
- **high**: 高値  
- **low**: 安値
- **close**: 終値
- **complete**: ローソク確定フラグ

### ✅ 推奨される特徴量
```python
# 価格ベースの特徴量
price_features = [
    'open', 'high', 'low', 'close',
    'hl_ratio',      # (high - low) / close
    'oc_ratio',      # (open - close) / close  
    'typical_price', # (high + low + close) / 3
    'price_range',   # high - low
]

# テクニカル指標（価格のみ使用）
technical_features = [
    'sma_20',        # 単純移動平均
    'ema_12',        # 指数移動平均
    'rsi_14',        # RSI
    'bollinger_upper', # ボリンジャーバンド
    'macd',          # MACD
]
```

## 実装時の注意点

### SQLクエリ
```sql
-- ✅ 正しい
SELECT timestamp, open, high, low, close 
FROM candles_h1 
WHERE instrument = 'USD_JPY';

-- ❌ 間違い  
SELECT timestamp, open, high, low, close, volume
FROM candles_h1 
WHERE instrument = 'USD_JPY';
```

### Pandas DataFrame
```python
# ✅ 正しい
df = pd.read_sql_query(
    "SELECT timestamp, open, high, low, close FROM candles_h1",
    conn,
    parse_dates=['timestamp'],
    index_col='timestamp'
)

# ❌ 間違い
df = pd.read_sql_query(
    "SELECT * FROM candles_h1",  # volumeも含まれてしまう
    conn
)
```

## チェックリスト

開発時は以下を必ず確認：

- [ ] SQLクエリにvolumeカラムが含まれていないか？
- [ ] DataFrameのカラムリストにvolumeが含まれていないか？
- [ ] 特徴量生成でvolumeを使用していないか？
- [ ] 機械学習のinput_featuresにvolumeが含まれていないか？
- [ ] 可視化やレポートでvolumeを使用していないか？

## まとめ

**OHLC価格データのみを使用し、volumeは完全に無視すること**

これは単なる推奨事項ではなく、**データの正確性と分析結果の信頼性を保つための必須要件**です。