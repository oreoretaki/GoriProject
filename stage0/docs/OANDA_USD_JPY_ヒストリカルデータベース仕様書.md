# OANDA USD/JPY ヒストリカルデータベース仕様書
このドキュメントは、OANDA取引所から取得したUSD/JPY通貨ペアの過去価格データを格納したSQLiteデータベース（oanda_historical.db）の詳細仕様を記載しています。

生成日時: 2025-07-05 11:59:21
## 1. データベース概要
- **ファイル名**: `oanda_historical.db`
- **格納場所**: `data/oanda_historical.db`
- **SQLiteバージョン**: 3.45.1
- **ファイルサイズ**: 715.94 MB
- **総テーブル数**: 12
- **通貨ペア**: USD/JPY
- **データ期間**: 2017年1月3日 〜 2025年6月17日（約8.5年）

## 2. テーブル構造詳細

### metadata - メタデータテーブル
**レコード数**: 1

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| granularity | TEXT | No | No |  |
| first_timestamp | DATETIME | Yes | No |  |
| last_timestamp | DATETIME | Yes | No |  |
| total_records | INTEGER | Yes | No |  |
| last_update | DATETIME | Yes | No |  |

#### インデックス
- **sqlite_autoindex_metadata_1** (UNIQUE): instrument, granularity

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| total_records | 3104383.000000 | 3104383.000000 | 3104383.000000 |

### sqlite_sequence - 
**レコード数**: 8

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| name |  | Yes | No |  |
| seq |  | Yes | No |  |

### candles_m1 - 1分足データ
**レコード数**: 3,104,383

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_m1_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_m1_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 101.200000 | 161.950000 | 123.050375 |
| high | 101.340000 | 161.950000 | 123.060225 |
| low | 101.180000 | 161.944000 | 123.040472 |
| close | 101.202000 | 161.950000 | 123.050388 |
| volume | 1.000000 | 7293.000000 | 91.947152 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T23:24:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-03T11:55:00 | 2017-01-03T11:57:00 | 0.001 |
| 2017-01-03T12:12:00 | 2017-01-03T12:14:00 | 0.001 |
| 2017-01-03T13:41:00 | 2017-01-03T13:43:00 | 0.001 |
| 2017-01-03T14:19:00 | 2017-01-03T14:21:00 | 0.001 |
| 2017-01-03T14:51:00 | 2017-01-03T14:53:00 | 0.001 |
| 2017-01-04T06:59:00 | 2017-01-04T07:02:00 | 0.002 |
| 2017-01-04T07:02:00 | 2017-01-04T07:04:00 | 0.001 |
| 2017-01-04T07:30:00 | 2017-01-04T07:32:00 | 0.001 |
| 2017-01-05T06:22:00 | 2017-01-05T06:24:00 | 0.001 |
| 2017-01-05T06:59:00 | 2017-01-05T07:01:00 | 0.001 |

### candles_m5 - 5分足データ
**レコード数**: 630,474

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_m5_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_m5_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 101.346000 | 161.950000 | 122.959494 |
| high | 101.402000 | 161.950000 | 122.983390 |
| low | 101.180000 | 161.932000 | 122.935295 |
| close | 101.348000 | 161.950000 | 122.959554 |
| volume | 1.000000 | 12048.000000 | 452.737425 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T23:20:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T06:55:00 | 2017-01-09T07:00:00 | 2.003 |
| 2017-01-14T06:55:00 | 2017-01-16T07:00:00 | 2.003 |
| 2017-01-21T06:55:00 | 2017-01-23T07:00:00 | 2.003 |
| 2017-01-28T06:55:00 | 2017-01-30T07:00:00 | 2.003 |
| 2017-02-04T06:55:00 | 2017-02-06T07:00:00 | 2.003 |
| 2017-02-11T06:55:00 | 2017-02-13T07:00:00 | 2.003 |
| 2017-02-18T06:55:00 | 2017-02-20T07:00:00 | 2.003 |
| 2017-02-21T06:55:00 | 2017-02-21T07:05:00 | 0.007 |
| 2017-02-25T06:55:00 | 2017-02-27T07:00:00 | 2.003 |
| 2017-03-04T06:55:00 | 2017-03-06T07:00:00 | 2.003 |

### candles_m15 - 15分足データ
**レコード数**: 210,477

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_m15_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_m15_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 101.346000 | 161.941000 | 122.943144 |
| high | 101.954000 | 161.950000 | 122.985774 |
| low | 101.180000 | 161.918000 | 122.899400 |
| close | 101.348000 | 161.942000 | 122.943379 |
| volume | 1.000000 | 29364.000000 | 1356.153760 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T23:15:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T06:45:00 | 2017-01-09T07:00:00 | 2.01 |
| 2017-01-14T06:45:00 | 2017-01-16T07:00:00 | 2.01 |
| 2017-01-21T06:45:00 | 2017-01-23T07:00:00 | 2.01 |
| 2017-01-28T06:45:00 | 2017-01-30T07:00:00 | 2.01 |
| 2017-02-04T06:45:00 | 2017-02-06T07:00:00 | 2.01 |
| 2017-02-11T06:45:00 | 2017-02-13T07:00:00 | 2.01 |
| 2017-02-18T06:45:00 | 2017-02-20T07:00:00 | 2.01 |
| 2017-02-25T06:45:00 | 2017-02-27T07:00:00 | 2.01 |
| 2017-03-04T06:45:00 | 2017-03-06T07:00:00 | 2.01 |
| 2017-03-11T06:45:00 | 2017-03-13T06:00:00 | 1.969 |

### candles_m30 - 30分足データ
**レコード数**: 105,252

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_m30_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_m30_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 101.840000 | 161.941000 | 122.941610 |
| high | 102.178000 | 161.950000 | 123.002449 |
| low | 101.180000 | 161.909000 | 122.878184 |
| close | 101.838000 | 161.942000 | 122.942079 |
| volume | 1.000000 | 47854.000000 | 2711.959630 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T23:00:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T06:30:00 | 2017-01-09T07:00:00 | 2.021 |
| 2017-01-14T06:30:00 | 2017-01-16T07:00:00 | 2.021 |
| 2017-01-21T06:30:00 | 2017-01-23T07:00:00 | 2.021 |
| 2017-01-28T06:30:00 | 2017-01-30T07:00:00 | 2.021 |
| 2017-02-04T06:30:00 | 2017-02-06T07:00:00 | 2.021 |
| 2017-02-11T06:30:00 | 2017-02-13T07:00:00 | 2.021 |
| 2017-02-18T06:30:00 | 2017-02-20T07:00:00 | 2.021 |
| 2017-02-25T06:30:00 | 2017-02-27T07:00:00 | 2.021 |
| 2017-03-04T06:30:00 | 2017-03-06T07:00:00 | 2.021 |
| 2017-03-11T06:30:00 | 2017-03-13T06:00:00 | 1.979 |

### candles_h1 - 1時間足データ
**レコード数**: 52,632

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_h1_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_h1_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 102.072000 | 161.941000 | 122.940586 |
| high | 102.342000 | 161.950000 | 123.027569 |
| low | 101.180000 | 161.896000 | 122.848959 |
| close | 102.070000 | 161.942000 | 122.941529 |
| volume | 1.000000 | 91270.000000 | 5423.300939 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T23:00:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T06:00:00 | 2017-01-09T07:00:00 | 2.042 |
| 2017-01-14T06:00:00 | 2017-01-16T07:00:00 | 2.042 |
| 2017-01-21T06:00:00 | 2017-01-23T07:00:00 | 2.042 |
| 2017-01-28T06:00:00 | 2017-01-30T07:00:00 | 2.042 |
| 2017-02-04T06:00:00 | 2017-02-06T07:00:00 | 2.042 |
| 2017-02-11T06:00:00 | 2017-02-13T07:00:00 | 2.042 |
| 2017-02-18T06:00:00 | 2017-02-20T07:00:00 | 2.042 |
| 2017-02-25T06:00:00 | 2017-02-27T07:00:00 | 2.042 |
| 2017-03-04T06:00:00 | 2017-03-06T07:00:00 | 2.042 |
| 2017-03-11T06:00:00 | 2017-03-13T06:00:00 | 2.0 |

### candles_h4 - 4時間足データ
**レコード数**: 13,601

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_h4_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_h4_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 102.084000 | 161.941000 | 122.942421 |
| high | 102.566000 | 161.950000 | 123.120235 |
| low | 101.180000 | 161.726000 | 122.755746 |
| close | 102.086000 | 161.942000 | 122.945022 |
| volume | 40.000000 | 281919.000000 | 20986.631498 |

#### データ期間
- **開始日時**: 2017-01-03T08:00:00
- **終了日時**: 2025-06-17T20:00:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T04:00:00 | 2017-01-09T04:00:00 | 2.0 |
| 2017-01-14T04:00:00 | 2017-01-16T04:00:00 | 2.0 |
| 2017-01-21T04:00:00 | 2017-01-23T04:00:00 | 2.0 |
| 2017-01-28T04:00:00 | 2017-01-30T04:00:00 | 2.0 |
| 2017-02-04T04:00:00 | 2017-02-06T04:00:00 | 2.0 |
| 2017-02-11T04:00:00 | 2017-02-13T04:00:00 | 2.0 |
| 2017-02-18T04:00:00 | 2017-02-20T04:00:00 | 2.0 |
| 2017-02-25T04:00:00 | 2017-02-27T04:00:00 | 2.0 |
| 2017-03-04T04:00:00 | 2017-03-06T04:00:00 | 2.0 |
| 2017-03-11T04:00:00 | 2017-03-13T04:00:00 | 2.0 |

### candles_d - 日足データ
**レコード数**: 2,640

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_d_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_d_1** (UNIQUE): instrument, timestamp

#### 価格統計
| Column | Min | Max | Average |
|--------|-----|-----|--------|
| open | 102.292000 | 161.663000 | 122.931916 |
| high | 103.254000 | 161.950000 | 123.355676 |
| low | 101.180000 | 161.269000 | 122.499158 |
| close | 102.294000 | 161.664000 | 122.945890 |
| volume | 1423.000000 | 962181.000000 | 108120.899621 |

#### データ期間
- **開始日時**: 2017-01-03T00:00:00
- **終了日時**: 2025-06-17T00:00:00
- **総取引日数**: 2,640日

#### データギャップ（最初の10件）
| From | To | Gap (days) |
|------|----|-----------|
| 2017-01-07T00:00:00 | 2017-01-09T00:00:00 | 2.0 |
| 2017-01-14T00:00:00 | 2017-01-16T00:00:00 | 2.0 |
| 2017-01-21T00:00:00 | 2017-01-23T00:00:00 | 2.0 |
| 2017-01-28T00:00:00 | 2017-01-30T00:00:00 | 2.0 |
| 2017-02-04T00:00:00 | 2017-02-06T00:00:00 | 2.0 |
| 2017-02-11T00:00:00 | 2017-02-13T00:00:00 | 2.0 |
| 2017-02-18T00:00:00 | 2017-02-20T00:00:00 | 2.0 |
| 2017-02-25T00:00:00 | 2017-02-27T00:00:00 | 2.0 |
| 2017-03-04T00:00:00 | 2017-03-06T00:00:00 | 2.0 |
| 2017-03-11T00:00:00 | 2017-03-13T00:00:00 | 2.0 |

### candles_w - 週足データ（未使用）
**レコード数**: 0

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_w_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_w_1** (UNIQUE): instrument, timestamp

### candles_m - 月足データ（未使用）
**レコード数**: 0

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| timestamp | DATETIME | No | No | Candle timestamp (UTC) |
| open | REAL | No | No | Opening price |
| high | REAL | No | No | Highest price |
| low | REAL | No | No | Lowest price |
| close | REAL | No | No | Closing price |
| volume | INTEGER | Yes | No | ⚠️ ティック数（使用禁止・実際の取引量ではない） |
| complete | BOOLEAN | Yes | No | Candle completion status |
| source | TEXT | Yes | No |  |
| created_at | DATETIME | Yes | No | Record creation timestamp |

#### インデックス
- **idx_candles_m_instrument_timestamp** (INDEX): instrument, timestamp
- **sqlite_autoindex_candles_m_1** (UNIQUE): instrument, timestamp

### data_integrity - データ整合性チェックテーブル
**レコード数**: 0

#### カラム構造
| Column | Type | Nullable | Primary Key | Description |
|--------|------|----------|-------------|-------------|
| id | INTEGER | Yes | Yes |  |
| check_timestamp | DATETIME | Yes | No |  |
| instrument | TEXT | No | No | Trading pair (e.g., USD_JPY) |
| granularity | TEXT | No | No |  |
| date_from | DATETIME | Yes | No |  |
| date_to | DATETIME | Yes | No |  |
| expected_count | INTEGER | Yes | No |  |
| actual_count | INTEGER | Yes | No |  |
| missing_count | INTEGER | Yes | No |  |
| status | TEXT | Yes | No |  |
| details | TEXT | Yes | No |  |

## 3. データ品質レポート

## 4. テーブル間の関係性
- **candles_m1** → **candles_m5**: temporal_aggregation (2640 common trading days)
- **candles_m1** → **candles_m15**: temporal_aggregation (2640 common trading days)
- **candles_m1** → **candles_m30**: temporal_aggregation (2640 common trading days)
- **candles_m1** → **candles_h1**: temporal_aggregation (2640 common trading days)

## 5. 使用例

### Basic Query Examples
```sql
-- Get latest 100 candles from 1-minute data
SELECT * FROM candles_m1 
WHERE instrument = 'USD_JPY' 
ORDER BY timestamp DESC 
LIMIT 100;

-- Get daily OHLC for specific date range
SELECT * FROM candles_d 
WHERE timestamp BETWEEN '2024-01-01' AND '2024-12-31' 
ORDER BY timestamp;

-- Calculate simple moving average (OHLC価格のみ使用)
SELECT 
    timestamp,
    open,
    high,
    low,
    close,
    AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma20
FROM candles_h1
WHERE instrument = 'USD_JPY'
-- 注意: volumeカラムは絶対に使用しないこと
```

### Python Usage Example
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/oanda_historical.db')

# Load 1-hour candles into DataFrame (volumeを除外)
df = pd.read_sql_query(
    """
    SELECT timestamp, open, high, low, close 
    FROM candles_h1 
    WHERE instrument = 'USD_JPY'
    """,
    conn,
    parse_dates=['timestamp'],
    index_col='timestamp'
)

# Calculate technical indicators (OHLC価格のみ使用)
df['sma20'] = df['close'].rolling(window=20).mean()
df['rsi'] = calculate_rsi(df['close'], period=14)
# 注意: volumeは実際の取引量ではないため絶対に使用しない
```

## 6. 重要な注意事項
1. All timestamps are in UTC
2. Price data precision: 6 decimal places
3. **⚠️ VOLUME フィールドは絶対に使用禁止** 
   - Volumeは実際の取引量ではなく、ティック数（価格更新回数）を表す
   - 真の取引量データではないため、分析・学習には使用してはいけない
   - OHLC価格データのみを使用すること
4. Weekend gaps are normal in forex data
5. Some candles may be missing during low liquidity periods
6. The 'complete' field indicates whether the candle is finalized
7. Higher timeframes are aggregated from lower timeframes (source='aggregated')

## 7. データ整合性
- All OHLC data follows the rule: Low ≤ Open, Close ≤ High
- Timestamps are sequential within each table
- Higher timeframe candles are consistent with lower timeframe aggregations
- The metadata table tracks the overall data range and last update time
