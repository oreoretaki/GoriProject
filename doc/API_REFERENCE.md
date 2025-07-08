# Stage 1 API リファレンス

## 📚 概要

Stage 1の各モジュールのクラス・関数の詳細仕様を記載。開発者が実装を理解・拡張する際の参考資料。

---

## 🔧 データ処理モジュール

### `MultiTFWindowSampler`

**ファイル**: `src/window_sampler.py`

マルチタイムフレームから同期されたウィンドウをサンプリングするクラス。

#### コンストラクタ

```python
def __init__(
    self,
    tf_data: Dict[str, pd.DataFrame],
    seq_len: int,
    split: str = "train",
    val_split: float = 0.2,
    min_coverage: float = 0.8
)
```

**パラメータ**:
- `tf_data`: `{tf_name: DataFrame}` 形式のタイムフレームデータ
- `seq_len`: M1でのシーケンス長
- `split`: `"train"` または `"val"`  
- `val_split`: 検証データ割合 (0.0-1.0)
- `min_coverage`: 最小データカバレッジ (0.0-1.0)

**戻り値**: `MultiTFWindowSampler` インスタンス

#### 主要メソッド

##### `__getitem__(idx: int) -> Dict[str, pd.DataFrame]`

指定インデックスのマルチTFウィンドウを取得。

**パラメータ**:
- `idx`: ウィンドウインデックス (0 <= idx < len(sampler))

**戻り値**: `{tf_name: DataFrame}` ウィンドウデータ

**例外**:
- `IndexError`: インデックスが範囲外

##### `get_sample_window_info(idx: int = 0) -> Dict`

デバッグ用ウィンドウ情報取得。

**戻り値**:
```python
{
    "window_index": int,
    "start_time": pd.Timestamp,
    "end_time": pd.Timestamp, 
    "duration_hours": float,
    "tf_lengths": Dict[str, int]
}
```

---

### `FeatureEngineer`

**ファイル**: `src/feature_engineering.py`

OHLC データから6特徴量 `[open, high, low, close, Δclose, %body]` を計算。

#### コンストラクタ

```python
def __init__(self, config: dict)
```

**パラメータ**:
- `config`: 設定辞書 (`data.n_features`, `data.timeframes` を含む)

#### 主要メソッド

##### `process_window(window_data: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor]`

ウィンドウデータを特徴量・ターゲットに変換。

**パラメータ**:
- `window_data`: `{tf_name: DataFrame}` マルチTFウィンドウ

**戻り値**:
- `features`: `[n_tf, seq_len, n_features]` 特徴量テンソル
- `targets`: `[n_tf, seq_len, 4]` ターゲット（OHLC）テンソル

##### `get_feature_stats(window_data: Dict[str, pd.DataFrame]) -> Dict`

特徴量統計情報取得（デバッグ用）。

**戻り値**: TFごとの統計情報辞書

---

### `MaskingStrategy`

**ファイル**: `src/masking.py`

ランダム連続ブロックマスキング戦略の実装。

#### コンストラクタ

```python
def __init__(self, config: dict)
```

**パラメータ**:
- `config`: マスキング設定 (`masking.*` パラメータ)

#### 主要メソッド

##### `generate_masks(features: torch.Tensor, seed: int = None) -> torch.Tensor`

マルチTF特徴量用マスクを生成。

**パラメータ**:
- `features`: `[batch, n_tf, seq_len, n_features]` 特徴量
- `seed`: ランダムシード（再現性用、オプション）

**戻り値**:
- `masks`: `[batch, n_tf, seq_len]` マスクテンソル (1=マスク, 0=観測)

##### `get_mask_statistics(masks: torch.Tensor) -> Dict`

マスク統計取得（デバッグ用）。

**戻り値**:
```python
{
    tf_name: {
        'mask_ratio': float,
        'masked_tokens': int,
        'total_tokens': int,
        'n_blocks': int,
        'block_lengths': List[int],
        'avg_block_length': float
    }
}
```

---

### `TFNormalizer`

**ファイル**: `src/normalization.py`

タイムフレーム別z-score正規化クラス。

#### コンストラクタ

```python
def __init__(self, config: dict, cache_stats: bool = True)
```

**パラメータ**:
- `config`: 正規化設定
- `cache_stats`: 統計ファイルキャッシュの有効/無効

#### 主要メソッド

##### `fit(tf_data: Dict[str, pd.DataFrame]) -> None`

訓練データから正規化統計を計算・保存。

**パラメータ**:
- `tf_data`: `{tf_name: DataFrame}` TFデータ

##### `normalize(features: torch.Tensor) -> torch.Tensor`

特徴量を正規化。

**パラメータ**:
- `features`: `[batch, n_tf, seq_len, n_features]` 正規化前特徴量

**戻り値**:
- `normalized`: 正規化後特徴量 (同形状)

##### `denormalize(normalized: torch.Tensor, tf_indices: Optional[torch.Tensor] = None) -> torch.Tensor`

正規化済みデータを元スケールに復元。

**パラメータ**:
- `normalized`: 正規化済みテンソル
- `tf_indices`: 対象TFインデックス（Noneで全TF）

**戻り値**: 元スケールテンソル

---

## 🧠 モデルモジュール

### `Stage1Model`

**ファイル**: `src/model.py`

マルチTF自己教師あり再構築モデルのメインクラス。

#### コンストラクタ

```python
def __init__(self, config: dict)
```

**パラメータ**:
- `config`: モデル設定辞書

#### 主要メソッド

##### `forward(features: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]`

フォワードパス実行。

**パラメータ**:
- `features`: `[batch, n_tf, seq_len, n_features]` 入力特徴量
- `masks`: `[batch, n_tf, seq_len]` パディングマスク（オプション）

**戻り値**:
```python
{
    'reconstructed': torch.Tensor,  # [batch, n_tf, seq_len, 4] 再構築OHLC
    'encoded': torch.Tensor         # [batch, n_tf, latent_len, d_model] エンコード表現
}
```

##### `get_model_info() -> Dict`

モデル情報取得。

**戻り値**:
```python
{
    'total_parameters': int,
    'trainable_parameters': int,
    'model_size_mb': float,
    'architecture': Dict
}
```

---

### `TFSpecificStem`

**ファイル**: `src/model.py`

TF固有ステム（1D depth-wise CNN）。

#### コンストラクタ

```python
def __init__(self, n_features: int, d_model: int, kernel_size: int = 3)
```

**パラメータ**:
- `n_features`: 入力特徴量数
- `d_model`: 出力次元数
- `kernel_size`: カーネルサイズ

#### メソッド

##### `forward(x: torch.Tensor) -> torch.Tensor`

**パラメータ**:
- `x`: `[batch, seq_len, n_features]` 入力

**戻り値**:
- `out`: `[batch, seq_len, d_model]` 投影済み特徴量

---

## 🎯 損失関数モジュール

### `Stage1CombinedLoss`

**ファイル**: `src/losses.py`

4成分統合損失関数。

#### コンストラクタ

```python
def __init__(self, config: dict)
```

**パラメータ**:
- `config`: 損失設定 (`loss.*` パラメータ)

#### 主要メソッド

##### `forward(pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor, m1_data: torch.Tensor = None) -> Dict[str, torch.Tensor]`

統合損失計算。

**パラメータ**:
- `pred`: `[batch, n_tf, seq_len, 4]` 予測OHLC
- `target`: `[batch, n_tf, seq_len, 4]` 正解OHLC
- `masks`: `[batch, n_tf, seq_len]` マスク
- `m1_data`: `[batch, seq_len, 4]` M1データ（クロス損失用、オプション）

**戻り値**:
```python
{
    'total': torch.Tensor,      # 総損失
    'recon_tf': torch.Tensor,   # Huber損失
    'spec_tf': torch.Tensor,    # STFT損失
    'cross': torch.Tensor,      # クロス整合性損失  
    'amp_phase': torch.Tensor   # 振幅位相損失
}
```

---

### `HuberLoss`

Huber損失単体実装。

#### `forward(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor`

**パラメータ**:
- `pred`, `target`: 予測・正解OHLC
- `mask`: マスク（1=損失計算対象）

**戻り値**: スカラー損失

---

### `STFTLoss`

マルチ解像度STFT損失。

#### `forward(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor`

**パラメータ**: `HuberLoss` と同様

**戻り値**: STFT損失

---

## 📊 データローダーモジュール

### `Stage1Dataset`

**ファイル**: `src/data_loader.py`

PyTorch Dataset実装。

#### コンストラクタ

```python
def __init__(
    self,
    data_dir: str,
    config: dict,
    split: str = "train",
    cache_stats: bool = True
)
```

**パラメータ**:
- `data_dir`: Stage 0出力ディレクトリパス
- `config`: 設定辞書
- `split`: `"train"` または `"val"`
- `cache_stats`: 正規化統計キャッシュ有効/無効

#### 主要メソッド

##### `__getitem__(idx: int) -> Dict[str, torch.Tensor]`

**戻り値**:
```python
{
    'features': torch.Tensor,    # [n_tf, seq_len, n_features] マスク済み特徴量
    'targets': torch.Tensor,     # [n_tf, seq_len, 4] OHLC正解
    'masks': torch.Tensor,       # [n_tf, seq_len] マスク情報
    'tf_ids': torch.Tensor       # [n_tf] TF識別子
}
```

---

### `create_stage1_dataloaders`

**ファイル**: `src/data_loader.py`

DataLoaderペア作成関数。

```python
def create_stage1_dataloaders(
    data_dir: str,
    config: dict,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]
```

**パラメータ**:
- `data_dir`: データディレクトリ
- `config`: 設定辞書
- `batch_size`: バッチサイズ（Noneで設定ファイル値使用）

**戻り値**:
- `train_loader`: 訓練用DataLoader
- `val_loader`: 検証用DataLoader

---

## 🚀 実行スクリプト

### `train_stage1.py`

**ファイル**: `scripts/train_stage1.py`

#### コマンドライン引数

```bash
python3 scripts/train_stage1.py \
  --config CONFIG_PATH \
  --data_dir DATA_DIR \
  [--gpus GPUS] \
  [--resume_from CHECKPOINT]
```

**引数**:
- `--config`: 設定ファイルパス（必須）
- `--data_dir`: データディレクトリパス（必須）
- `--gpus`: GPU数（デフォルト: 1）
- `--resume_from`: 再開用チェックポイント（オプション）

#### 出力

- チェックポイント: `checkpoints/stage1-{epoch}-{val_correlation_mean}.ckpt`
- ログ: `logs/stage1/` (TensorBoard)

---

### `evaluate_stage1.py`

**ファイル**: `scripts/evaluate_stage1.py`

#### コマンドライン引数

```bash
python3 scripts/evaluate_stage1.py \
  --config CONFIG_PATH \
  --ckpt CHECKPOINT_PATH \
  --data_dir DATA_DIR \
  [--output OUTPUT_FILE]
```

**引数**:
- `--config`: 設定ファイルパス（必須）
- `--ckpt`: 評価対象チェックポイント（必須）
- `--data_dir`: データディレクトリパス（必須）  
- `--output`: 結果出力ファイル（デフォルト: `evaluation_results.json`）

#### 出力JSON形式

```json
{
  "correlation_per_tf": {
    "m1": {"mean": 0.85, "ohlc": [0.84, 0.86, 0.84, 0.87]},
    "m5": {"mean": 0.82, "ohlc": [0.81, 0.83, 0.80, 0.85]},
    ...
  },
  "reconstruction_quality": {
    "m1": {"mse": 0.001, "mae": 0.02, "rmse": 0.032},
    ...
  },
  "spectral_delta": {
    "m1": {"per_feature": [0.1, 0.12, 0.11, 0.09], "mean": 0.105},
    ...
  },
  "consistency_ratio": {
    "m5": {"ratio": 0.96, "consistent_samples": 4800, "total_samples": 5000},
    ...
  },
  "summary": {
    "mean_correlation": 0.83,
    "mean_mse": 0.0015,
    "mean_spectral_delta": 0.11,
    "mean_consistency_ratio": 0.95
  }
}
```

---

## 🧪 テストAPI

### `test_stage1_data.py`

**ファイル**: `tests/test_stage1_data.py`

#### 実行

```bash
cd tests
python3 test_stage1_data.py
```

#### 終了コード

- `0`: 全テスト合格
- `1`: テスト失敗

#### テスト項目

1. **ウィンドウサンプラー基本テスト**
   - 有効ウィンドウ数 > 0
   - M1シーケンス長一致
   - 全TF含有確認

2. **特徴量エンジニアリングテスト**
   - 出力形状確認
   - NaN/Inf検出
   - OHLC妥当性確認

3. **マスキング戦略テスト** 
   - マスク形状確認
   - マスク率範囲確認
   - 連続ブロック検証

4. **正規化テスト**
   - 統計計算確認
   - 形状保持確認
   - 逆正規化確認

5. **データセット統合テスト**
   - Stage 0データ読み込み
   - サンプル取得確認
   - 必要キー存在確認

6. **データ整列テスト**
   - 時間整列確認
   - OHLC論理制約確認

---

## ⚙️ 設定ファイル仕様

### `configs/base.yaml`

#### データ設定
```yaml
data:
  seq_len: 200                    # M1シーケンス長
  n_timeframes: 6                 # TF数
  n_features: 6                   # 特徴量数  
  timeframes: [m1, m5, m15, m30, h1, h4, d]
  data_dir: "../data/derived"     # データパス
  stats_file: "stats.json"        # 正規化統計ファイル
```

#### マスキング設定
```yaml
masking:
  mask_ratio: 0.15                # マスク率
  mask_span_min: 5                # 最小スパン
  mask_span_max: 60               # 最大スパン  
  sync_across_tf: true            # TF間同期
```

#### モデル設定
```yaml
model:
  tf_stem:
    kernel_size: 3                # CNN カーネルサイズ
    d_model: 128                  # 埋め込み次元
  encoder:
    n_layers: 8                   # エンコーダー層数
    d_model: 128                  # モデル次元
    cross_attn_every: 2           # クロス注意頻度
  bottleneck:
    latent_len: 50                # 圧縮長  
    stride: 4                     # ストライド
```

#### 訓練設定
```yaml
training:
  batch_size: 24                  # バッチサイズ
  epochs: 40                      # エポック数
  optimizer:
    name: "AdamW"
    betas: [0.9, 0.98]
    weight_decay: 0.01
  scheduler:
    name: "OneCycleLR"
    max_lr: 5.0e-4
    div_factor: 3.33
    final_div_factor: 50
```

---

## 🔗 型ヒント

### 主要型定義

```python
from typing import Dict, List, Tuple, Optional, Union
import torch
import pandas as pd

# データ型
TFData = Dict[str, pd.DataFrame]
WindowData = Dict[str, pd.DataFrame]
ConfigDict = Dict[str, any]

# テンソル型
FeatureTensor = torch.Tensor  # [batch, n_tf, seq_len, n_features]
TargetTensor = torch.Tensor   # [batch, n_tf, seq_len, 4]
MaskTensor = torch.Tensor     # [batch, n_tf, seq_len]

# 出力型
ModelOutput = Dict[str, torch.Tensor]
LossOutput = Dict[str, torch.Tensor]
MetricsOutput = Dict[str, Union[float, Dict]]
```

---

この API リファレンスにより、Stage 1 の全コンポーネントの詳細仕様が網羅されています。開発者は各クラス・関数の正確な使用方法を理解し、効率的に開発・デバッグ・拡張が可能になります。