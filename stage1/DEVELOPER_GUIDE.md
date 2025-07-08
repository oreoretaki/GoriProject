# Stage 1 開発者ガイド

## 🎯 概要

このドキュメントは、Stage 1の自己教師ありマルチタイムフレーム再構築モデルの技術的詳細を説明します。

## 🏗️ アーキテクチャ概要

### データフロー
```
Raw OHLC Data → Window Sampling → Feature Engineering → Masking → Model → Loss Calculation
```

### 主要コンポーネント
1. **Window Sampler**: 効率的なベクトル化ウィンドウサンプリング
2. **Feature Engineering**: 6つの特徴量を36チャンネルに変換
3. **Masking**: ランダム連続ブロックマスキング
4. **Model**: 動的アーキテクチャ（CNN + Transformer）
5. **Loss Functions**: 4種類の損失関数の組み合わせ

## 📊 データ処理

### ウィンドウサンプリング (`src/window_sampler.py`)
- **目的**: 異なるタイムフレームの時系列データを効率的にサンプリング
- **手法**: 右端整列、3.3時間ウィンドウ
- **最適化**: ベクトル化により22倍高速化達成

#### 主要パラメータ
- `seq_len`: シーケンス長（推奨：48）
- `timeframes`: [M1, M5, M15, H1, H4, D]
- `window_hours`: 3.3時間

### 特徴量エンジニアリング (`src/feature_engineering.py`)
- **入力**: OHLC データ
- **出力**: 6つの特徴量 × 6つのタイムフレーム = 36チャンネル
- **特徴量**: 
  - Open, High, Low, Close（正規化済み）
  - Delta Close（変化率）
  - Body Percentage（実体の割合）

### マスキング戦略 (`src/masking.py`)
- **手法**: ランダム連続ブロックマスキング
- **パラメータ**: 
  - マスク率: 15%
  - ブロック長: 5-60バー
  - タイムフレーム間同期

## 🧠 モデルアーキテクチャ

### 動的モデル (`src/model.py`)
```python
class Stage1Model(nn.Module):
    def __init__(self, config):
        # TF固有CNN
        self.tf_cnns = nn.ModuleList([
            CNNBlock(config) for _ in range(n_timeframes)
        ])
        
        # 共有エンコーダー
        self.shared_encoder = TransformerEncoder(config)
        
        # 動的Bottleneck
        self.dynamic_bottleneck = DynamicBottleneck(config)
        
        # TF別デコーダー
        self.tf_decoders = nn.ModuleList([
            Decoder(config) for _ in range(n_timeframes)
        ])
```

#### 主要特徴
- **TF固有CNN**: 各タイムフレームに特化した特徴抽出
- **共有エンコーダー**: クロスタイムフレーム情報の学習
- **動的Bottleneck**: 情報圧縮と次元削減
- **TF別デコーダー**: タイムフレーム固有の再構築

## 📈 損失関数

### 4種類の損失関数 (`src/losses.py`)

#### 1. Huber Loss
```python
huber_loss = F.smooth_l1_loss(reconstructed, targets, reduction='none')
```
- **目的**: 基本的な再構築誤差
- **特徴**: 外れ値に対してロバスト

#### 2. STFT Loss
```python
stft_loss = compute_stft_loss(reconstructed, targets)
```
- **目的**: 周波数ドメインでの整合性
- **特徴**: スペクトラム特性の保持

#### 3. Cross-Timeframe Consistency Loss
```python
consistency_loss = compute_consistency_loss(reconstructed)
```
- **目的**: タイムフレーム間の整合性
- **特徴**: 異なるタイムフレームの予測の一貫性

#### 4. Amplitude-Phase Correlation Loss
```python
correlation_loss = compute_correlation_loss(reconstructed, targets)
```
- **目的**: 振幅と位相の相関保持
- **特徴**: 時系列の構造的特徴の保持

## ⚡ 高速化最適化

### データローダー最適化
- **num_workers**: 8（I/O並列化）
- **prefetch_factor**: 4（プリフェッチ）
- **persistent_workers**: True（ワーカー再利用）

### 学習最適化
- **precision**: "16-mixed"（Tensor Core活用）
- **accumulate_grad_batches**: 2（勾配蓄積）
- **batch_size**: 64（メモリ効率）

### キャッシュ機能
- **ウィンドウキャッシュ**: インデックス保存により20秒→数百ms
- **統計キャッシュ**: 正規化統計の事前計算

## 🔧 設定ファイル

### configs/test.yaml（推奨）
```yaml
data:
  seq_len: 48
  n_timeframes: 6
  n_features: 6
  
training:
  batch_size: 64
  precision: "16-mixed"
  accumulate_grad_batches: 2
  
dataloader:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true
```

### configs/base.yaml（基本設定）
```yaml
data:
  seq_len: 64
  n_timeframes: 6
  n_features: 6
  
training:
  batch_size: 32
  precision: "32-true"
  accumulate_grad_batches: 1
```

## 🧪 テスト・デバッグ

### 開発用ツール（old/ディレクトリ）

#### import_test.py
```bash
python3 old/import_test.py
```
- 基本的なインポートチェック
- ファイル構造確認

#### test_components.py
```bash
python3 old/test_components.py
```
- 各コンポーネントの動作確認
- 高速化性能テスト
- メモリ使用量チェック

#### syntax_check.py
```bash
python3 old/syntax_check.py
```
- Python文法チェック
- AST解析による構文検証

## 🚀 パフォーマンス指標

### 目標値
- **学習速度**: 7-9 it/s（5-7倍高速化）
- **相関係数**: >0.8（val_correlation_mean）
- **TF別性能**: 全タイムフレーム>0.7
- **収束時間**: 40エポック以内

### 監視メトリクス
- `train_total`: 総合訓練損失
- `val_total`: 総合検証損失
- `val_correlation_mean`: 平均相関係数
- `val_corr_m1`, `val_corr_m5`, etc.: TF別相関係数

## 🔍 トラブルシューティング

### よくある問題

#### 1. メモリ不足
```bash
# バッチサイズを減らす
--batch_size 32

# ワーカー数を減らす
--num_workers 4
```

#### 2. 学習速度が遅い
```bash
# 高速化設定を使用
--config configs/test.yaml

# 精度を下げる
--precision 16-mixed
```

#### 3. 収束しない
```bash
# 学習率を調整
--max_lr 1e-4

# 損失重みを調整
--loss_weights [1.0, 0.5, 0.3, 0.2]
```

### デバッグ手順
1. `old/import_test.py`でBasic動作確認
2. `old/test_components.py`でコンポーネント確認
3. `fast_dev_run`で1バッチテスト
4. 短期間学習（3エポック）で動作確認
5. 本格学習開始

## 📝 コード品質

### 規約
- **命名規則**: snake_case（Python標準）
- **ドキュメント**: 全関数にdocstring
- **型ヒント**: 可能な限り使用
- **エラーハンドリング**: 適切な例外処理

### テスト方針
- **単体テスト**: 各コンポーネント
- **統合テスト**: データパイプライン全体
- **性能テスト**: 高速化効果確認

## 🔄 拡張方針

### 新機能追加
1. 新しい損失関数の追加
2. 異なるマスキング戦略の実装
3. 新しいアーキテクチャの試行

### 最適化方向
1. さらなる高速化（分散学習など）
2. メモリ効率の改善
3. モデル圧縮手法の導入

## 📚 参考資料

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Transformers Architecture](https://arxiv.org/abs/1706.03762)
- [Self-Supervised Learning](https://arxiv.org/abs/2006.08217)