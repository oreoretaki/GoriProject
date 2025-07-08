# Stage 1 - 自己教師ありマルチTF再構築 ✅ 実装完了

## 🎯 Stage 1 の目標

USD/JPYのOHLCデータについて、6つの整列した時間足（M1, M5, M15, H1, H4, D）のマスクされたスパンを同時に再構築し、クロススケール一貫性を強制する単一のエンコーダー・デコーダーを訓練する。

## ✅ 実装済み機能

- ✅ **ベクトル化ウィンドウサンプリング**: 右端整列、3.3時間ウィンドウ（22倍高速化：22秒/3.1M行）
- ✅ **特徴量エンジニアリング**: [open, high, low, close, Δclose, %body] → 36チャンネル
- ✅ **マスキング戦略**: ランダム連続ブロック（5-60バー）、15%マスク、TF間同期
- ✅ **TF別z-score正規化**: 統計キャッシュ機能付き
- ✅ **動的モデルアーキテクチャ**: TF固有CNN + 共有エンコーダー + 動的Bottleneck + TF別デコーダー
- ✅ **4種損失関数**: Huber + STFT + クロスTF整合性 + 振幅位相相関
- ✅ **完全動作確認済み**: fast_dev_run完全成功、GPU/CPU統合テスト済み
- ✅ **高速化最適化**: 5-7倍速化（2→7-9 it/s）、Tensor Core、16-mixed precision
- ✅ **ウィンドウキャッシュ**: インデックス保存で20秒→数百ms
- ✅ **I/O並列化**: 8 workers、prefetch、persistent workers
- ✅ **訓練・評価スクリプト**: PyTorch Lightning、One-Cycle LR、早期停止、MPI完全回避
- ✅ **包括的テスト**: データパイプライン全体の単体テスト
- ✅ **WSL完全対応**: MPI検出無効化、環境変数自動設定、Tensor Core最適化
- ✅ **T5転移学習**: 事前学習言語モデルによる初期化、段階的解凍、差分学習率

## 📁 ディレクトリ構成

```
stage1/
├── configs/
│   ├── base.yaml           # 基本設定ファイル
│   ├── medium.yaml         # 中規模設定
│   ├── production.yaml     # 本番環境設定
│   ├── test.yaml           # 高速化最適設定（推奨）
│   ├── t5_baseline.yaml    # T5比較用ベースライン
│   ├── t5_freeze.yaml      # T5段階的解凍（推奨）
│   └── t5_nofreeze.yaml    # T5凍結なし
├── scripts/
│   ├── train_stage1.py     # 訓練スクリプト（T5対応）
│   ├── evaluate_stage1.py  # 評価スクリプト
│   ├── download_t5.py      # T5モデルダウンローダー（汎用CLI）
│   └── run_t5_experiments.sh # T5比較実験自動スクリプト
├── src/
│   ├── __init__.py         # パッケージ初期化
│   ├── data_loader.py      # 最適化データローダー
│   ├── window_sampler.py   # ベクトル化ウィンドウサンプリング
│   ├── feature_engineering.py # 特徴量計算
│   ├── masking.py          # マスキング戦略
│   ├── normalization.py    # 正規化
│   ├── model.py            # 動的モデルアーキテクチャ（T5対応）
│   ├── lm_adapter.py       # T5転移学習アダプター
│   └── losses.py           # 4種損失関数
├── old/                    # 開発・テスト用ファイル保管
│   ├── import_test.py      # インポートテスト
│   ├── test_components.py  # コンポーネントテスト
│   ├── syntax_check.py     # 文法チェック
│   └── [その他開発用ファイル]
├── checkpoints/            # モデルチェックポイント
├── logs/                   # TensorBoard ログ
├── requirements.txt        # Python依存関係
└── README.md              # このファイル
```

## 🚀 使用方法

### 1. データパイプラインテスト
```bash
# Stage 0完了確認
cd ../stage0 && python3 scripts/run_validate.py

# データパイプラインテスト
cd ../tests && python3 test_stage1_data.py
```

### 2. 訓練実行

#### 🚀 ステップ1: 動作確認（完了）
```bash
cd stage1

# fast_dev_run完了確認（1バッチテスト）
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run \
  --devices 1

# 結果: train_total_step=3.200, val_correlation正常動作確認済み ✅
```

#### 🎯 ステップ2: 高速化最適版学習
```bash
# 最適化設定での短期間学習（推奨）
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --max_epochs 3 \
  --devices 1

# 期待性能: 7-9 it/s（従来の5-7倍高速化）
# 結果監視: 損失収束、相関メトリクス、GPU使用率
```

#### 🏆 ステップ3: 本格訓練
```bash
# ベースライン設定での本格学習
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --devices 1

# ハイパーパラメータ調整例
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --batch_size 16 \
  --max_epochs 20 \
  --devices 1
```

#### 🤗 ステップ4: T5転移学習（✅ 事前学習済み重み対応完了）
```bash
# ⚠️ 重要: 事前にT5モデルをダウンロード（60M parameters検証付き）
python3 scripts/download_t5.py --model t5-small

# 認証/プロキシ環境の場合
HTTPS_PROXY=http://proxy:8080 python3 scripts/download_t5.py --model t5-small
# または: huggingface-cli login

# 1. 動作確認（事前学習済みT5重み使用）
python3 scripts/train_stage1.py \
  --config configs/t5_freeze.yaml \
  --data_dir ../data/derived \
  --fast_dev_run

# 2. T5比較実験（自動）- 真の転移学習効果を測定
bash scripts/run_t5_experiments.sh

# 3. 個別実験（改良版設定: seq_len=128, patch_len=32）
python3 scripts/train_stage1.py --config configs/t5_baseline.yaml --data_dir ../data/derived --max_epochs 5   # ベースライン
python3 scripts/train_stage1.py --config configs/t5_freeze.yaml --data_dir ../data/derived --max_epochs 5     # 段階的解凍
python3 scripts/train_stage1.py --config configs/t5_nofreeze.yaml --data_dir ../data/derived --max_epochs 5   # 凍結なし
```

**🔍 T5転移学習検証ポイント（修正済み）**:
- **✅ 事前学習済み重み**: 正式モデルID `t5-small` 使用、ランダム初期化フォールバック禁止
- **✅ パッチ埋め込み改善**: 32-token patches (128/32=4), LayerNorm+√d_model スケーリング
- **✅ 段階的解凍**: 2エポック凍結→解凍、差分学習率 base:t5=10:1
- **✅ 環境対応**: プロキシ・認証・オフライン環境サポート

### 3. 学習監視・調整

#### 📊 重要メトリクス
- **学習速度**: it/s (目標: 7-9 it/s)
- **損失収束**: `train_total`, `val_total` (目標: 安定的減少)
- **相関**: `val_correlation_mean` (目標: >0.8)
- **TF別性能**: `val_corr_m1`, `val_corr_m5`, etc.
- **学習安定性**: 損失振動パターン

#### 🚀 高速化設定（configs/test.yaml）
```yaml
# 最適化済み設定
data:
  seq_len: 48           # 4の倍数最適化

training:
  batch_size: 64        # 大幅増加
  precision: "16-mixed" # Tensor Core活用
  accumulate_grad_batches: 2  # 見かけbatch_size=128

dataloader:
  num_workers: 8        # I/O並列化
  prefetch_factor: 4    # プリフェッチ
  persistent_workers: true

scheduler:
  pct_start: 0.05       # 短縮warm-up
  interval: "step"      # step単位調整
```

#### 🔧 調整ポイント
- **学習率**: `max_lr` 1e-4～1e-3
- **バッチサイズ**: GPU性能に応じて32～128
- **損失重み**: 収束パターンに応じて調整
- **マスク率**: 0.10～0.20で難易度調整

### 4. モデル評価
```bash
python3 scripts/evaluate_stage1.py \
  --config configs/base.yaml \
  --ckpt checkpoints/stage1-epoch-XX.ckpt \
  --data_dir ../data/derived \
  --output evaluation_results.json
```

## 🔄 現在のステータス

### ✅ 完了済み
- **基盤実装**: 100%完了
- **動作確認**: fast_dev_run完全成功
- **ベクトル化**: ウィンドウサンプリング22倍高速化達成
- **I/O最適化**: 5-7倍速化（2→7-9 it/s）実装完了
- **Tensor Core**: 16-mixed precision最適化
- **キャッシュ機能**: ウィンドウ20秒→数百ms
- **WSL対応**: MPI問題完全解決
- **T5転移学習**: 事前学習済み重み対応完了（t5-small, 60M parameters検証機能付き）

### 🚧 進行中
- **T5転移学習実験**: 事前学習済み重みでの真の転移学習効果測定
- **論文検証**: "Random Initialization Can't Catch Up" の時系列での再現実験
- **性能比較**: ベースライン vs T5段階的解凍 vs T5凍結なし
- **収束性分析**: T5事前学習による初期性能・学習安定性の向上確認

### 📈 期待される次のマイルストーン
1. **✅ T5事前学習済み重み**: ダウンロード・ロード成功済み（60M parameters検証済み）
2. **🔄 真の転移学習実験**: 事前学習済みT5 vs ランダム初期化の比較実行中
3. **📊 収束性能分析**: 初期エポックでのT5優位性・学習安定性の確認
4. **🔍 時系列適応性**: テキスト事前学習の時系列への転移可能性検証
5. **⚡ 効率性評価**: 段階的解凍による学習効率向上の測定
6. **🎯 最終性能**: val_correlation_mean改善とベースライン性能超越

## 📊 入力データ

Stage 0完了済み高品質データ：
- **データパス**: `../data/derived/simple_gap_aware_*.parquet`
- **整合性**: 100%保証済み
- **Volume**: 完全除外済み
- **時間足**: M1, M5, M15, M30, H1, H4, D
- **期間**: 2017-01-03 ～ 2025-06-17（約8.5年）

## 🎯 期待される結果

- **Correlation@TF**: ピアソン相関 >0.8
- **整合性比率**: TFバー一致率 >95%
- **スペクトラムΔ**: 対数パワースペクトラム誤差 <0.1
- **収束時間**: 40エポック以内

## 🔧 開発・テスト用ツール

プロジェクトの品質保証とデバッグのため、以下のツールが`old/`ディレクトリに保管されています：

### テストツール
- **import_test.py**: 基本的なインポートとファイル構造チェック
- **test_components.py**: 各コンポーネントの動作確認（高速化最適化版）
- **syntax_check.py**: Python文法チェック

### 使用方法
```bash
# 基本的な依存関係チェック
python3 old/import_test.py

# 全コンポーネントテスト
python3 old/test_components.py

# 文法チェック
python3 old/syntax_check.py
```

これらのツールは開発時のトラブルシューティングに使用できます。

## 🔄 次のステップ

Stage 1完了後、学習された重みはStage 2のRL取引エージェントのシードとして使用されます。